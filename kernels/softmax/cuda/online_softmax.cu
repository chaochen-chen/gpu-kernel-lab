// Online softmax in CUDA (fp32, 2D row-major). One block per row.
// Each thread keeps a (max, denominator) pair; pairs are merged with the online-softmax
// combine rule, first within a warp (shuffles), then across warps (shared memory).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math.h>

struct MD {
    float m;  // running max
    float d;  // running denominator, relative to m
};

__device__ __forceinline__ MD combine(MD a, MD b) {
    MD r;
    r.m = fmaxf(a.m, b.m);
    float ea = (a.m == -INFINITY) ? 0.f : __expf(a.m - r.m);
    float eb = (b.m == -INFINITY) ? 0.f : __expf(b.m - r.m);
    r.d = a.d * ea + b.d * eb;
    return r;
}

__global__ void online_softmax_kernel(const float* __restrict__ x, float* __restrict__ y,
                                      int n_cols) {
    const float* xr = x + (size_t)blockIdx.x * n_cols;
    float* yr = y + (size_t)blockIdx.x * n_cols;

    MD p = {-INFINITY, 0.f};
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        p = combine(p, MD{xr[i], 1.f});
    }

    // Warp-level reduction
    for (int off = 16; off > 0; off >>= 1) {
        MD o = {__shfl_down_sync(0xffffffff, p.m, off), __shfl_down_sync(0xffffffff, p.d, off)};
        p = combine(p, o);
    }

    // Block-level reduction
    __shared__ MD warp_md[32];
    __shared__ MD total;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int n_warps = (blockDim.x + 31) / 32;
    if (lane == 0) warp_md[warp] = p;
    __syncthreads();
    if (warp == 0) {
        p = (lane < n_warps) ? warp_md[lane] : MD{-INFINITY, 0.f};
        for (int off = 16; off > 0; off >>= 1) {
            MD o = {__shfl_down_sync(0xffffffff, p.m, off),
                    __shfl_down_sync(0xffffffff, p.d, off)};
            p = combine(p, o);
        }
        if (lane == 0) total = p;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        yr[i] = __expf(xr[i] - total.m) / total.d;
    }
}

torch::Tensor online_softmax(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only fp32 is supported");
    x = x.contiguous();
    auto y = torch::empty_like(x);
    const int n_rows = x.size(0);
    const int n_cols = x.size(1);
    const int threads = 256;
    online_softmax_kernel<<<n_rows, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n_cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("online_softmax", &online_softmax, "Online softmax (fp32, CUDA)");
}
