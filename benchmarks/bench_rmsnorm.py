"""RMSNorm bandwidth benchmark. Run: python -m benchmarks.bench_rmsnorm"""

import torch
import triton

from benchmarks.utils import RESULTS_DIR, bandwidth_gbps, gpu_name
from kernels.rmsnorm import IMPLS, REF

M = 4096
DTYPE = torch.bfloat16


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(7, 15)],
        x_log=True,
        line_arg="impl",
        line_vals=["torch", *IMPLS],
        line_names=["torch", *IMPLS],
        ylabel="GB/s",
        plot_name=f"rmsnorm_{gpu_name()}",
        args={"M": M},
    )
)
def bench(M, N, impl):
    x = torch.randn(M, N, device="cuda", dtype=DTYPE)
    w = torch.randn(N, device="cuda", dtype=DTYPE)
    fn = REF if impl == "torch" else IMPLS[impl]
    ms = triton.testing.do_bench(lambda: fn(x, w))
    return bandwidth_gbps(2 * x.numel() * x.element_size(), ms)  # read x + write y (w is tiny)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    bench.run(save_path=str(RESULTS_DIR), print_data=True)
