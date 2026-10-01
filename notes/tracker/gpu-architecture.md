# GPU architecture timeline (kernel-author view)

> Summary from memory - verify details against NVIDIA's architecture whitepapers and the CUDA
> Programming Guide before relying on them.

| Arch | Example GPU | Key features for kernel authors | What changed in programming |
|------|-------------|---------------------------------|-----------------------------|
| Volta | V100 | First Tensor Cores (FP16) | `wmma` / MMA fragments |
| Ampere | A100 | BF16/TF32 Tensor Cores, `cp.async` | Async global->shared copies, software pipelining |
| Hopper | H100 | TMA, WGMMA, thread block clusters, FP8 | Warp-group MMA, producer/consumer pipelines |
| Blackwell | B200 | 5th-gen Tensor Cores, tensor memory (TMEM), FP4/FP6 | New MMA path and memory space |
