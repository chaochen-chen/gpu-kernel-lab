# gpu-notes

Notes, kernels, tests and benchmarks from learning GPU programming: *Programming Massively Parallel
Processors* (PMPP), [Liger-Kernel](https://github.com/linkedin/Liger-Kernel), and
[NVIDIA/online-softmax](https://github.com/NVIDIA/online-softmax).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch            # pick the build matching your CUDA version
make install                 # pip install -e ".[dev]"

make test                    # all tests (needs a CUDA GPU)
make test K=triton_online    # only tests matching a name
make bench                   # benchmarks -> CSVs in results/
make lint                    # ruff
make standalone              # build PMPP .cu exercises (needs nvcc)
```

The CUDA softmax is registered only when `nvcc` is on `PATH`; it JIT-compiles on first use.

## Layout

| Path | Purpose |
|------|---------|
| `notes/` | Written notes: `pmpp/`, `liger-kernel/`, `papers/`, `profiling/`, `tracker/` |
| `kernels/<name>/` | Implementations. Each exposes `IMPLS` (dict of impls with a common signature) and `REF` |
| `cuda_standalone/` | Pure `.cu` programs with `main()` (no PyTorch) |
| `tests/` | One generic test per kernel, parametrized over `IMPLS`, shapes, dtypes |
| `benchmarks/` | One benchmark per kernel; CSVs written to `results/` |
| `results/` | Benchmark CSVs, named with the GPU (e.g. `softmax_nvidia-a100-...csv`) |

**Adding a kernel:** create `kernels/<name>/` with `reference.py` and implementations, register them in
`__init__.py`, then copy `tests/test_softmax.py` and `benchmarks/bench_softmax.py`.

## Kernel index

| Kernel | Implementations | Notes |
|--------|-----------------|-------|
| softmax | `triton_naive`, `triton_online`, `cuda_online` (fp32) | [papers](notes/papers/) |
| rmsnorm | `triton` (forward only) | [liger-kernel](notes/liger-kernel/) |

## Benchmark highlights

_Add a table or link to CSVs in `results/` after your first run._

## Progress / TODO

- [x] Repo scaffold, softmax + rmsnorm with tests and benchmarks
- [ ] Run on a GPU, record first results
- [ ] RMSNorm backward, compare against Liger
- [ ] Fill in PMPP ch04-06 notes
- [ ] Cross-entropy / layernorm

See also the [tracker](notes/tracker/) for new architectures, ops and tooling.

