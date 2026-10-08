# Liger-Kernel Benchmark System Design

I'll look at the repo's current benchmark code rather than relying on memory.

Here is a summary of how Liger-Kernel's benchmark system is designed. It is based on the repo's contributing docs and an auto-generated DeepWiki analysis of a recent commit, not on a line-by-line read of the source, so check details against `benchmark/` directly.

## Core idea

The system is a small, convention-driven harness. Each kernel gets one thin script that declares what to measure, and shared utilities handle measurement, sweeps, storage and plotting. The aim is low boilerplate and a consistent way to compare `liger` against `huggingface` (and sometimes `torch_compile`) across model configs and input sizes.

## Layers

### 1. Per-kernel scripts (convention over configuration)

- Each kernel has `benchmark/scripts/benchmark_<kernel>.py`.
- A script defines a `setup_fn` that builds the Liger and baseline modules, inputs and a forward/backward callable, keyed on provider and mode.
- Baselines are reused from the unit-test suite (`test/transformers/test_<kernel>.py`), so correctness references and perf baselines come from the same place.

### 2. Shared utilities (`benchmark/scripts/utils.py`)

- **Typed dataclasses** carry data through the pipeline. `SingleBenchmarkRunInput` holds the x-value, provider and mode (forward, backward, full, no-grad-forward). `SingleBenchmarkRunOutput` holds the median, 20th and 80th percentile results.
- **Measurement primitives.** Speed uses Triton's `do_bench()` with percentiles, so outliers don't skew results. Memory uses peak `max_memory_allocated`.
- **Builders** such as `build_speed_bench_fn(setup_fn)` and `build_memory_bench_fn(setup_fn)` wrap `setup_fn` into runnable benchmark functions.
- **`run_benchmarks()`** is the orchestrator. It loops over x-values, providers and modes, collects results and writes them to CSV.

### 3. Sweep builders (`benchmark_model_configs.py`)

- `build_model_config_sweep` varies model configs (hidden size, vocab, dtype) while keeping total tokens (B×T) roughly constant.
- `build_token_length_sweep` scales one dimension (T, B or B×T) with a fixed model config.
- Scripts choose a mode through CLI flags such as `--sweep-mode`, `--model` and `--bt`.

### 4. Storage: a single flat, append-only CSV

- Each row is one measurement, with columns for kernel name, provider, mode, metric (speed in ms, memory in MB), x-axis name/label/value, the three percentiles, and a JSON `extra_benchmark_config_str`.
- Each row also records the GPU name, timestamp and `liger_version`, so results are traceable.
- Rows are deduplicated by a unique key, and there is an `--overwrite` option.

### 5. Execution and CI

- Benchmarks run on dedicated H100s through **Modal** so the hardware stays consistent.
- A GitHub Actions workflow (`benchmark.yml`) can run on any commit. For an old commit it pulls the latest `benchmark/` folder from `main`, so current scripts run against old code.
- Results are committed to the `gh-pages` branch as `benchmarks/{path}/benchmark.csv`, with a `commits.txt` index. The docs deploy workflow backs up and restores that folder so publishing docs doesn't wipe the data.

### 6. Visualization

- `benchmarks_visualizer.py` reads the CSV and draws matplotlib/seaborn line plots, with percentile bands, into an untracked `benchmark/visualizations/` folder.
- It filters to the most recent GPU if the CSV has several, and supports both sweep modes.

### 7. Backend variants

- For alternative backends (CuTile, CuTe-DSL), a driver script such as `run_cutedsl_compare.py` runs the same benchmark twice with different `LIGER_KERNEL_IMPL` environment variables.
- It tags providers (for example `liger_triton` vs `liger_cutedsl`) so rows don't collide, and writes to separate CSVs.

## Design patterns at a glance

| Pattern | Where it shows up |
|---|---|
| Template method / strategy | Scripts supply only `setup_fn`. The harness owns the loop, timing and storage. |
| Convention over configuration | `benchmark_<kernel>.py` naming, `make run-benchmarks` discovers all scripts. |
| Builder / factory | `build_*_bench_fn` and the sweep builders. |
| Typed data contract | Dataclasses and a fixed CSV schema decouple producers from the visualizer and CI. |
| Append-only log | One CSV with version, GPU and timestamp enables historical tracking. |
| Reproducibility by environment | Modal H100s, percentile statistics, and `liger_version` stamping. |

A new kernel needs one script, and nothing else changes.
