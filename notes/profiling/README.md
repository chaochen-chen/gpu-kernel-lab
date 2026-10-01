# Profiling

## Quick commands

```bash
# Timeline (kernel launches, memcpys, gaps)
nsys profile -o results/trace python -m benchmarks.bench_softmax

# Per-kernel metrics (use --kernel-name to focus; ncu is slow)
ncu --set full -o results/softmax --kernel-name regex:softmax python your_script.py
```

Profiler outputs (`*.nsys-rep`, `*.ncu-rep`) are gitignored.

## Checklist
- Memory-bound or compute-bound? Compare achieved GB/s / TFLOPS to hardware peak.
- Occupancy, register and shared-memory usage per kernel.
- Coalescing and bank conflicts.
