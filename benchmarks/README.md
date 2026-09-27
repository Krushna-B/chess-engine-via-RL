# Benchmarks

This directory contains repeatable engine evaluation commands. Generated PGNs,
logs, and match metadata are written below `results/` and are ignored for new
runs.

From the repository root:

```bash
benchmarks/random_selfplay.sh
benchmarks/run_baseline.sh
MODEL=artifacts/eval/gen_0005_jit.pt benchmarks/run_eval.sh
```

Historical result files already tracked by Git have been left untouched.

