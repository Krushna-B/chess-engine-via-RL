# Longleaf Slurm

These jobs are launch templates only. They do not submit themselves.

Copy `longleaf.env.example`, set the repository, artifact, and Lc0 paths, then
load the training environment:

```bash
cp deployment/slurm/longleaf.env.example ~/chess-engine-longleaf.env
sbatch --export=ALL,CONFIG_FILE=$HOME/chess-engine-longleaf.env \
  deployment/slurm/generation.sbatch
```

The generation job runs self-play, training, and TorchScript export in order.
Use `self_play.sbatch` or `train.sbatch` when those stages need to be debugged
independently.

Required cluster setup:

1. Build the headless ML target with the cluster's LibTorch installation.
2. Create the Python environment with `uv sync --extra tracking` in `training/`.
3. Make `ARTIFACT_ROOT` a persistent project or scratch path and back up
   checkpoints and metrics separately.
4. Set `WANDB_API_KEY` in the job environment or use `WANDB_MODE=offline`.

The scripts write shards, checkpoints, immutable model snapshots, and JSONL
metrics under `ARTIFACT_ROOT`.
