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

For one node with four GPUs and 64 CPU cores:

```bash
sbatch --export=ALL,CONFIG_FILE=$HOME/chess-engine-longleaf.env,GENERATION=1 \
  deployment/slurm/multi_gpu_generation.sbatch
```

This launches four self-play workers. Each worker receives one GPU and 16 CPU
cores, writes a unique shard, and records its own metrics file before the
training and export stages start.

Training uses one full model replica per GPU with DDP. To run training alone:

```bash
sbatch --export=ALL,CONFIG_FILE=$HOME/chess-engine-longleaf.env \
  deployment/slurm/distributed_train.sbatch
```

Required cluster setup:

1. Build the headless ML target with the cluster's LibTorch installation.
2. Create the Python environment with `uv sync --extra tracking` in `training/`.
3. Make `ARTIFACT_ROOT` a persistent project or scratch path and back up
   checkpoints and metrics separately.
4. Set `WANDB_API_KEY` in the job environment or use `WANDB_MODE=offline`.

Set `APPTAINER_IMAGE` to the built `.sif` file to run jobs inside Apptainer.
Leave it empty to use the host environment during initial debugging.

The scripts write shards, checkpoints, immutable model snapshots, and JSONL
metrics under `ARTIFACT_ROOT`.
