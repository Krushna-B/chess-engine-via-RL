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

## Large Lc0 pretraining

Do not extract every Lc0 game or create one compact shard per game. The bulk
converter streams each tar archive and groups positions into restart-safe large
shards. Create a stable archive list from the repository root:

```bash
find "${ARTIFACT_ROOT}/lc0/raw/test91" -type f -name '*.tar' | sort \
  > "${ARTIFACT_ROOT}/lc0/test91_tar_files.txt"
archive_count=$(wc -l < "${ARTIFACT_ROOT}/lc0/test91_tar_files.txt")
```

Set `LC0_TAR_LIST`, `LC0_SHARD_ROOT`, `LC0_SAMPLE_ROOT`,
`LC0_SAMPLE_PER_ARCHIVE`, and `LC0_POSITIONS_PER_SHARD` in the Longleaf
environment file. The converter keeps the complete compact corpus and also
builds a uniform per-archive reservoir sample. Point `LC0_DATASET_DIR` at the
sample root for fast training startup; the full root remains available for
larger later phases. Submit a bounded array so the parallel filesystem is not
flooded with simultaneous readers:

```bash
mkdir -p artifacts/slurm "${LC0_SHARD_ROOT}"
sbatch --array="0-$((archive_count - 1))%8" \
  --export=ALL,CONFIG_FILE=$HOME/chess-engine-longleaf.env \
  deployment/slurm/convert_lc0_array.sbatch
```

Every archive writes to its own directory and receives `manifest.json` only
after conversion completes. Retrying the same array is safe: completed archives
are skipped and interrupted archives are replaced atomically.

Inspect progress with:

```bash
squeue -u "$USER"
find "${LC0_SHARD_ROOT}" -name manifest.json | wc -l
find "${LC0_SHARD_ROOT}" -name '*.bin' -printf '%s\n' | \
  awk '{ bytes += $1; files += 1 } END { print files, bytes }'
```

The trainer discovers compact shards recursively. For a bounded run, set
`TRAIN_EXAMPLES_PER_EPOCH`; reservoir sampling scans the converted corpus once
on rank zero and writes an atomic shared sample cache. The other DDP ranks load
that cache instead of each rescanning the full corpus. Reusing the same run ID,
generation, sample count, and Lc0 fraction reuses the cache. Pure Lc0 pretraining
no longer requires a placeholder self-play shard.

## GPU and W&B settings

The large-run environment supports configurable per-GPU batch size, loader
workers, pinned memory, TF32, fused AdamW, `torch.compile`, step logging, and
periodic checkpoints. Start with the values in `longleaf.env.example`. If CUDA
runs out of memory, reduce `TRAIN_BATCH_SIZE` from 64 to 32. If W&B shows low
GPU utilization while CPUs are busy, tune `TRAIN_NUM_WORKERS` between 0 and 4;
each DDP rank creates that many workers, so account for total CPU and RAM use.

Authenticate once rather than writing a token into the environment file:

```bash
uv run --project training wandb login
```

Set a stable `WANDB_RUN_ID` and `TRAIN_RESUME=1`. `latest.pt` contains model,
optimizer, step, best validation loss, and random-number states. Resubmitting
the same job resumes the checkpoint and W&B run. Use `WANDB_MODE=offline` when
compute nodes cannot reach W&B, then run `wandb sync` later.

For one-node, four-GPU training:

```bash
set -a
source "$HOME/chess-engine-longleaf.env"
set +a
mkdir -p artifacts/slurm
sbatch --export=ALL,CONFIG_FILE=$HOME/chess-engine-longleaf.env \
  deployment/slurm/distributed_train.sbatch
```

Use distinct configuration files and W&B run IDs for curriculum phases. A
conservative sequence is 100% Lc0, then 90/10, 75/25, 50/50, and finally
20/80 Lc0/self-play. Change `LC0_FRACTION` only after enough self-play shards
exist to supply the requested share. Keep `WARM_START=1` between phases.

Export a completed checkpoint for C++ self-play without occupying a GPU:

```bash
sbatch --export=ALL,CONFIG_FILE=$HOME/chess-engine-longleaf.env \
  deployment/slurm/export_model.sbatch
```
