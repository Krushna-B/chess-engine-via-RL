#!/bin/bash
#SBATCH --job-name=chess-generation
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

set -euo pipefail

CONFIG_FILE=${CONFIG_FILE:?Set CONFIG_FILE to a Longleaf environment file}
source "${CONFIG_FILE}"

cd "${REPO_ROOT}"
mkdir -p "${ARTIFACT_ROOT}/slurm"
export REPO_ROOT ARTIFACT_ROOT

export SELFPLAY_DIR="${ARTIFACT_ROOT}/selfplay"
export METRICS_DIR="${ARTIFACT_ROOT}/metrics"
export CHECKPOINT_DIR="${ARTIFACT_ROOT}/checkpoints"
export MODELS_DIR="${ARTIFACT_ROOT}/models"
export GENERATION=${GENERATION:-0}

srun --cpu-bind=cores bash -c '
  worker_id=${SLURM_PROCID}
  export SELFPLAY_WORKER_ID="${worker_id}"
  export SELFPLAY_SHARD_PATH="${ARTIFACT_ROOT}/selfplay/gen_${GENERATION}_worker_${worker_id}.bin"
  export METRICS_PATH="${ARTIFACT_ROOT}/metrics/gen_${GENERATION}_worker_${worker_id}.jsonl"
  exec "${REPO_ROOT}/build/self_play" \
    "${ARTIFACT_ROOT}/checkpoints/chess_model_jit.pt"
'

uv run --project training torchrun \
  --standalone \
  --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" \
  -m chess_training.train
uv run --project training python -m chess_training.export_libtorch
