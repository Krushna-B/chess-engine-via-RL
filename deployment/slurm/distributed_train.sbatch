#!/bin/bash
#SBATCH --job-name=chess-train-ddp
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

set -euo pipefail

CONFIG_FILE=${CONFIG_FILE:?Set CONFIG_FILE to a Longleaf environment file}
set -a
source "${CONFIG_FILE}"
set +a

module load cuda/13.0

cd "${REPO_ROOT}"
export REPO_ROOT ARTIFACT_ROOT
export SELFPLAY_DIR="${SELFPLAY_DIR:-${ARTIFACT_ROOT}/selfplay}"
export METRICS_PATH="${METRICS_PATH:-${ARTIFACT_ROOT}/metrics/train_${SLURM_JOB_ID}.jsonl}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ARTIFACT_ROOT}/checkpoints}"

if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
  apptainer exec --nv \
    --bind "${REPO_ROOT}:${REPO_ROOT},${ARTIFACT_ROOT}:${ARTIFACT_ROOT}" \
    "${APPTAINER_IMAGE}" uv run --project training torchrun \
    --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" \
    -m chess_training.train
else
  uv run --project training torchrun \
    --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" \
    -m chess_training.train
fi
