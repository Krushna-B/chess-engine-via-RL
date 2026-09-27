#!/bin/bash
#SBATCH --job-name=chess-generation
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

set -euo pipefail

CONFIG_FILE=${CONFIG_FILE:?Set CONFIG_FILE to a Longleaf environment file}
source "${CONFIG_FILE}"

cd "${REPO_ROOT}"
mkdir -p "${ARTIFACT_ROOT}/slurm"

export SELFPLAY_DIR="${ARTIFACT_ROOT}/selfplay"
export METRICS_DIR="${ARTIFACT_ROOT}/metrics"
export CHECKPOINT_DIR="${ARTIFACT_ROOT}/checkpoints"
export MODELS_DIR="${ARTIFACT_ROOT}/models"

if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
  srun apptainer exec --nv \
    --bind "${REPO_ROOT}:${REPO_ROOT},${ARTIFACT_ROOT}:${ARTIFACT_ROOT}" \
    "${APPTAINER_IMAGE}" uv run --project training python -m chess_training.run_loop
else
  srun uv run --project training python -m chess_training.run_loop
fi
