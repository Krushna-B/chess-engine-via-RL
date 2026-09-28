#!/bin/bash
#SBATCH --job-name=chess-selfplay
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

set -euo pipefail

CONFIG_FILE=${CONFIG_FILE:?Set CONFIG_FILE to a Longleaf environment file}
set -a
source "${CONFIG_FILE}"
set +a
module load cuda/13.0

CUDA_ROOT=${CUDA_ROOT:-/nas/longleaf/rhel9/apps/cuda/13.0}
SELF_PLAY_BIN=${SELF_PLAY_BIN:-${REPO_ROOT}/build-longleaf-cuda13/self_play}

cd "${REPO_ROOT}"
mkdir -p "${ARTIFACT_ROOT}/slurm"
export SELFPLAY_SHARD_PATH="${ARTIFACT_ROOT}/selfplay/neural_selfplay_shard_${SLURM_JOB_ID}.bin"
export METRICS_PATH="${ARTIFACT_ROOT}/metrics/selfplay_${SLURM_JOB_ID}.jsonl"

if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
  srun apptainer exec --nv \
    --bind "${REPO_ROOT}:${REPO_ROOT},${ARTIFACT_ROOT}:${ARTIFACT_ROOT},${CUDA_ROOT}:${CUDA_ROOT}" \
    "${APPTAINER_IMAGE}" env \
    "${SELF_PLAY_BIN}" \
    "${ARTIFACT_ROOT}/checkpoints/chess_model_jit.pt"
else
  srun "${SELF_PLAY_BIN}" \
    "${ARTIFACT_ROOT}/checkpoints/chess_model_jit.pt"
fi
