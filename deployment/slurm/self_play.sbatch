#!/bin/bash
#SBATCH --job-name=chess-selfplay
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
export SELFPLAY_SHARD_PATH="${ARTIFACT_ROOT}/selfplay/neural_selfplay_shard_${SLURM_JOB_ID}.bin"
export METRICS_PATH="${ARTIFACT_ROOT}/metrics/selfplay_${SLURM_JOB_ID}.jsonl"

if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
  srun apptainer exec --nv \
    --bind "${REPO_ROOT}:${REPO_ROOT},${ARTIFACT_ROOT}:${ARTIFACT_ROOT}" \
    "${APPTAINER_IMAGE}" "${REPO_ROOT}/build/self_play" \
    "${ARTIFACT_ROOT}/checkpoints/chess_model_jit.pt"
else
  srun "${REPO_ROOT}/build/self_play" \
    "${ARTIFACT_ROOT}/checkpoints/chess_model_jit.pt"
fi
