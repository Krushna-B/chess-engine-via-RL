#!/bin/bash
#SBATCH --job-name=chess-selfplay
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

set -euo pipefail

CONFIG_FILE=${CONFIG_FILE:?Set CONFIG_FILE to a Longleaf environment file}
set -a
source "${CONFIG_FILE}"
set +a

module load cuda/13.0

cd "${REPO_ROOT}"
mkdir -p "${ARTIFACT_ROOT}/slurm"
export REPO_ROOT ARTIFACT_ROOT APPTAINER_IMAGE

export SELFPLAY_DIR="${SELFPLAY_DIR:-${ARTIFACT_ROOT}/selfplay}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ARTIFACT_ROOT}/checkpoints}"
export GENERATION=${GENERATION:-0}
SELF_PLAY_BIN=${SELF_PLAY_BIN:-${REPO_ROOT}/build-longleaf-cuda13/self_play}
export SELF_PLAY_BIN

srun --cpu-bind=cores bash -c '
  worker_id=${SLURM_PROCID}
  export SELFPLAY_WORKER_ID="${worker_id}"
  export SELFPLAY_SHARD_PATH="${ARTIFACT_ROOT}/selfplay/gen_${GENERATION}_worker_${worker_id}.bin"
  export METRICS_PATH="${ARTIFACT_ROOT}/metrics/gen_${GENERATION}_worker_${worker_id}.jsonl"
  if [[ -n "${APPTAINER_IMAGE:-}" ]]; then
    exec apptainer exec --nv \
      --bind "${REPO_ROOT}:${REPO_ROOT},${ARTIFACT_ROOT}:${ARTIFACT_ROOT}" \
      "${APPTAINER_IMAGE}" "${SELF_PLAY_BIN}" \
      "${CHECKPOINT_DIR}/chess_model_jit.pt"
  else
    exec "${SELF_PLAY_BIN}" \
      "${CHECKPOINT_DIR}/chess_model_jit.pt"
  fi
'
