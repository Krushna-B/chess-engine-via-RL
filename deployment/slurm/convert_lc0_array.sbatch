#!/bin/bash
#SBATCH --job-name=lc0-convert
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/slurm/%x-%A_%a.out
#SBATCH --error=artifacts/slurm/%x-%A_%a.err

set -euo pipefail

CONFIG_FILE=${CONFIG_FILE:?Set CONFIG_FILE to a Longleaf environment file}
set -a
source "${CONFIG_FILE}"
set +a

LC0_TAR_LIST=${LC0_TAR_LIST:?Set LC0_TAR_LIST in the configuration}
LC0_SHARD_ROOT=${LC0_SHARD_ROOT:?Set LC0_SHARD_ROOT in the configuration}
LC0_POSITIONS_PER_SHARD=${LC0_POSITIONS_PER_SHARD:-250000}
LC0_SAMPLE_ROOT=${LC0_SAMPLE_ROOT:-}
LC0_SAMPLE_PER_ARCHIVE=${LC0_SAMPLE_PER_ARCHIVE:-5000}

archive=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${LC0_TAR_LIST}")
if [[ -z "${archive}" ]]; then
  echo "No archive for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
echo "Converting ${archive}"
arguments=(
  "${archive}"
  "${LC0_SHARD_ROOT}"
  --positions-per-shard "${LC0_POSITIONS_PER_SHARD}"
  --seed "${SLURM_ARRAY_TASK_ID}"
)
if [[ -n "${LC0_SAMPLE_ROOT}" && "${LC0_SAMPLE_PER_ARCHIVE}" -gt 0 ]]; then
  arguments+=(
    --sample-destination "${LC0_SAMPLE_ROOT}"
    --sample-per-archive "${LC0_SAMPLE_PER_ARCHIVE}"
  )
fi
uv run --project training python -m chess_training.convert_lc0_tar "${arguments[@]}"
