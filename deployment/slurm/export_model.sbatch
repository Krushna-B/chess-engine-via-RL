#!/bin/bash
#SBATCH --job-name=chess-export
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/slurm/%x-%j.out
#SBATCH --error=artifacts/slurm/%x-%j.err

set -euo pipefail

CONFIG_FILE=${CONFIG_FILE:?Set CONFIG_FILE to a Longleaf environment file}
set -a
source "${CONFIG_FILE}"
set +a

cd "${REPO_ROOT}"
export SELFPLAY_DIR="${SELFPLAY_DIR:-${ARTIFACT_ROOT}/selfplay}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ARTIFACT_ROOT}/checkpoints}"
uv run --project training python -m chess_training.export_libtorch
