"""
Each iteration (generation):
  1. C++ self_play generates a new shard using the current frozen model
  2. train.py trains on the replay buffer (recent shards), warm-started from best_model.pt
  3. export_libtorch.py re-exports best_model.pt for the next round of self play

Run:  uv run python -m chess_training.run_loop
"""

import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from chess_training.chess_model import ChessTransformer

REPO_ROOT = Path(__file__).resolve().parents[3]
SELF_PLAY_BIN = REPO_ROOT / "build" / "self_play"
SELFPLAY_DIR = REPO_ROOT / "artifacts" / "selfplay"
METRICS_DIR = REPO_ROOT / "artifacts" / "metrics"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
# Immutable per-generation snapshots -> the strength ladder for evaluation.
MODELS_DIR = REPO_ROOT / "artifacts" / "models"
JIT_MODEL = CHECKPOINT_DIR / "chess_model_jit.pt"

ITERATIONS = int(os.environ.get("LOOP_ITERATIONS", "10"))
# Keep at most this many shards on disk (matches train.py's replay window)
REPLAY_WINDOW = int(os.environ.get("REPLAY_WINDOW", "20"))


def configure_run() -> str:
    """Give this run its own id + metrics file so runs stay separable.

    RUN_ID can be set explicitly; otherwise it defaults to a timestamp. Both
    self_play (C++) and train.py read RUN_ID + METRICS_PATH from the env.
    """
    run_id = os.environ.setdefault(
        "RUN_ID", datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    os.environ["METRICS_PATH"] = str(METRICS_DIR / f"{run_id}.jsonl")
    print(f"[run] RUN_ID={run_id} -> {os.environ['METRICS_PATH']}", flush=True)
    return run_id


def ensure_initial_model() -> None:
    """Cold start: trace a randomly-initialized net so generation 1 has a model"""
    if JIT_MODEL.exists():
        return

    JIT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    # TransformerEncoder's fastpath breaks TorchScript tracing
    torch.backends.mha.set_fastpath_enabled(False)

    model = ChessTransformer().eval()
    example_input = torch.zeros(1, 64, 18)

    # NOTE: no torch.jit.freeze -- frozen weights become CONSTANTS that don't
    # move with module.to(cuda) in C++, causing a cpu/cuda device mismatch.
    traced = torch.jit.trace(model, example_input, check_trace=False)
    traced.save(str(JIT_MODEL))

    print(f"[bootstrap] wrote random initial model -> {JIT_MODEL}")


def next_generation_index() -> int:
    """Continue numbering from existing shards so a resumed Volume keeps goin."""
    existing = sorted(SELFPLAY_DIR.glob("neural_selfplay_shard_*.bin"))
    if not existing:
        return 1
    return int(existing[-1].stem.split("_")[-1]) + 1


def prune_old_shards() -> None:
    shards = sorted(SELFPLAY_DIR.glob("neural_selfplay_shard_*.bin"))
    for stale in shards[:-REPLAY_WINDOW]:
        stale.unlink()


def run(cmd: list) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"[run] {printable}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_generation(generation: int) -> None:
    shard_path = SELFPLAY_DIR / f"neural_selfplay_shard_{generation:04d}.bin"

    # Self-play writes this generation's shard (env-configured, absolute path);
    # the generation tag flows into every metrics record.
    os.environ["SELFPLAY_SHARD_PATH"] = str(shard_path)
    os.environ["SELFPLAY_GENERATION"] = str(generation)

    # 1. Generate self-play data with the current frozen model
    run([str(SELF_PLAY_BIN), str(JIT_MODEL)])

    # 2. Train on the replay buffer, warm-started from the previous generation
    run([sys.executable, "-m", "chess_training.train"])

    # 3. Re-export the trained model for the next self-play round
    run([sys.executable, "-m", "chess_training.export_libtorch"])

    # 4. Archive this generation as an immutable snapshot (the strength ladder).
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(JIT_MODEL, MODELS_DIR / f"gen_{generation:04d}_jit.pt")
    weights = CHECKPOINT_DIR / "best_model.pt"
    if weights.exists():
        shutil.copy(weights, MODELS_DIR / f"gen_{generation:04d}.pt")

    # 5. Bound the replay buffer's disk (snapshots in MODELS_DIR are kept).
    prune_old_shards()


def main() -> None:
    if not SELF_PLAY_BIN.exists():
        raise FileNotFoundError(
            f"self_play binary not found: {SELF_PLAY_BIN} (build it first)"
        )

    configure_run()
    ensure_initial_model()

    generation = next_generation_index()
    for _ in range(ITERATIONS):
        print(f"\n===== generation {generation} =====", flush=True)
        run_generation(generation)
        generation += 1


if __name__ == "__main__":
    main()
