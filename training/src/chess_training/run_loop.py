"""
Each iteration:
  1. C++ self_play generates a shard using the current frozen model
  2. train.py trains the best ckeckpoint PyTorch model on that shard
  3. export_libtorch.py re-exports best_model.pt for next round of self play and training

Run:  uv run python -m chess_training.run_loop
"""

import os
import subprocess
import sys
from pathlib import Path

import torch

from chess_training.chess_model import ChessTransformer

REPO_ROOT = Path(__file__).resolve().parents[3]
SELF_PLAY_BIN = REPO_ROOT / "build" / "self_play"
JIT_MODEL = REPO_ROOT / "artifacts" / "checkpoints" / "chess_model_jit.pt"

ITERATIONS = int(os.environ.get("LOOP_ITERATIONS", "10"))


def ensure_initial_model() -> None:
    """Cold start: trace a randomly-initialized net so iteration 1 has a model"""
    if JIT_MODEL.exists():
        return

    JIT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    # TransformerEncoder's fastpath breaks TorchScript tracing
    torch.backends.mha.set_fastpath_enabled(False)

    model = ChessTransformer().eval()
    example_input = torch.zeros(1, 64, 18)

    traced = torch.jit.trace(model, example_input, check_trace=False)
    traced = torch.jit.freeze(traced)
    traced.save(str(JIT_MODEL))

    print(f"[bootstrap] wrote random initial model -> {JIT_MODEL}")


def run(cmd: list) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"[run] {printable}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    if not SELF_PLAY_BIN.exists():
        raise FileNotFoundError(
            f"self_play binary not found: {SELF_PLAY_BIN} (build it first)"
        )

    ensure_initial_model()

    for iteration in range(1, ITERATIONS + 1):
        print(f"\n===== iteration {iteration}/{ITERATIONS} =====", flush=True)

        # 1. Generate self-play data with the current frozen model
        run([str(SELF_PLAY_BIN), str(JIT_MODEL)])

        # 2. Train on the freshly generated shard
        run([sys.executable, "-m", "chess_training.train"])

        # 3. Re-export the trained model for the next self-play round
        run([sys.executable, "-m", "chess_training.export_libtorch"])


if __name__ == "__main__":
    main()
