"""Train on modal GPU's

C++ self_play engine is compiled inside the image against the LibTorch that
ships with the torch
Generated shards and chekcpoint checkpoints live on a persistent Volume mounted at artifacts/, committed after
every iteration
"""

import sys

import modal

REPO = "/root/chess-engine"
PACKAGE_SRC = f"{REPO}/training/src"
ARTIFACTS = f"{REPO}/artifacts"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "cmake", "git")
    .pip_install("torch", "numpy")
    .env({"PYTHONPATH": PACKAGE_SRC})
    .add_local_dir(
        ".",
        remote_path=REPO,
        copy=True,
        ignore=[
            "**/.git",
            "**/build",
            "**/artifacts",
            "**/archive",
            "**/.venv",
            "**/__pycache__",
            "**/*.bin",
            "**/*.pt",
        ],
    )
    .run_commands(
        # Configure headless (no raylib) and point CMake at the wheel's LibTorch
        f"cd {REPO} && cmake -S . -B build -DBUILD_GUI=OFF "
        "-DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') "
        "&& cmake --build build --target self_play -j",
    )
)

app = modal.App("chess-engine-training")

# Persists shards & checkpoints
artifacts_volume = modal.Volume.from_name("chess-artifacts", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    volumes={ARTIFACTS: artifacts_volume},
    timeout=24 * 60 * 60,
)
def train_loop(
    iterations: int = 10,
    games: int = 50,
    simulations: int = 300,
):
    import os

    from chess_training import run_loop

    os.environ["SELFPLAY_GAMES"] = str(games)
    os.environ["SELFPLAY_SIMULATIONS"] = str(simulations)

    run_loop.ensure_initial_model()

    for iteration in range(1, iterations + 1):
        print(f"\n===== iteration {iteration}/{iterations} =====", flush=True)

        run_loop.run([str(run_loop.SELF_PLAY_BIN), str(run_loop.JIT_MODEL)])
        run_loop.run([sys.executable, "-m", "chess_training.train"])
        run_loop.run([sys.executable, "-m", "chess_training.export_libtorch"])

        # Persist this iteration's shard + checkpoints before the next one
        artifacts_volume.commit()


@app.local_entrypoint()
def main(iterations: int = 10, games: int = 50, simulations: int = 800):
    train_loop.remote(
        iterations=iterations,
        games=games,
        simulations=simulations,
    )
