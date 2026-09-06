"""Train on modal GPU's

C++ self_play engine is compiled inside the image against the LibTorch that
ships with the torch
Generated shards and chekcpoint checkpoints live on a persistent Volume mounted at artifacts/, committed after
every iteration
"""

import modal

REPO = "/root/chess-engine"
PACKAGE_SRC = f"{REPO}/training/src"
ARTIFACTS = f"{REPO}/artifacts"

image = (
    # CUDA *devel* base so the CUDA toolkit (nvcc + libs) is present at build
    # time -- required for find_package(Torch) to configure the C++ engine.
    # torch is pinned to the matching CUDA (cu124) so versions line up.
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("build-essential", "git")
    # Modern CMake via pip -- the base image's apt cmake is 3.22, project needs >=3.23.
    .pip_install("numpy", "cmake")
    .pip_install(
        "torch==2.5.1", index_url="https://download.pytorch.org/whl/cu124"
    )
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
    cpu=16.0,  # self-play MCTS is CPU-bound; give the worker threads real cores
    memory=32768,
    volumes={ARTIFACTS: artifacts_volume},
    timeout=24 * 60 * 60,
)
def train_loop(
    iterations: int = 10,
    games: int = 2000,
    simulations: int = 800,
    max_plays: int = 512,
    concurrency: int = 512,
    run_id: str = "",
):
    import os

    from chess_training import run_loop

    if run_id:
        os.environ["RUN_ID"] = run_id
    os.environ["SELFPLAY_GAMES"] = str(games)
    os.environ["SELFPLAY_SIMULATIONS"] = str(simulations)
    os.environ["SELFPLAY_MAX_PLAYS"] = str(max_plays)
    os.environ["SELFPLAY_CONCURRENCY"] = str(concurrency)

    run_loop.configure_run()
    run_loop.ensure_initial_model()

    generation = run_loop.next_generation_index()
    for _ in range(iterations):
        print(f"\n===== generation {generation} =====", flush=True)
        run_loop.run_generation(generation)
        generation += 1

        # Persist this generation's shard + checkpoints before the next one
        artifacts_volume.commit()


@app.local_entrypoint()
def main(
    iterations: int = 10,
    games: int = 2000,
    simulations: int = 800,
    max_plays: int = 512,
    concurrency: int = 512,
    run_id: str = "",
):
    # spawn (not remote) so the entrypoint returns immediately -- there's no
    # blocking client to cancel on Ctrl-C. Requires --detach, otherwise the
    # ephemeral app tears down and cancels the call when the entrypoint returns.
    call = train_loop.spawn(
        iterations=iterations,
        games=games,
        simulations=simulations,
        max_plays=max_plays,
        concurrency=concurrency,
        run_id=run_id,
    )
    print(f"Spawned train_loop -> call {call.object_id}")
    print("Follow logs: modal app logs chess-engine-training")


# Pulling a checkpoint off the Volume doesn't need the CUDA/C++ build image.
download_image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=download_image, volumes={ARTIFACTS: artifacts_volume})
def _read_artifact(rel_path: str) -> bytes:
    from pathlib import Path

    return (Path(ARTIFACTS) / rel_path).read_bytes()


@app.function(image=download_image, volumes={ARTIFACTS: artifacts_volume})
def _list_models() -> list:
    from pathlib import Path

    models = Path(ARTIFACTS) / "models"
    return sorted(p.name for p in models.glob("*.pt")) if models.exists() else []


@app.local_entrypoint()
def list_checkpoints():
    for name in _list_models.remote():
        print(name)


@app.local_entrypoint()
def download(model: str = "models/gen_0001_jit.pt", dest: str = "artifacts/eval"):
    """Copy one checkpoint from the chess-artifacts Volume to this machine.

    `model` is a path under artifacts/ on the Volume (a traced *_jit.pt is what
    engine_neural loads). Point the eval script at the file this writes.
    """
    from pathlib import Path

    data = _read_artifact.remote(model)
    out = Path(dest) / Path(model).name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"Wrote {len(data)} bytes -> {out}")
