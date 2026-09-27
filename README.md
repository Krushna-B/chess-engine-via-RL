# Chess Engine

A C++ chess engine and an experimental AlphaZero-style training pipeline. The
engine supplies chess rules, move generation, Monte Carlo tree search (MCTS),
batched neural-network inference, and UCI executables. The Python package trains
the policy/value network from self-play data and exports it back to LibTorch.

The project is currently a single-machine prototype. Its components are being
separated so self-play, training, evaluation, and model promotion can later run
as independent cluster jobs.

## How the system works

One generation currently runs as a sequential loop:

1. `self_play` loads the current TorchScript model.
2. Concurrent games use MCTS and share a batched inference worker.
3. Finished positions are written to a binary replay shard.
4. Python trains a policy/value transformer on recent shards.
5. The best checkpoint from that training run is exported to TorchScript.
6. The exported model becomes the model used by the next generation.

See [the architecture guide](docs/architecture.md) for the data flow, model
interfaces, and current scaling limitations.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/`, `include/` | C++ engine, MCTS, inference, UCI, and self-play code |
| `training/` | Installable Python training package and locked dependencies |
| `benchmarks/` | Evaluation commands, match history, and generated output |
| `deployment/` | Remote execution definitions, currently Modal |
| `configs/` | Checked-in configuration examples |
| `tools/` | Standalone C++ development utilities |
| `tests/` | C++ test and diagnostic programs |
| `scripts/` | General repository maintenance commands |
| `assets/` | GUI piece images and fonts |
| `data/` | Small, required engine lookup data |
| `external/` | Third-party Git submodules, currently Cute Chess |
| `artifacts/` | Generated shards, checkpoints, models, and metrics (ignored) |
| `archive/` | Superseded local artifacts (ignored) |

The ownership and cleanup rules for these paths are documented in
[the repository layout guide](docs/repository-layout.md).

## Build

Initialize third-party submodules first:

```bash
git submodule update --init --recursive
```

For the core engine without the GUI or LibTorch:

```bash
cmake --preset core
cmake --build --preset core
```

The `headless-ml` preset builds self-play and the neural UCI engine. CMake must
be able to find the LibTorch installation. With a Python PyTorch installation:

```bash
cmake --preset headless-ml \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build --preset headless-ml
```

The original `build/` directory remains supported. `BUILD_GUI` and `BUILD_ML`
can each be enabled or disabled independently.

## Verify move generation

`perft` counts legal move-tree leaves. The standard starting position has
197,281 leaves at depth four:

```bash
printf '%s\n%s\n' \
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' \
  '4' | ./build-core/perft
```

## Python environment

The training package uses Python 3.12 and `uv`:

```bash
cd training
uv sync
```

For a small local run, review `configs/local-smoke.env`, export the desired
values, then run from `training/`:

```bash
set -a
source ../configs/local-smoke.env
set +a
uv run chess-training
```

The self-play executable must already exist at `build/self_play`; this path is
currently fixed by the Python orchestrator.

## Generated data

Do not commit local shards or model weights. Runtime outputs belong under
`artifacts/`:

```text
artifacts/
├── selfplay/       binary replay shards
├── checkpoints/    mutable training checkpoints and current TorchScript model
├── models/         immutable generation snapshots
├── metrics/        JSONL run telemetry
└── eval/           models downloaded for local evaluation
```

Benchmark match output is generated under `benchmarks/results/`. Existing
historical results remain in the repository, but new results are ignored.
