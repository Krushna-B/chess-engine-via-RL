# Architecture

## Current generation loop

```text
                  current TorchScript model
                             │
                             v
  many game threads ──> batched LibTorch inference
          │                  │
          └──── MCTS <───────┘
                    │
                    v
             binary replay shard
                    │
                    v
       Python replay-window dataset
                    │
                    v
       policy loss + value loss
                    │
                    v
         PyTorch weights checkpoint
                    │
                    v
           TorchScript export
                    │
                    └──────────── next generation
```

`training/src/chess_training/run_loop.py` owns this sequence. It starts with a
randomly initialized model if no checkpoint exists, invokes the C++ self-play
binary, invokes Python training, exports the updated model, and archives an
immutable generation snapshot.

## C++ engine

### Chess state and legal moves

`board.cpp`, `move_generator.cpp`, and `move_list.cpp` implement board state,
move application, terminal-state checks, and legal move generation. `perft` is
the primary correctness probe for this layer.

### Neural representation

Each position is encoded from the side-to-move perspective as 64 squares with
18 features per square, producing a `[64, 18]` tensor.

The policy uses the AlphaZero-style 73 move planes over 64 origin squares:

```text
73 × 64 = 4,672 policy actions
```

`neural_net.cpp` owns position and move encoding. The same encoding contract
must be preserved by the C++ producer and Python model.

### MCTS

`mcts.cpp` combines neural policy priors and value predictions with tree search.
During self-play, root Dirichlet noise encourages exploration. Early moves are
sampled from visit counts; later moves choose the highest-visit action.

Each game searches sequentially, but many games run concurrently. Their leaf
evaluations are sent to one shared inference service so the GPU sees batches
instead of individual positions.

### Self-play output

`self_play.cpp` produces training examples containing:

- encoded position: `64 × 18` float32 values;
- MCTS policy target: `4,672` float32 values;
- final game value from the current player's perspective: `-1`, `0`, or `1`.

`training_data.cpp` writes a small header followed by fixed-width records. The
current format is version 1 and Python reads it as little-endian float32 data.
Changing the representation requires a new data-format version.

## Python training

`ChessTransformer` projects each square into a learned embedding, applies a
Transformer encoder, and returns:

- one logit for each of the 4,672 policy actions;
- one scalar value in `[-1, 1]`.

Training minimizes the sum of soft-target policy cross-entropy and value mean
squared error. A replay window selects recent generation shards. Training
normally warm-starts from the previous `best_model.pt`.

`export_libtorch.py` traces the Python model and verifies exported outputs on
real replay positions. The C++ engine loads that TorchScript artifact.

## Executables

| Target | Responsibility |
| --- | --- |
| `main` | Desktop raylib GUI |
| `engine` | Random-move UCI baseline |
| `engine_neural` | Neural policy/MCTS UCI engine |
| `self_play` | Parallel game generation and shard creation |
| `perft` | Move-generation correctness/performance probe |
| `generate_magic` | Magic-bitboard data utility |
| `torch_model_test` | Basic LibTorch loading/inference probe |

## Runtime outputs

The pipeline treats `artifacts/` as runtime state rather than source code:

- `selfplay/`: replay data;
- `checkpoints/`: mutable current state;
- `models/`: immutable generation history;
- `metrics/`: JSONL configuration, game, inference, and training records;
- `eval/`: locally downloaded evaluation candidates.

Modal mounts persistent storage at this same path, allowing generations to
continue across remote function invocations.

## Known scaling boundaries

These are architectural constraints to address after repository cleanup:

1. A self-play process retains all completed games in memory and writes its
   shard only after the full job completes.
2. Python loads and concatenates every selected replay shard into RAM.
3. Shared generation numbering and shard names do not support multiple writers.
4. The orchestrator runs self-play, training, export, and archival sequentially
   in one remote function.
5. A newly trained model automatically replaces the previous model; there is
   no candidate-versus-champion promotion gate.
6. Random position splitting can put positions from the same game in training
   and validation sets.
7. Configuration is distributed across C++ environment reads, Python module
   constants, shell variables, and Modal function parameters.
8. The shard format has no per-record checksum, game boundary, worker identity,
   or recovery mechanism for interrupted writes.

These issues should be fixed as part of cluster preparation, not hidden inside
the repository-layout cleanup.

