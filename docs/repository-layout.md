# Repository layout and ownership

The repository separates durable source code from reproducible runtime output.
This distinction becomes important when local development expands to ephemeral
cluster workers and shared object storage.

## Source-controlled paths

| Path | Owner | Notes |
| --- | --- | --- |
| `src/`, `include/` | C++ engine | Engine behavior and executable entry points |
| `training/` | Python training | Package source, metadata, and lockfile |
| `scripts/` | Developer tooling | Small repeatable local commands |
| `benchmarks/` | Evaluation | Match runners and historical results |
| `deployment/` | Cluster execution | Modal now; other schedulers can be added later |
| `configs/` | Configuration | Checked-in examples, never secrets |
| `tools/` | C++ utilities | Standalone development programs |
| `tests/` | Verification | Test and diagnostic entry points |
| `docs/` | Project documentation | Architecture, contracts, and operations |
| `assets/` | GUI | Small resources needed at runtime |
| `data/` | Engine | Small deterministic lookup data |
| `external/` | Third parties | Git submodules; do not edit as first-party code |
| `CMakeLists.txt` | Build system | Target definitions and dependency boundaries |
| `CMakePresets.json` | Build system | Named reproducible local/CI configurations |
| `deployment/modal_app.py` | Deployment | Current Modal image and remote entry points |

## Generated paths

These paths are local or remote runtime state and should not receive new
source-controlled files:

| Path | Contents | Retention policy |
| --- | --- | --- |
| `build*/` | Compilers, objects, and executables | Rebuild at any time |
| `artifacts/selfplay/` | Replay shards | Bounded by replay-window policy |
| `artifacts/checkpoints/` | Mutable current checkpoints | Keep latest recoverable state |
| `artifacts/models/` | Immutable model snapshots | Keep for evaluation/lineage |
| `artifacts/metrics/` | JSONL telemetry | Persist per run |
| `artifacts/eval/` | Downloaded model copies | Re-download when needed |
| `benchmarks/results/` | PGNs and match logs | Store externally for large runs |
| `archive/` | Superseded local experiments | Migrate or remove deliberately |

Existing benchmark results already tracked by Git are historical data and have
not been removed. The ignore rules prevent accidental addition of future run
output.

## Root-directory policy

The root should contain only project entry points and cross-cutting metadata.
New experimental scripts, PGNs, model weights, logs, or datasets should not be
placed there.

Recommended destinations:

| New item | Destination |
| --- | --- |
| C++ implementation/header | `src/` or `include/` |
| Python training module | `training/src/chess_training/` |
| Documentation | `docs/` |
| Reusable command | `scripts/` |
| Model or replay data | `artifacts/` |
| Evaluation output | `benchmarks/results/` |
| Third-party source | Git submodule under `external/` |

## Naming rules for future distributed output

The current names are generation-based and assume a single writer. Cluster
output should eventually include a run, generation, and unique worker/shard ID:

```text
artifacts/selfplay/<run-id>/generation-000042/
  worker-0007/shard-<uuid>.bin
```

Checkpoints should similarly distinguish candidates from promoted champions:

```text
artifacts/models/<run-id>/
  candidates/generation-000042.pt
  champion/current.pt
  champion/history/generation-000037.pt
```

This is a future storage contract, not the current implementation.
