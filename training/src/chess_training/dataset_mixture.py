import os
from pathlib import Path

import numpy as np

from chess_training.read_dataset_v2 import (
    CompactTrainingExample,
    iter_compact_shard,
    load_compact_shard,
)


def _resolve_shards(source) -> list[Path]:
    if isinstance(source, (list, tuple)):
        return [Path(path) for path in source]

    path = Path(source)
    if path.is_dir():
        return sorted(path.rglob("*.bin"))

    return [path]


def _load_shard(shard: Path, report: list[dict] | None):
    try:
        return load_compact_shard(shard)
    except (OSError, RuntimeError) as error:
        if os.environ.get("STRICT_SHARDS", "0") == "1":
            raise
        failure = {"path": str(shard), "error": str(error)}
        if report is not None:
            report.append(failure)
        print(f"[dataset] skipping shard {shard}: {error}")
        return []


def load_examples(
    source, report: list[dict] | None = None
) -> list[CompactTrainingExample]:
    shards = _resolve_shards(source)
    if not shards:
        raise FileNotFoundError(f"No shards found for {source}")

    examples = []
    for shard in shards:
        examples.extend(_load_shard(shard, report))
    if not examples:
        raise FileNotFoundError(f"No valid examples found for {source}")
    return examples


def sample_examples(
    source, count: int, seed: int, report: list[dict] | None = None
) -> list[CompactTrainingExample]:
    if count < 1:
        return []

    shards = _resolve_shards(source)
    if not shards:
        raise FileNotFoundError(f"No shards found for {source}")

    generator = np.random.default_rng(seed)
    reservoir = []
    seen = 0
    for shard in shards:
        try:
            for example in iter_compact_shard(shard):
                seen += 1
                if len(reservoir) < count:
                    reservoir.append(example)
                else:
                    index = generator.integers(seen)
                    if index < count:
                        reservoir[index] = example
        except (OSError, RuntimeError) as error:
            if os.environ.get("STRICT_SHARDS", "0") == "1":
                raise
            failure = {"path": str(shard), "error": str(error)}
            if report is not None:
                report.append(failure)
            print(f"[dataset] skipping shard {shard}: {error}")

    if not reservoir:
        raise FileNotFoundError(f"No examples found for {source}")
    return reservoir


def mix_examples(
    lc0_examples: list[CompactTrainingExample],
    self_play_examples: list[CompactTrainingExample],
    lc0_fraction: float,
    examples_per_epoch: int,
    seed: int,
) -> list[CompactTrainingExample]:
    if not 0.0 <= lc0_fraction <= 1.0:
        raise ValueError("lc0_fraction must be between 0 and 1")
    if examples_per_epoch < 1:
        raise ValueError("examples_per_epoch must be positive")
    if lc0_fraction > 0.0 and not lc0_examples:
        raise ValueError("Lc0 examples are required for this mixture")
    if lc0_fraction < 1.0 and not self_play_examples:
        raise ValueError("Self-play examples are required for this mixture")

    lc0_count = round(examples_per_epoch * lc0_fraction)
    self_play_count = examples_per_epoch - lc0_count
    generator = np.random.default_rng(seed)

    def sample(examples, count):
        if count == 0:
            return []
        indices = generator.choice(
            len(examples), size=count, replace=count > len(examples)
        )
        return [examples[index] for index in indices]

    mixed = sample(lc0_examples, lc0_count)
    mixed.extend(sample(self_play_examples, self_play_count))
    generator.shuffle(mixed)
    return mixed


def load_mixed_examples(
    lc0_source,
    self_play_source,
    lc0_fraction: float,
    examples_per_epoch: int,
    seed: int,
    report: list[dict] | None = None,
) -> list[CompactTrainingExample]:
    return mix_examples(
        load_examples(lc0_source, report),
        load_examples(self_play_source, report),
        lc0_fraction,
        examples_per_epoch,
        seed,
    )
