from pathlib import Path

import numpy as np

from chess_training.read_dataset_v2 import CompactTrainingExample, load_compact_shard


def _resolve_shards(source) -> list[Path]:
    if isinstance(source, (list, tuple)):
        return [Path(path) for path in source]

    path = Path(source)
    if path.is_dir():
        return sorted(path.glob("*.bin"))

    return [path]


def _load_examples(source) -> list[CompactTrainingExample]:
    shards = _resolve_shards(source)
    if not shards:
        raise FileNotFoundError(f"No shards found for {source}")

    examples = []
    for shard in shards:
        examples.extend(load_compact_shard(shard))
    return examples


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
) -> list[CompactTrainingExample]:
    return mix_examples(
        _load_examples(lc0_source),
        _load_examples(self_play_source),
        lc0_fraction,
        examples_per_epoch,
        seed,
    )
