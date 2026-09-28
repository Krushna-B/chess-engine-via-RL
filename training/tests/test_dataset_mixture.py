import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from chess_training.dataset_mixture import load_examples, mix_examples
from chess_training.read_dataset_v2 import (
    CompactTrainingExample,
    TrainingSource,
)


def _examples(source, count):
    return [
        CompactTrainingExample(
            planes=np.zeros(104, dtype="<u8"),
            castling_rights=0,
            side_to_move=0,
            rule50_count=0,
            policy=[],
            wdl=np.zeros(3, dtype="<f4"),
            moves_left=0.0,
            game_id=index,
            ply=0,
            source=source,
        )
        for index in range(count)
    ]


class DatasetMixtureTest(unittest.TestCase):
    def test_skips_incomplete_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.bin"
            invalid = Path(directory) / "invalid.bin"
            invalid.write_bytes(b"incomplete")
            expected = _examples(TrainingSource.SELF_PLAY, 1)

            def load(path):
                if path == valid:
                    return expected
                raise RuntimeError("Compact training-data file is incomplete")

            with patch(
                "chess_training.dataset_mixture.load_compact_shard",
                side_effect=load,
            ):
                report = []
                examples = load_examples([valid, invalid], report)

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].game_id, expected[0].game_id)
            self.assertEqual(report[0]["path"], str(invalid))

    def test_strict_shards_preserves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.bin"
            invalid.write_bytes(b"incomplete")
            with patch(
                "chess_training.dataset_mixture.load_compact_shard",
                side_effect=RuntimeError("Compact training-data file is incomplete"),
            ), patch.dict(os.environ, {"STRICT_SHARDS": "1"}):
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    load_examples([invalid])

    def test_phase_ratios_are_exact(self):
        lc0 = _examples(TrainingSource.LC0, 10)
        self_play = _examples(TrainingSource.SELF_PLAY, 10)

        for fraction, expected_lc0 in ((1.0, 20), (0.75, 15), (0.5, 10),
                                       (0.25, 5), (0.0, 0)):
            mixed = mix_examples(lc0, self_play, fraction, 20, seed=7)
            self.assertEqual(len(mixed), 20)
            self.assertEqual(
                sum(example.source == TrainingSource.LC0 for example in mixed),
                expected_lc0,
            )

    def test_seed_makes_sampling_reproducible(self):
        lc0 = _examples(TrainingSource.LC0, 2)
        self_play = _examples(TrainingSource.SELF_PLAY, 2)
        first = mix_examples(lc0, self_play, 0.5, 12, seed=11)
        second = mix_examples(lc0, self_play, 0.5, 12, seed=11)
        self.assertEqual(
            [(example.source, example.game_id) for example in first],
            [(example.source, example.game_id) for example in second],
        )


if __name__ == "__main__":
    unittest.main()
