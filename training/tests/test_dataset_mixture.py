import unittest

import numpy as np

from chess_training.dataset_mixture import mix_examples
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
