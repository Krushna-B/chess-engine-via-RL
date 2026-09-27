import unittest

import numpy as np

from chess_training.read_dataset_v2 import (
    CompactPolicyEntry,
    CompactTrainingExample,
    TrainingSource,
)
from chess_training.training_metrics import summarize_examples


def _example(source, wdl, moves_left, policy):
    return CompactTrainingExample(
        planes=np.zeros(104, dtype="<u8"),
        castling_rights=0,
        side_to_move=0,
        rule50_count=0,
        policy=[CompactPolicyEntry(index, probability) for index, probability in policy],
        wdl=np.array(wdl, dtype="<f4"),
        moves_left=moves_left,
        game_id=0,
        ply=0,
        source=source,
    )


class TrainingMetricsTest(unittest.TestCase):
    def test_summarizes_sources_targets_and_policy(self):
        examples = [
            _example(TrainingSource.LC0, [1.0, 0.0, 0.0], 10.0, [(1, 1.0)]),
            _example(TrainingSource.SELF_PLAY, [0.0, 1.0, 0.0], 20.0,
                     [(1, 0.5), (2, 0.5)]),
        ]

        metrics = summarize_examples(examples)

        self.assertEqual(metrics["examples"], 2)
        self.assertEqual(metrics["lc0_examples"], 1)
        self.assertEqual(metrics["self_play_examples"], 1)
        self.assertEqual(metrics["lc0_fraction"], 0.5)
        self.assertEqual(metrics["wdl_draw_mean"], 0.5)
        self.assertEqual(metrics["moves_left_mean"], 15.0)
        self.assertEqual(metrics["policy_support_max"], 2)
        self.assertAlmostEqual(metrics["policy_entropy_mean"], 0.3465736, places=5)

    def test_rejects_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize_examples([])


if __name__ == "__main__":
    unittest.main()
