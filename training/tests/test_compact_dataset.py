import unittest
import importlib.util

import numpy as np

from chess_training.read_dataset_v2 import CompactTrainingExample, TrainingSource

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for model contract tests")
class CompactDatasetTest(unittest.TestCase):
    def test_plane_adapter_and_model_contract(self):
        import torch

        from chess_training.chess_model import ChessTransformer
        from chess_training.compact_dataset import _plane_tensor

        example = CompactTrainingExample(
            planes=np.array([1] + [0] * 103, dtype="<u8"),
            castling_rights=5,
            side_to_move=1,
            rule50_count=50,
            policy=[],
            wdl=np.array([0.2, 0.3, 0.5], dtype="<f4"),
            moves_left=10.0,
            game_id=1,
            ply=0,
            source=TrainingSource.LC0,
        )
        planes = _plane_tensor(example)
        self.assertEqual(tuple(planes.shape), (112, 8, 8))
        self.assertEqual(planes[0, 0, 0], 1.0)
        self.assertEqual(planes[104, 0, 0], 1.0)
        self.assertEqual(planes[106, 0, 0], 1.0)
        self.assertEqual(planes[109, 0, 0], 0.5)

        model = ChessTransformer(model_dim=32, num_of_layers=1, feedforward_dim=64)
        policy, wdl = model(torch.zeros(2, 112, 8, 8))
        self.assertEqual(tuple(policy.shape), (2, 1858))
        self.assertEqual(tuple(wdl.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
