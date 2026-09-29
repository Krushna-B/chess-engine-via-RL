import numpy as np
import torch
from torch.utils.data import Dataset

from chess_training.read_dataset_v2 import (
    CompactTrainingExample,
    load_compact_shard,
)


def _plane_tensor(example: CompactTrainingExample) -> torch.Tensor:
    planes = torch.zeros((112, 8, 8), dtype=torch.float32)
    packed = np.asarray(example.planes, dtype="<u8").view(np.uint8).reshape(104, 8)
    bitplanes = np.unpackbits(packed, axis=1, bitorder="little").reshape(104, 8, 8)
    planes[:104] = torch.from_numpy(bitplanes.astype(np.float32, copy=False))

    planes[104:108] = torch.tensor(
        [
            float(example.castling_rights & 1 != 0),
            float(example.castling_rights & 2 != 0),
            float(example.castling_rights & 4 != 0),
            float(example.castling_rights & 8 != 0),
        ]
    ).view(4, 1, 1)
    planes[108].fill_(float(example.side_to_move))
    planes[109].fill_(min(example.rule50_count, 100) / 100.0)
    planes[111].fill_(1.0)
    return planes


def _policy_tensor(example: CompactTrainingExample) -> torch.Tensor:
    policy = torch.zeros(1858, dtype=torch.float32)
    for entry in example.policy:
        policy[entry.index] = entry.probability
    return policy


class CompactChessDataset(Dataset):
    def __init__(self, source):
        if isinstance(source, CompactTrainingExample):
            self.examples = [source]
        elif isinstance(source, (list, tuple)) and (
            not source or isinstance(source[0], CompactTrainingExample)
        ):
            self.examples = list(source)
        else:
            paths = source if isinstance(source, (list, tuple)) else [source]
            self.examples = []
            for path in paths:
                self.examples.extend(load_compact_shard(path))
        if not self.examples:
            raise FileNotFoundError(f"No compact examples found for {source}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return (
            _plane_tensor(example),
            _policy_tensor(example),
            torch.tensor(example.wdl, dtype=torch.float32),
        )
