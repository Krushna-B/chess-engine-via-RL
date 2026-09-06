from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from chess_training.read_dataset import load_shard

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_shards(source):
    """Accept a single shard path, a directory of shards, or a list of paths."""
    if isinstance(source, (list, tuple)):
        return [Path(p) for p in source]

    path = Path(source)
    if path.is_dir():
        return sorted(path.glob("*.bin"))

    return [path]


class ChessDataset(Dataset):
    def __init__(self, source):
        shard_paths = _resolve_shards(source)
        if not shard_paths:
            raise FileNotFoundError(f"No shards found for {source}")

        states_parts = []
        policies_parts = []
        values_parts = []

        for shard_path in shard_paths:
            states, policies, values = load_shard(shard_path)
            states_parts.append(states)
            policies_parts.append(policies)
            values_parts.append(values)

        states = np.concatenate(states_parts)
        policies = np.concatenate(policies_parts)
        values = np.concatenate(values_parts)

        # [N, 64, 18]
        self.states = torch.from_numpy(states)

        # [N, 73, 64] -> [N, 4672]
        self.policies = torch.from_numpy(
            policies.reshape(policies.shape[0], -1)
        ).float()

        # [N]
        self.values = torch.from_numpy(values)

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, index):
        return (
            self.states[index],
            self.policies[index],
            self.values[index],
        )


def main():
    dataset = ChessDataset(
        REPO_ROOT / "artifacts/selfplay/neural_selfplay_shard_0001.bin"
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    states, policies, values = next(iter(loader))

    print("Dataset size:", len(dataset))
    print("State batch:", states.shape)
    print("Policy batch:", policies.shape)
    print("Value batch:", values.shape)

    print("State dtype:", states.dtype)
    print("Policy dtype:", policies.dtype)
    print("Value dtype:", values.dtype)

    print("First batch policy sums:", policies.sum(dim=1)[:5])


if __name__ == "__main__":
    main()
