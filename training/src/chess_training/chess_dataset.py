from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from read_dataset import load_shard

REPO_ROOT = Path(__file__).resolve().parents[3]


class ChessDataset(Dataset):
    def __init__(self, filename):
        states, polices, values = load_shard(filename)

        # [N, 64, 18]
        self.states = torch.from_numpy(states)

        # [N, 73, 64] -> [N, 4672]
        self.policies = torch.from_numpy(polices.reshape(polices.shape[0], -1)).float()

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
    dataset = ChessDataset(REPO_ROOT / "selfplay_shard_0001.bin")
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
