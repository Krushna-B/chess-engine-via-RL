from pathlib import Path

import torch

from torch.utils.data import DataLoader

from chess_training.chess_dataset import ChessDataset
from chess_training.chess_model import ChessTransformer

REPO_ROOT = Path(__file__).resolve().parents[3]


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    device = choose_device()
    print("Device:", device)

    dataset = ChessDataset(REPO_ROOT / "artifacts/selfplay/neural_selfplay_shard_0001.bin")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    states, _, _ = next(iter(loader))
    states = states.to(device)

    model = ChessTransformer().to(device)

    state_dict = torch.load(
        REPO_ROOT / "artifacts/checkpoints/best_model.pt",
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits_1, values_1 = model(states)
        logits_2, values_2 = model(states)

    assert logits_1.shape == (1, 4672)
    assert values_1.shape == (1,)

    assert torch.isfinite(logits_1).all()
    assert torch.isfinite(values_1).all()

    # Evaluation mode should be deterministic
    assert torch.allclose(logits_1, logits_2)
    assert torch.allclose(values_1, values_2)

    probabilities = torch.softmax(
        logits_1,
        dim=1,
    )

    print("Logits shape:", logits_1.shape)
    print("Value:", values_1.item())
    print("Full policy sum:", probabilities.sum().item())
    print("Checkpoint verification passed")


if __name__ == "__main__":
    main()
