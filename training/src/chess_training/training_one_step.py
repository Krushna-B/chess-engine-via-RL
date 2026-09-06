import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from chess_dataset import ChessDataset
from chess_model import ChessTransformer

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def calculate_loss(
    policy_logits,
    predicted_values,
    policy_targets,
    value_targets,
):
    log_policy = F.log_softmax(
        policy_logits,
        dim=1,
    )

    policy_loss = -(policy_targets * log_policy).sum(dim=1).mean()

    value_loss = F.mse_loss(
        predicted_values,
        value_targets,
    )

    total_loss = policy_loss + value_loss

    return total_loss, policy_loss, value_loss


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    device = choose_device()
    print("Device:", device)

    dataset = ChessDataset(REPO_ROOT / "selfplay_shard_0001.bin")

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    model = ChessTransformer().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    states, policy_targets, value_targets = next(iter(loader))

    states = states.to(device)
    policy_targets = policy_targets.to(device)
    value_targets = value_targets.to(device)

    model.train()

    # Remove gradients left over from previous updates.
    optimizer.zero_grad(set_to_none=True)

    # Forward pass.
    policy_logits, predicted_values = model(states)

    # Calculate training objective.
    total_loss, policy_loss, value_loss = calculate_loss(
        policy_logits,
        predicted_values,
        policy_targets,
        value_targets,
    )

    print("Before update:")
    print("  Policy loss:", policy_loss.item())
    print("  Value loss:", value_loss.item())
    print("  Total loss:", total_loss.item())

    if not torch.isfinite(total_loss):
        raise RuntimeError("Training loss is not finite")

    # Calculate gradients for every model parameter.
    total_loss.backward()

    # Prevent extremely large gradients.
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    print("Gradient norm:", gradient_norm.item())

    if not torch.isfinite(gradient_norm):
        raise RuntimeError("Gradient norm is not finite")

    # Update the model parameters.
    optimizer.step()

    print("One training update passed")


if __name__ == "__main__":
    main()
