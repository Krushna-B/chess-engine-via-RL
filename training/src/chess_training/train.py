from pathlib import Path

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

from chess_training.chess_dataset import ChessDataset
from chess_training.chess_model import ChessTransformer


def calculate_loss(logits, values, target_policies, target_values):
    log_policy = F.log_softmax(logits, dim=1)

    policy_loss = -(target_policies * log_policy).sum(dim=1).mean()

    value_loss = F.mse_loss(values, target_values)

    return policy_loss + value_loss, policy_loss, value_loss


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_examples = 0
    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0

    for states, target_policies, target_values in loader:
        states = states.to(device)
        target_policies = target_policies.to(device)
        target_values = target_values.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits, values = model(states)

            loss, policy_loss, value_loss = calculate_loss(
                logits,
                values,
                target_policies,
                target_values,
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        batch_size = states.shape[0]
        total_examples += batch_size

        total_loss_sum += loss.item() * batch_size
        policy_loss_sum += policy_loss.item() * batch_size
        value_loss_sum += value_loss.item() * batch_size

    return {
        "total": total_loss_sum / total_examples,
        "policy": policy_loss_sum / total_examples,
        "value": value_loss_sum / total_examples,
    }


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    torch.manual_seed(42)

    device = choose_device()
    print("Device:", device)

    dataset = ChessDataset("selfplay_shard_0001.bin")

    train_size = int(0.9 * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    model = ChessTransformer().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    checkpoint_directory = Path("checkpoints")
    checkpoint_directory.mkdir(exist_ok=True)

    best_validation_loss = float("inf")
    number_of_epochs = 5

    for epoch in range(1, number_of_epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
        )

        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train={train_metrics['total']:.4f} "
            f"(policy={train_metrics['policy']:.4f}, "
            f"value={train_metrics['value']:.4f}) | "
            f"validation={validation_metrics['total']:.4f} "
            f"(policy={validation_metrics['policy']:.4f}, "
            f"value={validation_metrics['value']:.4f})"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
            },
            checkpoint_directory / "latest.pt",
        )

        if validation_metrics["total"] < best_validation_loss:
            best_validation_loss = validation_metrics["total"]

            torch.save(
                model.state_dict(),
                checkpoint_directory / "best_model.pt",
            )


if __name__ == "__main__":
    main()
