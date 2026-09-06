import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split

from chess_training.chess_dataset import ChessDataset
from chess_training.chess_model import ChessTransformer

REPO_ROOT = Path(__file__).resolve().parents[3]
SELFPLAY_DIR = REPO_ROOT / "artifacts" / "selfplay"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
METRICS_PATH = Path(
    os.environ.get("METRICS_PATH", REPO_ROOT / "artifacts" / "metrics" / "metrics.jsonl")
)
GENERATION = int(os.environ.get("SELFPLAY_GENERATION", "0"))
RUN_ID = os.environ.get("RUN_ID", "adhoc")

# Train on the most recent REPLAY_WINDOW shards (one shard per generation),
# and warm-start from the previous generation's weights unless disabled.
REPLAY_WINDOW = int(os.environ.get("REPLAY_WINDOW", "20"))
WARM_START = os.environ.get("WARM_START", "1") == "1"
NUMBER_OF_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "5"))
LEARNING_RATE = float(os.environ.get("TRAIN_LR", "3e-4"))


def replay_shards():
    shards = sorted(SELFPLAY_DIR.glob("*.bin"))
    if not shards:
        raise FileNotFoundError(f"No self-play shards in {SELFPLAY_DIR}")
    return shards[-REPLAY_WINDOW:]


def append_metric(record):
    record["run_id"] = RUN_ID
    record["generation"] = GENERATION
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "a") as out:
        out.write(json.dumps(record) + "\n")


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

    shards = replay_shards()
    print(f"Replay buffer: {len(shards)} shard(s)")
    dataset = ChessDataset(shards)
    print("Training positions:", len(dataset))

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

    # Warm-start from the previous generation so learning compounds.
    warm_start_path = CHECKPOINT_DIR / "best_model.pt"
    warm_started = WARM_START and warm_start_path.exists()
    if warm_started:
        model.load_state_dict(
            torch.load(warm_start_path, map_location=device, weights_only=True)
        )
        print(f"Warm-started from {warm_start_path}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    checkpoint_directory = REPO_ROOT / "artifacts/checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")
    train_start = time.time()

    for epoch in range(1, NUMBER_OF_EPOCHS + 1):
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

        append_metric(
            {
                "type": "train_epoch",
                "epoch": epoch,
                "learning_rate": LEARNING_RATE,
                "train_total": train_metrics["total"],
                "train_policy": train_metrics["policy"],
                "train_value": train_metrics["value"],
                "val_total": validation_metrics["total"],
                "val_policy": validation_metrics["policy"],
                "val_value": validation_metrics["value"],
            }
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

    append_metric(
        {
            "type": "train",
            "device": str(device),
            "shards": len(shards),
            "positions": len(dataset),
            "epochs": NUMBER_OF_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "warm_started": warm_started,
            "best_val_total": best_validation_loss,
            "parameters": model.count_parameters(),
            "duration_ms": int((time.time() - train_start) * 1000),
        }
    )


if __name__ == "__main__":
    main()
