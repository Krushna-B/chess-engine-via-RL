import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, random_split

from chess_training.compact_dataset import CompactChessDataset
from chess_training.chess_model import ChessTransformer
from chess_training.dataset_mixture import load_examples, mix_examples, sample_examples
from chess_training.training_metrics import summarize_examples
from chess_training.wandb_logging import WandbLogger

REPO_ROOT = Path(__file__).resolve().parents[3]
SELFPLAY_DIR = Path(
    os.environ.get("SELFPLAY_DIR", REPO_ROOT / "artifacts" / "selfplay")
)
CHECKPOINT_DIR = Path(
    os.environ.get("CHECKPOINT_DIR", REPO_ROOT / "artifacts" / "checkpoints")
)
METRICS_PATH = Path(
    os.environ.get(
        "METRICS_PATH", REPO_ROOT / "artifacts" / "metrics" / "metrics.jsonl"
    )
)
GENERATION = int(os.environ.get("SELFPLAY_GENERATION", "0"))
RUN_ID = os.environ.get("RUN_ID", "adhoc")

# Train on the most recent REPLAY_WINDOW shards (one shard per generation),
# and warm-start from the previous generation's weights unless disabled.
REPLAY_WINDOW = int(os.environ.get("REPLAY_WINDOW", "20"))
WARM_START = os.environ.get("WARM_START", "1") == "1"
NUMBER_OF_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "5"))
LEARNING_RATE = float(os.environ.get("TRAIN_LR", "3e-4"))
WANDB_ENABLED = os.environ.get("WANDB_ENABLED", "0") == "1"
LC0_SOURCE = os.environ.get("LC0_DATASET_DIR")
LC0_FRACTION = float(os.environ.get("LC0_FRACTION", "0.0"))
EXAMPLES_PER_EPOCH = int(os.environ.get("TRAIN_EXAMPLES_PER_EPOCH", "0"))


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


def calculate_loss(logits, wdl_logits, target_policies, target_wdl):
    log_policy = F.log_softmax(logits, dim=1)

    policy_loss = -(target_policies * log_policy).sum(dim=1).mean()

    value_loss = -(target_wdl * F.log_softmax(wdl_logits, dim=1)).sum(dim=1).mean()

    return policy_loss + value_loss, policy_loss, value_loss


def run_epoch(model, loader, device, optimizer=None, distributed=False):
    training = optimizer is not None
    model.train(training)

    total_examples = 0
    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0

    for states, target_policies, target_wdl in loader:
        states = states.to(device)
        target_policies = target_policies.to(device)
        target_wdl = target_wdl.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = device.type == "cuda" and torch.cuda.is_bf16_supported()
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ),
        ):
            logits, wdl_logits = model(states)

            loss, policy_loss, value_loss = calculate_loss(
                logits,
                wdl_logits,
                target_policies,
                target_wdl,
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

    metrics = torch.tensor(
        [total_examples, total_loss_sum, policy_loss_sum, value_loss_sum],
        device=device,
        dtype=torch.float64,
    )
    if distributed:
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

    count = metrics[0].item()
    return {
        "total": metrics[1].item() / count,
        "policy": metrics[2].item() / count,
        "value": metrics[3].item() / count,
    }


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def setup_distributed():
    if not torch.distributed.is_available() or "RANK" not in os.environ:
        return 0, False, choose_device()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, True, torch.device("cuda", local_rank)


def main():
    torch.manual_seed(42)

    rank, distributed, device = setup_distributed()
    is_primary = rank == 0

    wandb_logger = WandbLogger(
        WANDB_ENABLED and is_primary,
        {
            "generation": GENERATION,
            "replay_window": REPLAY_WINDOW,
            "epochs": NUMBER_OF_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "warm_start": WARM_START,
        },
    )

    try:
        print("Device:", device)

        shards = replay_shards()
        print(f"Replay buffer: {len(shards)} shard(s)")
        shard_errors = []
        if EXAMPLES_PER_EPOCH > 0:
            examples_per_epoch = EXAMPLES_PER_EPOCH
            lc0_count = round(examples_per_epoch * LC0_FRACTION)
            self_play_count = examples_per_epoch - lc0_count
            self_play_examples = sample_examples(
                shards, self_play_count, seed=GENERATION, report=shard_errors
            )
        else:
            self_play_examples = load_examples(shards, report=shard_errors)
            examples_per_epoch = len(self_play_examples)

        if LC0_FRACTION > 0.0:
            if not LC0_SOURCE:
                raise ValueError("LC0_DATASET_DIR is required when LC0_FRACTION > 0")
            if EXAMPLES_PER_EPOCH > 0:
                lc0_count = round(examples_per_epoch * LC0_FRACTION)
                lc0_examples = sample_examples(
                    LC0_SOURCE,
                    lc0_count,
                    seed=GENERATION + 1,
                    report=shard_errors,
                )
                examples = list(lc0_examples)
                examples.extend(self_play_examples)
                np.random.default_rng(GENERATION).shuffle(examples)
            else:
                lc0_examples = load_examples(LC0_SOURCE, report=shard_errors)
                examples = mix_examples(
                    lc0_examples,
                    self_play_examples,
                    LC0_FRACTION,
                    examples_per_epoch,
                    seed=GENERATION,
                )
        else:
            examples = self_play_examples

        dataset = CompactChessDataset(examples)
        print("Training positions:", len(dataset))
        composition = summarize_examples(examples)
        if is_primary:
            append_metric(
                {
                    "type": "dataset",
                    **composition,
                    "lc0_fraction_target": LC0_FRACTION,
                    "skipped_shards": len(shard_errors),
                    "skipped_shard_errors": shard_errors,
                }
            )
        wandb_logger.log(
            {
                "replay/shards": len(shards),
                "replay/positions": len(dataset),
                **{f"dataset/{key}": value for key, value in composition.items()},
                "dataset/lc0_fraction_target": LC0_FRACTION,
                "dataset/skipped_shards": len(shard_errors),
            },
            step=0,
        )

        train_size = int(0.9 * len(dataset))
        validation_size = len(dataset) - train_size

        train_dataset, validation_dataset = random_split(
            dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_sampler = (
            DistributedSampler(train_dataset, shuffle=True) if distributed else None
        )
        validation_sampler = (
            DistributedSampler(validation_dataset, shuffle=False)
            if distributed
            else None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=0,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=64,
            shuffle=False,
            sampler=validation_sampler,
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

        if distributed:
            model = DistributedDataParallel(model, device_ids=[device.index])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4,
        )

        checkpoint_directory = CHECKPOINT_DIR
        checkpoint_directory.mkdir(parents=True, exist_ok=True)

        best_validation_loss = float("inf")
        train_start = time.time()

        for epoch in range(1, NUMBER_OF_EPOCHS + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics = run_epoch(
                model, train_loader, device, optimizer, distributed
            )
            validation_metrics = run_epoch(
                model, validation_loader, device, None, distributed
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

            epoch_metrics = {
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
            if is_primary:
                append_metric(epoch_metrics)
            wandb_logger.log(epoch_metrics, step=epoch)

            if is_primary:
                state_model = model.module if distributed else model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": state_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_metrics": train_metrics,
                        "validation_metrics": validation_metrics,
                    },
                    checkpoint_directory / "latest.pt",
                )

            if validation_metrics["total"] < best_validation_loss:
                best_validation_loss = validation_metrics["total"]
                if is_primary:
                    state_model = model.module if distributed else model
                    torch.save(
                        state_model.state_dict(), checkpoint_directory / "best_model.pt"
                    )

        final_metrics = {
            "type": "train",
            "device": str(device),
            "shards": len(shards),
            "skipped_shards": len(shard_errors),
            "skipped_shard_errors": shard_errors,
            "positions": len(dataset),
            "epochs": NUMBER_OF_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "warm_started": warm_started,
            "best_val_total": best_validation_loss,
            "parameters": (model.module if distributed else model).count_parameters(),
            "duration_ms": int((time.time() - train_start) * 1000),
        }
        if is_primary:
            append_metric(final_metrics)
        wandb_logger.log(final_metrics, step=NUMBER_OF_EPOCHS)
    finally:
        wandb_logger.finish()
        if distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
