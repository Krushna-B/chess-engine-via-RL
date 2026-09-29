import json
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, random_split

from chess_training.compact_dataset import CompactChessDataset
from chess_training.chess_model import ChessTransformer
from chess_training.dataset_mixture import load_examples, mix_examples, sample_examples
from chess_training.read_dataset_v2 import write_compact_shard
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
BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("TRAIN_NUM_WORKERS", "0"))
PREFETCH_FACTOR = int(os.environ.get("TRAIN_PREFETCH_FACTOR", "2"))
PIN_MEMORY = os.environ.get("TRAIN_PIN_MEMORY", "1") == "1"
LOG_EVERY_STEPS = int(os.environ.get("TRAIN_LOG_EVERY_STEPS", "50"))
CHECKPOINT_EVERY_STEPS = int(os.environ.get("TRAIN_CHECKPOINT_EVERY_STEPS", "1000"))
RESUME = os.environ.get("TRAIN_RESUME", "1") == "1"
COMPILE_MODEL = os.environ.get("TRAIN_COMPILE", "0") == "1"
FUSED_ADAMW = os.environ.get("TRAIN_FUSED_ADAMW", "1") == "1"
TF32 = os.environ.get("TRAIN_TF32", "1") == "1"


def replay_shards(required=True):
    shards = sorted(SELFPLAY_DIR.glob("*.bin"))
    if not shards and required:
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


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    distributed=False,
    global_step=0,
    step_callback=None,
):
    training = optimizer is not None
    model.train(training)

    total_examples = 0
    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0

    interval_start = time.perf_counter()
    interval_examples = 0
    interval_steps = 0

    for states, target_policies, target_wdl in loader:
        states = states.to(device, non_blocking=PIN_MEMORY)
        target_policies = target_policies.to(device, non_blocking=PIN_MEMORY)
        target_wdl = target_wdl.to(device, non_blocking=PIN_MEMORY)

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

                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        batch_size = states.shape[0]
        total_examples += batch_size

        total_loss_sum += loss.item() * batch_size
        policy_loss_sum += policy_loss.item() * batch_size
        value_loss_sum += value_loss.item() * batch_size
        if training:
            global_step += 1
            interval_examples += batch_size
            interval_steps += 1
            should_log = LOG_EVERY_STEPS > 0 and global_step % LOG_EVERY_STEPS == 0
            should_checkpoint = (
                CHECKPOINT_EVERY_STEPS > 0
                and global_step % CHECKPOINT_EVERY_STEPS == 0
            )
            if step_callback is not None and (should_log or should_checkpoint):
                elapsed = max(time.perf_counter() - interval_start, 1e-9)
                step_metrics = None
                if should_log:
                    rank_rate = interval_examples / elapsed
                    world_size = (
                        torch.distributed.get_world_size() if distributed else 1
                    )
                    step_metrics = {
                        "train/total": loss.item(),
                        "train/policy": policy_loss.item(),
                        "train/value": value_loss.item(),
                        "train/gradient_norm": float(gradient_norm),
                        "performance/examples_per_second_per_rank": rank_rate,
                        "performance/examples_per_second_global": rank_rate * world_size,
                        "performance/seconds_per_step": elapsed / max(interval_steps, 1),
                        "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    if device.type == "cuda":
                        step_metrics.update(
                            {
                                "cuda/memory_allocated_gb": torch.cuda.memory_allocated(device) / 2**30,
                                "cuda/memory_reserved_gb": torch.cuda.memory_reserved(device) / 2**30,
                                "cuda/max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
                            }
                        )
                step_callback(global_step, step_metrics)
                if should_log:
                    interval_start = time.perf_counter()
                    interval_examples = 0
                    interval_steps = 0

    metrics = torch.tensor(
        [total_examples, total_loss_sum, policy_loss_sum, value_loss_sum],
        device=device,
        dtype=torch.float64,
    )
    if distributed:
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

    count = metrics[0].item()
    return (
        {
            "total": metrics[1].item() / count,
            "policy": metrics[2].item() / count,
            "value": metrics[3].item() / count,
        },
        global_step,
    )


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
    torch.distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    torch.cuda.set_device(local_rank)
    return rank, True, torch.device("cuda", local_rank)


def select_training_examples(shards, shard_errors):
    if EXAMPLES_PER_EPOCH > 0:
        examples_per_epoch = EXAMPLES_PER_EPOCH
        lc0_count = round(examples_per_epoch * LC0_FRACTION)
        self_play_count = examples_per_epoch - lc0_count
        self_play_examples = sample_examples(
            shards, self_play_count, seed=GENERATION, report=shard_errors
        )
    else:
        self_play_examples = load_examples(shards, report=shard_errors) if shards else []
        examples_per_epoch = len(self_play_examples)

    if LC0_FRACTION <= 0.0:
        return self_play_examples
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
        return examples

    lc0_examples = load_examples(LC0_SOURCE, report=shard_errors)
    if LC0_FRACTION == 1.0 and not self_play_examples:
        return lc0_examples
    return mix_examples(
        lc0_examples,
        self_play_examples,
        LC0_FRACTION,
        examples_per_epoch,
        seed=GENERATION,
    )


def main():
    torch.manual_seed(42)

    rank, distributed, device = setup_distributed()
    is_primary = rank == 0
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = TF32
        torch.backends.cudnn.allow_tf32 = TF32
        torch.set_float32_matmul_precision("high" if TF32 else "highest")

    wandb_logger = WandbLogger(
        WANDB_ENABLED and is_primary,
        {
            "generation": GENERATION,
            "replay_window": REPLAY_WINDOW,
            "epochs": NUMBER_OF_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "warm_start": WARM_START,
            "batch_size_per_gpu": BATCH_SIZE,
            "world_size": torch.distributed.get_world_size() if distributed else 1,
            "num_workers": NUM_WORKERS,
            "compile": COMPILE_MODEL,
            "fused_adamw": FUSED_ADAMW,
            "tf32": TF32,
            "lc0_fraction": LC0_FRACTION,
            "examples_per_epoch": EXAMPLES_PER_EPOCH,
        },
    )

    try:
        print("Device:", device)

        shards = replay_shards(required=LC0_FRACTION < 1.0)
        print(f"Replay buffer: {len(shards)} shard(s)")
        shard_errors = []
        if distributed and EXAMPLES_PER_EPOCH > 0:
            safe_run_id = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in RUN_ID
            )
            sample_cache = Path(
                os.environ.get(
                    "TRAIN_SAMPLE_CACHE",
                    CHECKPOINT_DIR
                    / (
                        f"sample_{safe_run_id}_g{GENERATION}_n{EXAMPLES_PER_EPOCH}_"
                        f"lc0_{LC0_FRACTION:.3f}.bin"
                    ),
                )
            )
            if is_primary and not sample_cache.exists():
                print(f"Building shared sample cache: {sample_cache}")
                examples = select_training_examples(shards, shard_errors)
                temporary_cache = sample_cache.with_suffix(".bin.tmp")
                write_compact_shard(temporary_cache, examples)
                os.replace(temporary_cache, sample_cache)
            torch.distributed.barrier()
            examples = load_examples(sample_cache, report=shard_errors)
        else:
            examples = select_training_examples(shards, shard_errors)

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
            }
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

        loader_options = {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY and device.type == "cuda",
            "persistent_workers": NUM_WORKERS > 0,
        }
        if NUM_WORKERS > 0:
            loader_options["prefetch_factor"] = PREFETCH_FACTOR

        train_loader = DataLoader(
            train_dataset,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **loader_options,
        )

        validation_loader = DataLoader(
            validation_dataset,
            shuffle=False,
            sampler=validation_sampler,
            **loader_options,
        )

        model = ChessTransformer().to(device)

        checkpoint_directory = CHECKPOINT_DIR
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        latest_path = checkpoint_directory / "latest.pt"
        resume_checkpoint = None
        if RESUME and latest_path.exists():
            resume_checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
            model.load_state_dict(resume_checkpoint["model_state_dict"])
            print(f"Resuming from {latest_path}")

        # Warm-start from the previous generation so learning compounds.
        warm_start_path = CHECKPOINT_DIR / "best_model.pt"
        warm_started = resume_checkpoint is None and WARM_START and warm_start_path.exists()
        if warm_started:
            model.load_state_dict(
                torch.load(warm_start_path, map_location=device, weights_only=True)
            )
            print(f"Warm-started from {warm_start_path}")

        if COMPILE_MODEL:
            model.compile()

        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                gradient_as_bucket_view=True,
                static_graph=True,
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4,
            fused=FUSED_ADAMW and device.type == "cuda",
        )
        start_epoch = 1
        global_step = 0
        best_validation_loss = float("inf")
        if resume_checkpoint is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
            global_step = int(resume_checkpoint.get("global_step", 0))
            best_validation_loss = float(
                resume_checkpoint.get("best_validation_loss", float("inf"))
            )
            if "torch_rng_state" in resume_checkpoint:
                torch.set_rng_state(resume_checkpoint["torch_rng_state"].cpu())
            if device.type == "cuda" and resume_checkpoint.get("cuda_rng_state"):
                torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng_state"])

        def save_latest(completed_epoch, checkpoint_step=None):
            if not is_primary:
                return
            state_model = model.module if distributed else model
            temporary = checkpoint_directory / "latest.pt.tmp"
            torch.save(
                {
                    "epoch": completed_epoch,
                    "global_step": global_step if checkpoint_step is None else checkpoint_step,
                    "best_validation_loss": best_validation_loss,
                    "model_state_dict": state_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else None,
                },
                temporary,
            )
            os.replace(temporary, latest_path)

        def on_step(step, metrics):
            if metrics is not None:
                wandb_logger.log(metrics, step=step)
            if CHECKPOINT_EVERY_STEPS > 0 and step % CHECKPOINT_EVERY_STEPS == 0:
                save_latest(epoch - 1, step)

        train_start = time.time()

        for epoch in range(start_epoch, NUMBER_OF_EPOCHS + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics, global_step = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                distributed,
                global_step,
                on_step if is_primary else None,
            )
            validation_metrics, _ = run_epoch(
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
                "global_step": global_step,
            }
            if is_primary:
                append_metric(epoch_metrics)
            wandb_logger.log(epoch_metrics, step=global_step)

            if validation_metrics["total"] < best_validation_loss:
                best_validation_loss = validation_metrics["total"]
                if is_primary:
                    state_model = model.module if distributed else model
                    torch.save(
                        state_model.state_dict(), checkpoint_directory / "best_model.pt"
                    )
            save_latest(epoch)

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
            "resumed": resume_checkpoint is not None,
            "global_step": global_step,
            "best_val_total": best_validation_loss,
            "parameters": (model.module if distributed else model).count_parameters(),
            "duration_ms": int((time.time() - train_start) * 1000),
        }
        if is_primary:
            append_metric(final_metrics)
        wandb_logger.log(final_metrics, step=global_step)
    finally:
        wandb_logger.finish()
        if distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
