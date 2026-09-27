import numpy as np

from chess_training.read_dataset_v2 import CompactTrainingExample, TrainingSource


def summarize_examples(
    examples: list[CompactTrainingExample],
) -> dict[str, float | int]:
    if not examples:
        raise ValueError("Cannot summarize an empty dataset")

    source_counts = {
        TrainingSource.LC0: 0,
        TrainingSource.SELF_PLAY: 0,
    }
    wdl = []
    moves_left = []
    policy_entropies = []
    policy_support = []

    for example in examples:
        source_counts[example.source] += 1
        wdl.append(example.wdl)
        moves_left.append(example.moves_left)
        policy_support.append(len(example.policy))

        probabilities = np.array(
            [entry.probability for entry in example.policy], dtype=np.float64
        )
        if probabilities.size:
            policy_entropies.append(
                float(-np.sum(probabilities * np.log(probabilities)))
            )
        else:
            policy_entropies.append(0.0)

    count = len(examples)
    wdl_mean = np.mean(np.stack(wdl), axis=0)

    return {
        "examples": count,
        "lc0_examples": source_counts[TrainingSource.LC0],
        "self_play_examples": source_counts[TrainingSource.SELF_PLAY],
        "lc0_fraction": source_counts[TrainingSource.LC0] / count,
        "self_play_fraction": source_counts[TrainingSource.SELF_PLAY] / count,
        "wdl_win_mean": float(wdl_mean[0]),
        "wdl_draw_mean": float(wdl_mean[1]),
        "wdl_loss_mean": float(wdl_mean[2]),
        "moves_left_mean": float(np.mean(moves_left)),
        "moves_left_std": float(np.std(moves_left)),
        "policy_entropy_mean": float(np.mean(policy_entropies)),
        "policy_support_mean": float(np.mean(policy_support)),
        "policy_support_max": max(policy_support),
    }
