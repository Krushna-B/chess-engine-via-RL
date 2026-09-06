from pathlib import Path

import torch

from chess_training.chess_dataset import ChessDataset
from chess_training.chess_model import ChessTransformer


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_PATH = REPO_ROOT / "artifacts/checkpoints/best_model.pt"
OUTPUT_PATH = REPO_ROOT / "artifacts/checkpoints/chess_model_jit.pt"
DATASET_PATH = REPO_ROOT / "artifacts/selfplay/neural_selfplay_shard_0001.bin"

NUMBER_OF_TEST_POSITIONS = 5


def compare_outputs(
    original_model,
    exported_model,
    state,
    index: int,
):
    # [64, 18] -> [1, 64, 18]
    model_input = state.unsqueeze(0).contiguous()

    with torch.inference_mode():
        expected_logits, expected_value = original_model(model_input)

        actual_logits, actual_value = exported_model(model_input)

    if expected_logits.shape != (1, 4672):
        raise RuntimeError(f"Unexpected original policy shape: {expected_logits.shape}")

    if actual_logits.shape != (1, 4672):
        raise RuntimeError(f"Unexpected exported policy shape: {actual_logits.shape}")

    if expected_value.shape != (1,):
        raise RuntimeError(f"Unexpected original value shape: {expected_value.shape}")

    if actual_value.shape != (1,):
        raise RuntimeError(f"Unexpected exported value shape: {actual_value.shape}")

    if not torch.isfinite(actual_logits).all():
        raise RuntimeError(f"Non-finite policy logits at position {index}")

    if not torch.isfinite(actual_value).all():
        raise RuntimeError(f"Non-finite value at position {index}")

    if not torch.allclose(
        expected_logits,
        actual_logits,
        atol=1e-5,
        rtol=1e-5,
    ):
        maximum_difference = (expected_logits - actual_logits).abs().max().item()

        raise RuntimeError(
            f"Policy mismatch at position {index}. "
            f"Maximum difference: {maximum_difference}"
        )

    if not torch.allclose(
        expected_value,
        actual_value,
        atol=1e-5,
        rtol=1e-5,
    ):
        maximum_difference = (expected_value - actual_value).abs().max().item()

        raise RuntimeError(
            f"Value mismatch at position {index}. "
            f"Maximum difference: {maximum_difference}"
        )

    policy_difference = (expected_logits - actual_logits).abs().max().item()

    value_difference = (expected_value - actual_value).abs().max().item()

    print(
        f"Position {index}: "
        f"policy difference={policy_difference:.8f}, "
        f"value difference={value_difference:.8f}"
    )


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Construct the Python model architecture.
    original_model = ChessTransformer()

    # Load the trained parameters.
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    original_model.load_state_dict(state_dict)
    original_model.eval()

    # TransformerEncoder can switch between different optimized
    # attention paths, causing TorchScript's graph checker to fail.
    torch.backends.mha.set_fastpath_enabled(False)

    dataset = ChessDataset(str(DATASET_PATH))

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")

    # Use one real position to trace the model.
    example_state, _, _ = dataset[0]

    # [64, 18] -> [1, 64, 18]
    example_input = example_state.unsqueeze(0).contiguous()

    # Record the model's tensor operations.
    traced_model = torch.jit.trace(
        original_model,
        example_input,
        check_trace=False,
    )

    # Remove training-only parts and fold constants where possible.
    traced_model = torch.jit.freeze(traced_model)

    # Save architecture, operations, and parameters together.
    traced_model.save(str(OUTPUT_PATH))

    print(f"Saved LibTorch model: {OUTPUT_PATH}")

    # Load the artifact exactly as C++ will load it.
    exported_model = torch.jit.load(
        str(OUTPUT_PATH),
        map_location="cpu",
    )

    exported_model.eval()

    number_of_tests = min(
        NUMBER_OF_TEST_POSITIONS,
        len(dataset),
    )

    for index in range(number_of_tests):
        state, _, _ = dataset[index]

        compare_outputs(
            original_model,
            exported_model,
            state,
            index,
        )

    # Show final output information.
    with torch.inference_mode():
        policy_logits, value = exported_model(example_input)

    print("Policy shape:", policy_logits.shape)
    print("Value shape:", value.shape)
    print("Value:", value.item())

    print(f"{number_of_tests}-position LibTorch export test passed")


if __name__ == "__main__":
    main()
