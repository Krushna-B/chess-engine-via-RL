import struct
from pathlib import Path

import numpy as np

DATASET_MAGIC = 0x43485A31
DATASET_VERSION = 1

EXPECTED_STATE_SIZE = 64 * 18
EXPECTED_POLICY_SIZE = 64 * 73

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_shard(filename: str):
    with open(filename, "rb") as file:
        # Corresponds to:
        # uint32 magic
        # uint32 version
        # uint32 state_size
        # uint32 policy_size
        # uint64 example_count
        header = file.read(24)

        if len(header) != 24:
            raise RuntimeError("Dataset header is incomplete")

        magic, version, state_size, policy_size, example_count = struct.unpack(
            "<IIIIQ", header
        )

        if magic != DATASET_MAGIC:
            raise RuntimeError(f"Invalid magic number: {magic:#x}")

        if version != DATASET_VERSION:
            raise RuntimeError(f"Unsupported version: {version}")

        if state_size != EXPECTED_STATE_SIZE:
            raise RuntimeError(f"Unexpected state size: {state_size}")

        if policy_size != EXPECTED_POLICY_SIZE:
            raise RuntimeError(f"Unexpected policy size: {policy_size}")

        record_size = state_size + policy_size + 1

        data = np.fromfile(file, dtype="<f4", count=example_count * record_size)

    expected_float_count = example_count * record_size

    if data.size != expected_float_count:
        raise RuntimeError(
            f"Expected {expected_float_count} floats, but loaded {data.size}"
        )

    records = data.reshape(example_count, record_size)

    states = records[:, :state_size]
    policies = records[:, state_size : state_size + policy_size]
    values = records[:, -1]

    # Restore conceptual tensor shapes
    states = states.reshape(example_count, 64, 18)
    policies = policies.reshape(example_count, 73, 64)

    return states, policies, values


if __name__ == "__main__":
    states, policies, values = load_shard(
        REPO_ROOT / "artifacts/selfplay/neural_selfplay_shard_0001.bin"
    )

    print("States:", states.shape)
    print("Policies:", policies.shape)
    print("Values:", values.shape)

    print("First policy sum:", policies[0].sum())
    print("Value targets:", np.unique(values))

    if not np.allclose(policies.sum(axis=(1, 2)), 1.0, atol=1e-5):
        raise RuntimeError("Some policy targets do not sum to 1")

    if not np.all(np.isin(values, [-1.0, 0.0, 1.0])):
        raise RuntimeError("Invalid value target detected")

    print("Python dataset validation passed")
