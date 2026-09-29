import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

DATASET_MAGIC = 0x43485A32
DATASET_VERSION = 2
INPUT_PLANES = 112
BITPLANES = 104
POLICY_SIZE = 1858

HEADER_FORMAT = "<IIIIIQ"
RECORD_FORMAT = "<QIBBBBHH"


class TrainingSource(IntEnum):
    LC0 = 0
    SELF_PLAY = 1


@dataclass
class CompactPolicyEntry:
    index: int
    probability: float


@dataclass
class CompactTrainingExample:
    planes: np.ndarray
    castling_rights: int
    side_to_move: int
    rule50_count: int
    policy: list[CompactPolicyEntry]
    wdl: np.ndarray
    moves_left: float
    game_id: int
    ply: int
    source: TrainingSource


def _read_exact(file, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise RuntimeError("Compact training-data file is incomplete")
    return data


def _validate_example(example: CompactTrainingExample) -> None:
    if not 0 <= example.castling_rights <= 0x0F:
        raise RuntimeError("Invalid castling rights in compact record")

    if example.side_to_move not in (0, 1):
        raise RuntimeError("Invalid side to move in compact record")

    if len(example.policy) > POLICY_SIZE:
        raise RuntimeError("Too many policy entries in compact record")

    used = set()
    for entry in example.policy:
        if not 0 <= entry.index < POLICY_SIZE or entry.index in used:
            raise RuntimeError("Invalid policy index in compact record")
        if not np.isfinite(entry.probability) or entry.probability <= 0.0:
            raise RuntimeError("Invalid policy probability in compact record")
        used.add(entry.index)

    if not np.all(np.isfinite(example.wdl)) or np.any(example.wdl < 0.0):
        raise RuntimeError("Invalid WDL target in compact record")

    if not np.isfinite(example.moves_left) or example.moves_left < 0.0:
        raise RuntimeError("Invalid moves-left target in compact record")

    if example.source not in (TrainingSource.LC0, TrainingSource.SELF_PLAY):
        raise RuntimeError("Invalid source in compact record")


def iter_compact_shard(filename: str | Path):
    with open(filename, "rb") as file:
        header = _read_exact(file, struct.calcsize(HEADER_FORMAT))
        magic, version, input_planes, bitplanes, policy_size, example_count = (
            struct.unpack(HEADER_FORMAT, header)
        )

        if (
            magic != DATASET_MAGIC
            or version != DATASET_VERSION
            or input_planes != INPUT_PLANES
            or bitplanes != BITPLANES
            or policy_size != POLICY_SIZE
        ):
            raise RuntimeError("Unsupported compact training-data header")

        for _ in range(example_count):
            record = _read_exact(file, struct.calcsize(RECORD_FORMAT))
            (
                game_id,
                ply,
                source,
                castling_rights,
                side_to_move,
                _reserved,
                rule50_count,
                policy_count,
            ) = struct.unpack(RECORD_FORMAT, record)

            planes = np.frombuffer(_read_exact(file, BITPLANES * 8), dtype="<u8").copy()
            wdl = np.frombuffer(_read_exact(file, 3 * 4), dtype="<f4").copy()
            moves_left = struct.unpack("<f", _read_exact(file, 4))[0]

            policy = []
            for _ in range(policy_count):
                index, probability = struct.unpack(
                    "<Hf", _read_exact(file, struct.calcsize("<Hf"))
                )
                policy.append(CompactPolicyEntry(index, probability))

            try:
                training_source = TrainingSource(source)
            except ValueError as error:
                raise RuntimeError("Invalid source in compact record") from error

            example = CompactTrainingExample(
                planes=planes,
                castling_rights=castling_rights,
                side_to_move=side_to_move,
                rule50_count=rule50_count,
                policy=policy,
                wdl=wdl,
                moves_left=moves_left,
                game_id=game_id,
                ply=ply,
                source=training_source,
            )
            _validate_example(example)
            yield example

        if file.read(1):
            raise RuntimeError("Unexpected trailing data in compact shard")



def load_compact_shard(filename: str | Path) -> list[CompactTrainingExample]:
    return list(iter_compact_shard(filename))


def write_compact_shard(filename: str | Path, examples) -> int:
    """Write materialized compact examples to one shard."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with filename.open("wb") as file:
        file.write(
            struct.pack(
                HEADER_FORMAT,
                DATASET_MAGIC,
                DATASET_VERSION,
                INPUT_PLANES,
                BITPLANES,
                POLICY_SIZE,
                0,
            )
        )
        for example in examples:
            _validate_example(example)
            file.write(
                struct.pack(
                    RECORD_FORMAT,
                    example.game_id,
                    example.ply,
                    int(example.source),
                    example.castling_rights,
                    example.side_to_move,
                    0,
                    example.rule50_count,
                    len(example.policy),
                )
            )
            file.write(np.asarray(example.planes, dtype="<u8").tobytes())
            file.write(np.asarray(example.wdl, dtype="<f4").tobytes())
            file.write(struct.pack("<f", example.moves_left))
            for entry in example.policy:
                file.write(struct.pack("<Hf", entry.index, entry.probability))
            count += 1
        file.seek(20)
        file.write(struct.pack("<Q", count))
    return count
