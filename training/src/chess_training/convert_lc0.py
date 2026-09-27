import argparse
import gzip
import math
import struct
from pathlib import Path

from chess_training.read_dataset_v2 import (
    BITPLANES,
    DATASET_MAGIC,
    DATASET_VERSION,
    INPUT_PLANES,
    POLICY_SIZE,
    TrainingSource,
)

LC0_RECORD_FORMAT = "<II1858f104Q8B15fIHH2f"
LC0_RECORD_SIZE = struct.calcsize(LC0_RECORD_FORMAT)
LC0_RECORD_FORMAT_V7 = LC0_RECORD_FORMAT + "fHH8f"
LC0_RECORD_SIZE_V7 = struct.calcsize(LC0_RECORD_FORMAT_V7)
LC0_CLASSICAL_INPUT = 1


def _reverse_bits(byte: int) -> int:
    return int(f"{byte:08b}"[::-1], 2)


def _reverse_bits_in_bytes(value: int) -> int:
    output = 0
    for index in range(8):
        byte = (value >> (index * 8)) & 0xFF
        output |= _reverse_bits(byte) << (index * 8)
    return output


def _wdl(result_q: float, result_d: float) -> tuple[float, float, float]:
    if not math.isfinite(result_q) or not math.isfinite(result_d):
        raise ValueError("Lc0 record contains a non-finite result")
    draw = max(0.0, min(1.0, result_d))
    decisive = 1.0 - draw
    win = max(0.0, min(1.0, (result_q + 1.0) * 0.5)) * decisive
    loss = decisive - win
    return win, draw, loss


def _write_header(file, count: int) -> None:
    file.write(
        struct.pack(
            "<IIIIIQ",
            DATASET_MAGIC,
            DATASET_VERSION,
            INPUT_PLANES,
            BITPLANES,
            POLICY_SIZE,
            count,
        )
    )


def _write_example(file, record, game_id: int, ply: int) -> None:
    (
        version,
        input_format,
        *values,
    ) = record
    if version not in (6, 7):
        raise ValueError(f"Unsupported Lc0 record version: {version}")
    if input_format != LC0_CLASSICAL_INPUT:
        raise ValueError(f"Unsupported Lc0 input format: {input_format}")

    probabilities = values[:POLICY_SIZE]
    planes_start = POLICY_SIZE
    planes = values[planes_start : planes_start + BITPLANES]
    metadata = values[planes_start + BITPLANES : planes_start + BITPLANES + 8]
    floats_start = planes_start + BITPLANES + 8
    floats = values[floats_start : floats_start + 15]

    castling_us_ooo, castling_us_oo, castling_them_ooo, castling_them_oo, side, rule50, _, _ = metadata
    result_q = floats[7]
    result_d = floats[8]
    plies_left = floats[6]

    sparse_policy = [
        (index, probability)
        for index, probability in enumerate(probabilities)
        if probability > 0.0 and math.isfinite(probability)
    ]
    if not sparse_policy:
        raise ValueError("Lc0 record contains an empty policy")

    rights = (
        (1 if castling_us_ooo else 0)
        | (2 if castling_us_oo else 0)
        | (4 if castling_them_ooo else 0)
        | (8 if castling_them_oo else 0)
    )
    wdl = _wdl(result_q, result_d)
    moves_left = max(0.0, plies_left) if math.isfinite(plies_left) else 0.0

    file.write(
        struct.pack(
            "<QIBBBBHH",
            game_id,
            ply,
            int(TrainingSource.LC0),
            rights,
            1 if side else 0,
            0,
            min(int(rule50), 65535),
            len(sparse_policy),
        )
    )
    file.write(
        struct.pack(
            "<104Q",
            *(_reverse_bits_in_bytes(plane) for plane in planes),
        )
    )
    file.write(struct.pack("<3f", *wdl))
    file.write(struct.pack("<f", moves_left))
    for index, probability in sparse_policy:
        file.write(struct.pack("<Hf", index, probability))


def convert_lc0_file(source: str | Path, destination: str | Path) -> int:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if source.suffix == ".gz" else open
    count = 0

    with opener(source, "rb") as input_file, destination.open("wb") as output_file:
        _write_header(output_file, 0)
        while True:
            prefix = input_file.read(8)
            if not prefix:
                break
            if len(prefix) != 8:
                raise ValueError("Truncated Lc0 training record")
            version = struct.unpack("<I", prefix[:4])[0]
            record_size = LC0_RECORD_SIZE_V7 if version == 7 else LC0_RECORD_SIZE
            data = prefix + input_file.read(record_size - 8)
            if len(data) != record_size:
                raise ValueError("Truncated Lc0 training record")
            record_format = LC0_RECORD_FORMAT_V7 if version == 7 else LC0_RECORD_FORMAT
            _write_example(
                output_file,
                struct.unpack(record_format, data),
                game_id=count,
                ply=0,
            )
            count += 1

        output_file.seek(20)
        output_file.write(struct.pack("<Q", count))

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    arguments = parser.parse_args()
    count = convert_lc0_file(arguments.source, arguments.destination)
    print(f"Converted {count} Lc0 records")


if __name__ == "__main__":
    main()
