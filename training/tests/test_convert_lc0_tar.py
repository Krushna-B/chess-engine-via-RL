import gzip
import io
import struct
import tarfile
import tempfile
import unittest
from pathlib import Path

from chess_training.convert_lc0 import LC0_RECORD_FORMAT
from chess_training.convert_lc0_tar import convert_archive
from chess_training.read_dataset_v2 import load_compact_shard


def lc0_record():
    values = [0.0] * 1858
    values[12] = 1.0
    values.extend([1] + [0] * 103)
    values.extend([0, 0, 0, 0, 1, 0, 0, 0])
    values.extend([0.0] * 6)
    values.extend([10.0, 1.0, 0.0])
    values.extend([0.0] * 6)
    values.extend([0, 0, 0, 0.0, 0.0])
    return struct.pack(LC0_RECORD_FORMAT, 6, 1, *values)


class Lc0TarConverterTest(unittest.TestCase):
    def test_streams_archive_into_resumable_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sample.tar"
            with tarfile.open(archive, "w") as tar:
                for index in range(2):
                    payload = gzip.compress(lc0_record())
                    member = tarfile.TarInfo(f"sample/training.{index}.gz")
                    member.size = len(payload)
                    tar.addfile(member, io.BytesIO(payload))

            first = convert_archive(
                archive,
                root / "output",
                positions_per_shard=1,
                sample_destination_root=root / "samples",
                sample_per_archive=1,
            )
            second = convert_archive(
                archive,
                root / "output",
                positions_per_shard=1,
                sample_destination_root=root / "samples",
                sample_per_archive=1,
            )

            self.assertEqual(first, second)
            self.assertTrue(first["complete"])
            self.assertEqual(first["games"], 2)
            self.assertEqual(first["positions"], 2)
            self.assertEqual(len(first["outputs"]), 2)
            self.assertEqual(first["sample_positions"], 1)
            self.assertEqual(len(first["sample_outputs"]), 1)
            for output in first["outputs"]:
                examples = load_compact_shard(
                    root / "output" / archive.stem / output["path"]
                )
                self.assertEqual(len(examples), 1)


if __name__ == "__main__":
    unittest.main()
