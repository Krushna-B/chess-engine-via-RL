import gzip
import struct
import tempfile
import unittest
from pathlib import Path

from chess_training.convert_lc0 import (
    LC0_RECORD_FORMAT,
    convert_lc0_file,
)
from chess_training.read_dataset_v2 import TrainingSource, load_compact_shard


class Lc0ConverterTest(unittest.TestCase):
    def test_converts_v6_record(self):
        values = [0.0] * 1858
        values[12] = 0.75
        values[1840] = 0.25
        values.extend([1] + [0] * 103)
        values.extend([1, 0, 1, 0, 1, 12, 0, 0])
        values.extend([0.0] * 6)
        values.extend([24.0, 0.5, 0.25])
        values.extend([0.0] * 6)
        values.extend([0, 0, 0, 0.0, 0.0])
        record = struct.pack(LC0_RECORD_FORMAT, 6, 1, *values)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.gz"
            destination = Path(directory) / "sample.bin"
            with gzip.open(source, "wb") as file:
                file.write(record)

            self.assertEqual(convert_lc0_file(source, destination), 1)
            examples = load_compact_shard(destination)

        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example.source, TrainingSource.LC0)
        self.assertEqual(example.castling_rights, 5)
        self.assertEqual(example.side_to_move, 1)
        self.assertEqual(example.rule50_count, 12)
        self.assertEqual(example.policy[0].index, 12)
        self.assertAlmostEqual(example.wdl[0], 0.5625)
        self.assertAlmostEqual(example.wdl[1], 0.25)
        self.assertAlmostEqual(example.wdl[2], 0.1875)


if __name__ == "__main__":
    unittest.main()
