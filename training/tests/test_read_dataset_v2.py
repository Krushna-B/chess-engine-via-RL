import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chess_training.read_dataset_v2 import (
    BITPLANES,
    DATASET_MAGIC,
    DATASET_VERSION,
    INPUT_PLANES,
    POLICY_SIZE,
    TrainingSource,
    iter_compact_shard,
    load_compact_shard,
    write_compact_shard,
)


class CompactDatasetReaderTest(unittest.TestCase):
    def test_reads_cpp_record_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "sample.bin"
            planes = np.arange(BITPLANES, dtype="<u8")
            wdl = np.array([0.2, 0.3, 0.5], dtype="<f4")

            with filename.open("wb") as file:
                file.write(
                    struct.pack(
                        "<IIIIIQ",
                        DATASET_MAGIC,
                        DATASET_VERSION,
                        INPUT_PLANES,
                        BITPLANES,
                        POLICY_SIZE,
                        1,
                    )
                )
                file.write(struct.pack("<QIBBBBHH", 42, 7, 0, 5, 1, 0, 12, 2))
                file.write(planes.tobytes())
                file.write(wdl.tobytes())
                file.write(struct.pack("<f", 18.5))
                file.write(struct.pack("<Hf", 12, 0.75))
                file.write(struct.pack("<Hf", 1840, 0.25))

            examples = load_compact_shard(filename)
            streamed = list(iter_compact_shard(filename))
            round_trip_path = Path(directory) / "round-trip.bin"
            self.assertEqual(write_compact_shard(round_trip_path, examples), 1)
            round_trip = load_compact_shard(round_trip_path)

        self.assertEqual(len(examples), 1)
        self.assertEqual(len(streamed), 1)
        self.assertEqual(round_trip[0].game_id, 42)
        self.assertEqual(round_trip[0].policy[0].index, 12)
        example = examples[0]
        np.testing.assert_array_equal(example.planes, planes)
        np.testing.assert_allclose(example.wdl, wdl)
        self.assertEqual(example.game_id, 42)
        self.assertEqual(example.ply, 7)
        self.assertEqual(example.source, TrainingSource.LC0)
        self.assertEqual(example.castling_rights, 5)
        self.assertEqual(example.side_to_move, 1)
        self.assertEqual(example.rule50_count, 12)
        self.assertEqual(example.moves_left, 18.5)
        self.assertEqual([(entry.index, entry.probability) for entry in example.policy],
                         [(12, 0.75), (1840, 0.25)])

    def test_rejects_trailing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "sample.bin"
            filename.write_bytes(
                struct.pack(
                    "<IIIIIQ",
                    DATASET_MAGIC,
                    DATASET_VERSION,
                    INPUT_PLANES,
                    BITPLANES,
                    POLICY_SIZE,
                    0,
                )
                + b"unexpected"
            )

            with self.assertRaisesRegex(RuntimeError, "trailing"):
                load_compact_shard(filename)


if __name__ == "__main__":
    unittest.main()
