"""Convert an Lc0 tar archive directly into a small number of large shards.

The archive is streamed: its millions of gzip members never need to exist as
individual filesystem files. Each archive gets its own output directory and a
completion manifest, which makes Slurm-array retries safe.
"""

import argparse
import gzip
import json
import os
import random
import tarfile
from pathlib import Path

from chess_training.convert_lc0 import _write_header, read_record, write_example


class ShardWriter:
    def __init__(self, destination: Path, prefix: str, positions_per_shard: int):
        self.destination = destination
        self.prefix = prefix
        self.positions_per_shard = positions_per_shard
        self.index = 0
        self.count = 0
        self.file = None
        self.temporary_path = None
        self.outputs = []

    def _open(self):
        self.destination.mkdir(parents=True, exist_ok=True)
        final_path = self.destination / f"{self.prefix}_{self.index:05d}.bin"
        self.temporary_path = final_path.with_suffix(".bin.tmp")
        self.file = self.temporary_path.open("wb")
        _write_header(self.file, 0)

    def write(self, record, game_id: int, ply: int):
        if self.file is None:
            self._open()
        write_example(self.file, record, game_id, ply)
        self.count += 1
        if self.count >= self.positions_per_shard:
            self._finish_shard()

    def _finish_shard(self):
        if self.file is None:
            return
        self.file.seek(20)
        self.file.write(self.count.to_bytes(8, "little"))
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()

        final_path = self.temporary_path.with_suffix("")
        os.replace(self.temporary_path, final_path)
        self.outputs.append(
            {
                "path": final_path.name,
                "positions": self.count,
                "bytes": final_path.stat().st_size,
            }
        )
        self.index += 1
        self.count = 0
        self.file = None
        self.temporary_path = None

    def finish(self):
        self._finish_shard()
        return self.outputs


def convert_archive(
    archive: str | Path,
    destination_root: str | Path,
    positions_per_shard: int = 250_000,
    sample_destination_root: str | Path | None = None,
    sample_per_archive: int = 0,
    seed: int = 42,
) -> dict:
    archive = Path(archive)
    destination = Path(destination_root) / archive.stem
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as file:
            manifest = json.load(file)
        same_sampling = manifest.get("sample_per_archive", 0) == sample_per_archive
        outputs_exist = all(
            (destination / output["path"]).exists()
            for output in manifest.get("outputs", [])
        )
        if sample_per_archive > 0 and sample_destination_root is not None:
            sample_destination = Path(sample_destination_root) / archive.stem
            outputs_exist = outputs_exist and all(
                (sample_destination / output["path"]).exists()
                for output in manifest.get("sample_outputs", [])
            )
        if manifest.get("complete") is True and same_sampling and outputs_exist:
            print(f"Already complete: {archive}")
            return manifest

    writer = ShardWriter(destination, "shard", positions_per_shard)
    games = 0
    positions = 0
    samples = []
    generator = random.Random(seed)

    with tarfile.open(archive, "r") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".gz"):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Unable to read {member.name} from {archive}")
            with extracted, gzip.GzipFile(fileobj=extracted, mode="rb") as source:
                ply = 0
                while (record := read_record(source)) is not None:
                    writer.write(record, game_id=games, ply=ply)
                    positions += 1
                    if sample_per_archive > 0:
                        sampled = (record, games, ply)
                        if len(samples) < sample_per_archive:
                            samples.append(sampled)
                        else:
                            replacement = generator.randrange(positions)
                            if replacement < sample_per_archive:
                                samples[replacement] = sampled
                    ply += 1
            games += 1

    outputs = writer.finish()
    sample_outputs = []
    if sample_destination_root is not None and samples:
        sample_destination = Path(sample_destination_root) / archive.stem
        sample_writer = ShardWriter(
            sample_destination,
            "sample",
            positions_per_shard,
        )
        generator.shuffle(samples)
        for record, game_id, ply in samples:
            sample_writer.write(record, game_id, ply)
        sample_outputs = sample_writer.finish()
    manifest = {
        "complete": True,
        "source_archive": str(archive.resolve()),
        "games": games,
        "positions": positions,
        "positions_per_shard": positions_per_shard,
        "sample_per_archive": sample_per_archive,
        "outputs": outputs,
        "sample_positions": len(samples),
        "sample_outputs": sample_outputs,
    }
    destination.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    with temporary_manifest.open("w") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_manifest, manifest_path)
    print(f"Converted {games} games / {positions} positions -> {len(outputs)} shards")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("destination")
    parser.add_argument("--positions-per-shard", type=int, default=250_000)
    parser.add_argument("--sample-destination")
    parser.add_argument("--sample-per-archive", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if arguments.positions_per_shard < 1:
        parser.error("--positions-per-shard must be positive")
    if arguments.sample_per_archive < 0:
        parser.error("--sample-per-archive cannot be negative")
    if arguments.sample_per_archive and not arguments.sample_destination:
        parser.error("--sample-destination is required when sampling")
    convert_archive(
        arguments.archive,
        arguments.destination,
        arguments.positions_per_shard,
        arguments.sample_destination,
        arguments.sample_per_archive,
        arguments.seed,
    )


if __name__ == "__main__":
    main()
