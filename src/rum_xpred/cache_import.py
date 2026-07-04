from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


SAMPLE_RE = re.compile(r"sample-(\d+)\.safetensors$")


@dataclass(frozen=True)
class CacheImportStats:
    archives: int = 0
    extracted: int = 0
    skipped_existing: int = 0
    skipped_out_of_range: int = 0
    skipped_non_sample: int = 0
    skipped_unsafe: int = 0

    def add(self, **values: int) -> "CacheImportStats":
        data = self.__dict__.copy()
        for key, value in values.items():
            data[key] += value
        return CacheImportStats(**data)


def sample_index_from_name(name: str) -> int | None:
    match = SAMPLE_RE.search(PurePosixPath(name).name)
    return int(match.group(1)) if match else None


def chunk_id_for_sample(sample_index: int, *, start_index: int, chunk_size: int) -> int:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return (sample_index - start_index) // chunk_size


def target_path_for_member(
    member_name: str,
    *,
    dst_cache_root: Path,
    start_index: int,
    chunk_size: int,
    chunk_offset: int = 0,
    min_index: int | None = None,
    max_index: int | None = None,
) -> Path | None:
    path = PurePosixPath(member_name)
    sample_index = sample_index_from_name(member_name)
    if sample_index is None:
        return None
    if sample_index < start_index:
        return None
    if min_index is not None and sample_index < min_index:
        return None
    if max_index is not None and sample_index > max_index:
        return None

    chunk_id = chunk_id_for_sample(sample_index, start_index=start_index, chunk_size=chunk_size) + chunk_offset
    if chunk_id < 0:
        return None

    resolution = path.parent.name
    if not resolution or resolution == ".":
        return dst_cache_root / f"chunk-{chunk_id:04d}" / path.name
    return dst_cache_root / f"chunk-{chunk_id:04d}" / resolution / path.name


def is_safe_tar_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def import_cache_archives(
    src_dir: Path,
    dst_cache_root: Path,
    *,
    start_index: int,
    chunk_size: int,
    chunk_offset: int = 0,
    min_index: int | None = None,
    max_index: int | None = None,
    overwrite: bool = False,
) -> CacheImportStats:
    archives = sorted(src_dir.glob("*.tar.gz")) + sorted(src_dir.glob("*.tgz")) + sorted(src_dir.glob("*.tar"))
    if not archives:
        raise FileNotFoundError(f"no tar archives found in {src_dir}")

    stats = CacheImportStats(archives=len(archives))
    dst_cache_root.mkdir(parents=True, exist_ok=True)

    for archive in archives:
        with tarfile.open(archive, "r:*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                if not is_safe_tar_member(member.name):
                    stats = stats.add(skipped_unsafe=1)
                    continue

                sample_index = sample_index_from_name(member.name)
                target = target_path_for_member(
                    member.name,
                    dst_cache_root=dst_cache_root,
                    start_index=start_index,
                    chunk_size=chunk_size,
                    chunk_offset=chunk_offset,
                    min_index=min_index,
                    max_index=max_index,
                )
                if target is None:
                    if sample_index is None:
                        stats = stats.add(skipped_non_sample=1)
                    else:
                        stats = stats.add(skipped_out_of_range=1)
                    continue
                if target.exists() and not overwrite:
                    stats = stats.add(skipped_existing=1)
                    continue

                extracted = tf.extractfile(member)
                if extracted is None:
                    stats = stats.add(skipped_non_sample=1)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with extracted, target.open("wb") as out:
                    while True:
                        block = extracted.read(1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                stats = stats.add(extracted=1)

    return stats
