#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rum_xpred.cache_import import (
    CacheImportStats,
    import_cache_archives,
    sample_index_from_name,
    target_path_for_member,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import exported bucket tar files into chunked_rum cache/chunk-XXXX directories."
    )
    parser.add_argument("--src", required=True, help="Directory containing *.tar, *.tar.gz, or *.tgz cache archives.")
    parser.add_argument("--chunk-root", default=None, help="chunked_rum root that contains the cache/ directory.")
    parser.add_argument(
        "--dst-cache-root",
        default=None,
        help="Explicit cache root that directly contains chunk-XXXX directories. Overrides --chunk-root/cache.",
    )
    parser.add_argument("--start-index", required=True, type=int, help="chunked_rum start_index.")
    parser.add_argument("--chunk-size", default=3000, type=positive_int, help="chunked_rum chunk_size.")
    parser.add_argument("--chunk-offset", type=int, default=0, help="Add this offset to computed chunk ids.")
    parser.add_argument("--min-index", type=int, default=None, help="Optional lowest sample index to import.")
    parser.add_argument("--max-index", type=int, default=None, help="Optional highest sample index to import.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sample files.")
    parser.add_argument("--dry-run", action="store_true", help="Only scan and report target chunks; do not extract.")
    args = parser.parse_args()
    if not args.chunk_root and not args.dst_cache_root:
        parser.error("one of --chunk-root or --dst-cache-root is required")
    return args


def dry_run(args: argparse.Namespace) -> CacheImportStats:
    import tarfile

    src = Path(args.src)
    dst = Path(args.dst_cache_root) if args.dst_cache_root else Path(args.chunk_root) / "cache"
    archives = sorted(src.glob("*.tar.gz")) + sorted(src.glob("*.tgz")) + sorted(src.glob("*.tar"))
    if not archives:
        raise FileNotFoundError(f"no tar archives found in {src}")

    stats = CacheImportStats(archives=len(archives))
    by_chunk: dict[str, int] = {}
    with tqdm(desc="scanning cache archives", unit="file") as progress:
        for archive in archives:
            with tarfile.open(archive, "r:*") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    progress.update(1)
                    target = target_path_for_member(
                        member.name,
                        dst_cache_root=dst,
                        start_index=args.start_index,
                        chunk_size=args.chunk_size,
                        chunk_offset=args.chunk_offset,
                        min_index=args.min_index,
                        max_index=args.max_index,
                    )
                    if target is None:
                        if sample_index_from_name(member.name) is None:
                            stats = stats.add(skipped_non_sample=1)
                        else:
                            stats = stats.add(skipped_out_of_range=1)
                        continue
                    by_chunk[target.parents[1].name] = by_chunk.get(target.parents[1].name, 0) + 1
                    stats = stats.add(extracted=1)
    for chunk, count in sorted(by_chunk.items()):
        print(f"{chunk}: {count} sample(s)")
    return stats


def main() -> None:
    args = parse_args()
    try:
        if args.dry_run:
            stats = dry_run(args)
        else:
            stats = import_cache_archives(
                Path(args.src),
                Path(args.dst_cache_root) if args.dst_cache_root else Path(args.chunk_root) / "cache",
                start_index=args.start_index,
                chunk_size=args.chunk_size,
                chunk_offset=args.chunk_offset,
                min_index=args.min_index,
                max_index=args.max_index,
                overwrite=args.overwrite,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        "cache import complete: "
        f"archives={stats.archives} extracted={stats.extracted} "
        f"skipped_existing={stats.skipped_existing} skipped_out_of_range={stats.skipped_out_of_range} "
        f"skipped_non_sample={stats.skipped_non_sample} skipped_unsafe={stats.skipped_unsafe}"
    )


if __name__ == "__main__":
    main()
