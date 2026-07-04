#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rum_xpred.cache_import import import_cache_archives


# Edit this list when new transferred cache folders arrive.
SOURCE_DIRS = [
    # Path("/root/shared-nvme/cache/175000-180000"),
]

# Keep these values aligned with [chunked_rum] in configs/anima_xpred.example.toml.
CHUNK_ROOT = Path("/root/shared-nvme/RUM-anima-xpred/anima-jlt-xpred-turbo10-chunks")
START_INDEX = 69000
CHUNK_SIZE = 3000

# Existing sample files are skipped by default. Set to True only when replacing bad cache.
OVERWRITE = False


def main() -> None:
    dst_cache_root = CHUNK_ROOT / "cache"
    total_extracted = 0
    total_skipped_existing = 0
    for source_dir in SOURCE_DIRS:
        print(f"importing {source_dir} -> {dst_cache_root}")
        stats = import_cache_archives(
            source_dir,
            dst_cache_root,
            start_index=START_INDEX,
            chunk_size=CHUNK_SIZE,
            overwrite=OVERWRITE,
        )
        total_extracted += stats.extracted
        total_skipped_existing += stats.skipped_existing
        print(
            "  done: "
            f"archives={stats.archives} extracted={stats.extracted} "
            f"skipped_existing={stats.skipped_existing} skipped_out_of_range={stats.skipped_out_of_range} "
            f"skipped_non_sample={stats.skipped_non_sample} skipped_unsafe={stats.skipped_unsafe}"
        )

    print(
        "all imports complete: "
        f"extracted={total_extracted} skipped_existing={total_skipped_existing} "
        f"chunk_root={CHUNK_ROOT} start_index={START_INDEX} chunk_size={CHUNK_SIZE}"
    )


if __name__ == "__main__":
    main()
