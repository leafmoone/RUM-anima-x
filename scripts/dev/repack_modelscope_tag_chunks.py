#!/usr/bin/env python3
import argparse
import copy
import os
import re
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path


RANGE_RE = re.compile(r"^(\d+)-(\d+)$")
SAMPLE_RE = re.compile(r"sample-(\d+)")


@dataclass(frozen=True)
class RemoteDir:
    name: str
    path: str
    start: int
    end: int

    @property
    def sample_count(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class SourceMember:
    archive_path: Path
    member_name: str
    arcname: str


@dataclass(frozen=True)
class OutputChunk:
    start: int
    end: int
    indexes: tuple[int, ...]

    @property
    def name(self) -> str:
        return f"{self.start:06d}-{self.end:06d}"


def parse_range_dir(name: str, path: str | None = None) -> RemoteDir | None:
    match = RANGE_RE.fullmatch(name)
    if not match:
        return None
    start, end = (int(value) for value in match.groups())
    if end < start:
        return None
    return RemoteDir(name=name, path=path or name, start=start, end=end)


def is_full_chunk(remote_dir: RemoteDir, chunk_size: int) -> bool:
    return remote_dir.sample_count == chunk_size


def complete_chunk_index_ranges(remote_dirs: list[RemoteDir], chunk_size: int) -> list[range]:
    return [range(item.start, item.end + 1) for item in remote_dirs if is_full_chunk(item, chunk_size)]


def index_in_any_range(index: int, ranges: list[range]) -> bool:
    return any(index in item for item in ranges)


def range_fully_covered(remote_dir: RemoteDir, ranges: list[range]) -> bool:
    return all(index_in_any_range(index, ranges) for index in range(remote_dir.start, remote_dir.end + 1))


def abnormal_dirs(remote_dirs: list[RemoteDir], chunk_size: int) -> list[RemoteDir]:
    return [item for item in remote_dirs if not is_full_chunk(item, chunk_size)]


def contiguous_runs(indexes: list[int]) -> list[list[int]]:
    if not indexes:
        return []
    runs: list[list[int]] = [[indexes[0]]]
    for index in indexes[1:]:
        if index == runs[-1][-1] + 1:
            runs[-1].append(index)
        elif index != runs[-1][-1]:
            runs.append([index])
    return runs


def build_output_chunks(indexes: set[int], chunk_size: int) -> tuple[list[OutputChunk], list[list[int]]]:
    chunks: list[OutputChunk] = []
    leftovers: list[list[int]] = []
    for run in contiguous_runs(sorted(indexes)):
        full_count = len(run) // chunk_size
        for offset in range(full_count):
            group = tuple(run[offset * chunk_size : (offset + 1) * chunk_size])
            chunks.append(OutputChunk(start=group[0], end=group[-1], indexes=group))
        remaining = run[full_count * chunk_size :]
        if remaining:
            leftovers.append(remaining)
    return chunks, leftovers


def sample_index(member_name: str) -> int | None:
    match = SAMPLE_RE.search(member_name)
    if not match:
        return None
    return int(match.group(1))


def archive_resolution(path: Path) -> str:
    name = path.name
    if name.endswith(".tar.gz"):
        return name[: -len(".tar.gz")]
    if name.endswith(".tgz"):
        return name[: -len(".tgz")]
    if name.endswith(".tar"):
        return name[: -len(".tar")]
    raise ValueError(f"unsupported archive extension: {path}")


def find_archives(download_root: Path, remote_prefix: str, remote_dir_name: str) -> list[Path]:
    base = download_root / remote_prefix / remote_dir_name
    if not base.exists():
        return []
    return sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and (path.name.endswith(".tar") or path.name.endswith(".tar.gz") or path.name.endswith(".tgz"))
    )


def build_source_index(archives: list[Path]) -> dict[str, dict[int, SourceMember]]:
    sources: dict[str, dict[int, SourceMember]] = {}
    for archive_path in archives:
        resolution = archive_resolution(archive_path)
        resolution_sources = sources.setdefault(resolution, {})
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                index = sample_index(member.name)
                if index is None or index in resolution_sources:
                    continue
                resolution_sources[index] = SourceMember(
                    archive_path=archive_path,
                    member_name=member.name,
                    arcname=member.name,
                )
    return sources


def union_sample_indexes(sources: dict[str, dict[int, SourceMember]]) -> set[int]:
    indexes: set[int] = set()
    for by_index in sources.values():
        indexes.update(by_index)
    return indexes


def create_output_tar(
    *,
    resolution_sources: dict[int, SourceMember],
    indexes: tuple[int, ...],
    output_tar: Path,
) -> None:
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    by_archive: dict[Path, list[SourceMember]] = {}
    for index in indexes:
        by_archive.setdefault(resolution_sources[index].archive_path, []).append(resolution_sources[index])

    with tarfile.open(output_tar, "w") as output:
        for archive_path, members in sorted(by_archive.items(), key=lambda item: str(item[0])):
            with tarfile.open(archive_path, "r:*") as source:
                for source_member in sorted(members, key=lambda item: sample_index(item.member_name) or -1):
                    info = source.getmember(source_member.member_name)
                    fileobj = source.extractfile(info)
                    if fileobj is None:
                        raise ValueError(f"cannot read {source_member.member_name} from {archive_path}")
                    new_info = copy.copy(info)
                    new_info.name = source_member.arcname
                    output.addfile(new_info, fileobj)


def create_output_chunk_dir(
    *,
    sources: dict[str, dict[int, SourceMember]],
    chunk: OutputChunk,
    output_root: Path,
) -> Path:
    chunk_dir = output_root / chunk.name
    for resolution, resolution_sources in sorted(sources.items()):
        resolution_indexes = tuple(index for index in chunk.indexes if index in resolution_sources)
        if not resolution_indexes:
            continue
        output_tar = chunk_dir / resolution / f"{resolution}.tar"
        create_output_tar(resolution_sources=resolution_sources, indexes=resolution_indexes, output_tar=output_tar)
    return chunk_dir


def list_remote_dirs(repo_id: str, remote_prefix: str, token: str | None, page_size: int) -> list[RemoteDir]:
    from modelscope.hub.api import HubApi

    api = HubApi()
    out: list[RemoteDir] = []
    seen: set[str] = set()
    page_number = 1
    while True:
        items = api.get_dataset_files(
            repo_id,
            root_path=remote_prefix,
            recursive=False,
            page_number=page_number,
            page_size=page_size,
            token=token,
        )
        for item in items:
            if item.get("Type") != "tree":
                continue
            parsed = parse_range_dir(item["Name"], item["Path"])
            if parsed is not None and parsed.path not in seen:
                seen.add(parsed.path)
                out.append(parsed)
        if len(items) < page_size:
            break
        page_number += 1
    return sorted(out, key=lambda item: (item.start, item.end, item.name))


def run_modelscope_download(
    *,
    repo_id: str,
    remote_path: str,
    local_dir: Path,
    token: str | None,
    max_workers: int | None,
) -> None:
    command = [
        "modelscope",
        "download",
        "--repo-type",
        "dataset",
        "--local_dir",
        str(local_dir),
        "--include",
        f"{remote_path}/**",
    ]
    if token:
        command.extend(["--token", token])
    if max_workers is not None:
        command.extend(["--max-workers", str(max_workers)])
    command.append(repo_id)
    subprocess.run(command, check=True)


def run_modelscope_upload(
    *,
    repo_id: str,
    local_path: Path,
    path_in_repo: str,
    token: str | None,
    max_workers: int | None,
) -> None:
    command = [
        "modelscope",
        "upload",
        "--repo-type",
        "dataset",
    ]
    if token:
        command.extend(["--token", token])
    if max_workers is not None:
        command.extend(["--max-workers", str(max_workers)])
    command.extend([repo_id, str(local_path), path_in_repo])
    subprocess.run(command, check=True)


def print_remote_plan(remote_dirs: list[RemoteDir], chunk_size: int) -> None:
    bad = abnormal_dirs(remote_dirs, chunk_size)
    print(f"remote tag dirs: {len(remote_dirs)}")
    print(f"non-{chunk_size} dirs: {len(bad)}")
    for item in bad:
        print(f"  {item.name}: {item.sample_count} samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download non-standard ModelScope tag chunks and repack them into fixed-size chunks.")
    parser.add_argument("--repo-id", default="leafmoone/anima_x_45000")
    parser.add_argument("--remote-prefix", default="tag")
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--work-dir", type=Path, default=Path("cache-downloads/modelscope-tag-repack"))
    parser.add_argument("--token", default=None)
    parser.add_argument("--token-env", default="MODELSCOPE_TOKEN")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--plan-only", action="store_true", help="Only list abnormal remote directories.")
    parser.add_argument("--download-only", action="store_true", help="Download and inspect, but do not create or upload output chunks.")
    parser.add_argument("--skip-download", action="store_true", help="Use already downloaded archives in --work-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Do not download, write output tars, or upload.")
    parser.add_argument("--include-existing-complete", action="store_true", help="Also repack samples that already have a 3000-sample remote chunk.")
    parser.add_argument(
        "--download-covered-abnormal",
        action="store_true",
        help="Download abnormal directories even when their whole range is already covered by existing 3000-sample chunks.",
    )
    parser.add_argument("--upload", action="store_true", help="Upload generated output chunks.")
    parser.add_argument("--keep-output", action="store_true", help="Keep generated output chunks after upload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get(args.token_env)
    remote_dirs = list_remote_dirs(args.repo_id, args.remote_prefix, token, args.page_size)
    print_remote_plan(remote_dirs, args.chunk_size)
    all_bad_dirs = abnormal_dirs(remote_dirs, args.chunk_size)
    complete_ranges = complete_chunk_index_ranges(remote_dirs, args.chunk_size)
    covered_bad_dirs = [item for item in all_bad_dirs if range_fully_covered(item, complete_ranges)]
    bad_dirs = all_bad_dirs if args.download_covered_abnormal else [item for item in all_bad_dirs if item not in covered_bad_dirs]
    if covered_bad_dirs and not args.download_covered_abnormal:
        print(f"skip covered abnormal dirs: {len(covered_bad_dirs)}")
        for item in covered_bad_dirs:
            print(f"  {item.name}: already covered by existing {args.chunk_size}-sample chunks")
    print(f"abnormal dirs selected for download: {len(bad_dirs)}")
    if args.plan_only or args.dry_run:
        return

    downloads_dir = args.work_dir / "downloads"
    output_dir = args.work_dir / "repacked" / args.remote_prefix
    downloads_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        for remote_dir in bad_dirs:
            print(f"download {remote_dir.path}")
            run_modelscope_download(
                repo_id=args.repo_id,
                remote_path=remote_dir.path,
                local_dir=downloads_dir,
                token=token,
                max_workers=args.max_workers,
            )

    archives: list[Path] = []
    for remote_dir in bad_dirs:
        archives.extend(find_archives(downloads_dir, args.remote_prefix, remote_dir.name))
    print(f"local archives: {len(archives)}")
    sources = build_source_index(archives)
    for resolution, by_index in sorted(sources.items()):
        print(f"{resolution}: {len(by_index)} unique samples")
    common_indexes = union_sample_indexes(sources)
    if not args.include_existing_complete:
        common_indexes = {index for index in common_indexes if not index_in_any_range(index, complete_ranges)}
    chunks, leftovers = build_output_chunks(common_indexes, args.chunk_size)
    print(f"planned output chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"  {chunk.name}: {len(chunk.indexes)} samples")
    if leftovers:
        print("leftover contiguous runs below chunk size:")
        for run in leftovers:
            print(f"  {run[0]:06d}-{run[-1]:06d}: {len(run)} samples")
    if args.download_only:
        return

    if args.upload and not token:
        raise ValueError(f"{args.token_env} is not set and --token was not provided")

    for chunk in chunks:
        chunk_dir = create_output_chunk_dir(sources=sources, chunk=chunk, output_root=output_dir)
        remote_path = f"{args.remote_prefix}/{chunk.name}"
        print(f"prepared {chunk_dir} -> {args.repo_id}:{remote_path}")
        if args.upload:
            run_modelscope_upload(
                repo_id=args.repo_id,
                local_path=chunk_dir,
                path_in_repo=remote_path,
                token=token,
                max_workers=args.max_workers,
            )
            if not args.keep_output:
                import shutil

                shutil.rmtree(chunk_dir)


if __name__ == "__main__":
    main()
