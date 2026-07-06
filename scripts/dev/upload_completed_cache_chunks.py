#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile


SAMPLE_RE = re.compile(r"sample-(\d+)\.")
RESOLUTION_RE = re.compile(r"^\d+x\d+$")


@dataclass
class ProcessResult:
    uploaded_chunks: list[int | str] = field(default_factory=list)
    skipped_chunks: list[int | str] = field(default_factory=list)
    uploaded_model_chunks: list[str] = field(default_factory=list)
    skipped_model_chunks: list[str] = field(default_factory=list)


def parse_args():
    parser = argparse.ArgumentParser(description="Package and upload completed cache chunks to ModelScope.")
    parser.add_argument("--chunk-root", required=True, type=Path, help="Root containing chunk-manifest.json and cache chunks.")
    parser.add_argument("--repo-id", required=True, help="ModelScope repo id, for example leafmoone/anima_x_45000.")
    parser.add_argument("--repo-type", default="dataset", help="ModelScope repo type.")
    parser.add_argument("--remote-prefix", default="tag", help="Remote directory prefix inside the repo. Defaults to tag.")
    parser.add_argument("--token", default=None, help="ModelScope token. Use only from a local ignored launcher script.")
    parser.add_argument("--token-env", default="MODELSCOPE_TOKEN", help="Environment variable containing the ModelScope token.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without creating tars, uploading, deleting, or updating the manifest.")
    parser.add_argument("--staging-dir", type=Path, default=None, help="Temporary tar directory. Defaults to <chunk-root>/upload-staging.")
    parser.add_argument("--keep-tar", action="store_true", help="Keep local tar files after successful upload.")
    parser.add_argument("--chunk-id", action="append", default=None, help="Only process the selected chunk id. Can be repeated.")
    parser.add_argument("--max-upload-workers", type=int, default=None, help="Pass through to modelscope upload --max-workers.")
    parser.add_argument("--delete-cache-after-upload", action="store_true", help="Delete each local chunk cache after all its resolution tars upload successfully.")
    parser.add_argument("--upload-model-train-chunks", action="store_true", help="Upload completed model train chunk directories.")
    parser.add_argument("--model-train-dir", type=Path, default=None, help="Directory containing model train chunk subdirectories.")
    parser.add_argument("--model-repo-id", default=None, help="ModelScope model repo id for train chunk uploads.")
    parser.add_argument("--model-repo-type", default="model", help="ModelScope repo type for train chunk uploads. Defaults to model.")
    parser.add_argument("--model-remote-prefix", default="train", help="Remote directory prefix for train chunk uploads. Defaults to train.")
    parser.add_argument("--delete-model-after-upload", action="store_true", help="Delete each local model train chunk directory after successful upload.")
    parser.add_argument("--preserve-last-model-chunk", action="store_true", help="Preserve the last model train chunk directory by name-sorted order.")
    parser.add_argument("--watch", action="store_true", help="Keep scanning periodically instead of running once.")
    parser.add_argument("--scan-interval-seconds", type=int, default=7200, help="Seconds between scans in --watch mode. Defaults to 7200 (2 hours).")
    return parser.parse_args()


def load_manifest(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if isinstance(manifest, list):
        chunks = manifest
    elif isinstance(manifest, dict) and isinstance(manifest.get("chunks"), list):
        chunks = manifest["chunks"]
    else:
        raise ValueError(f"{manifest_path} must be a JSON list or an object with a chunks list")
    return manifest, chunks


def save_manifest(manifest_path: Path, manifest):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=manifest_path.parent, delete=False) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(manifest_path)


def chunk_identifier(chunk: dict):
    for key in ("chunk_id", "id", "index"):
        if key in chunk:
            return chunk[key]
    if "start_index" in chunk:
        return chunk["start_index"]
    raise ValueError(f"manifest chunk is missing chunk_id/id/index/start_index: {chunk!r}")


def chunk_prefix(chunk: dict) -> str:
    start_index = int(chunk["start_index"])
    num_samples = int(chunk["num_samples"])
    if num_samples < 1:
        raise ValueError(f"chunk {chunk_identifier(chunk)} num_samples must be >= 1")
    return f"{start_index}-{start_index + num_samples - 1}"


def candidate_cache_dirs(chunk_root: Path, chunk: dict) -> list[Path]:
    candidates = []
    for key in ("cache_dir", "chunk_dir", "path"):
        value = chunk.get(key)
        if value:
            path = Path(value)
            candidates.append(path if path.is_absolute() else chunk_root / path)

    chunk_id = chunk_identifier(chunk)
    chunk_id_text = str(chunk_id)
    candidates.extend(
        [
            chunk_root / "cache" / f"chunk-{chunk_id_text}",
            chunk_root / "cache" / f"chunk-{int(chunk_id):04d}" if str(chunk_id).isdigit() else None,
            chunk_root / f"chunk-{chunk_id_text}",
            chunk_root / f"chunk-{int(chunk_id):04d}" if str(chunk_id).isdigit() else None,
        ]
    )
    if "start_index" in chunk and "num_samples" in chunk:
        start_index = int(chunk["start_index"])
        end_index = start_index + int(chunk["num_samples"]) - 1
        candidates.extend(
            [
                chunk_root / "cache" / f"chunk-{start_index:06d}-{end_index:06d}",
                chunk_root / f"chunk-{start_index:06d}-{end_index:06d}",
                chunk_root / "cache" / chunk_prefix(chunk),
                chunk_root / chunk_prefix(chunk),
            ]
        )
    return dedupe_paths([path for path in candidates if path is not None])


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def resolve_cache_dir(chunk_root: Path, chunk: dict) -> Path:
    for path in candidate_cache_dirs(chunk_root, chunk):
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"cache path is not a directory: {path}")
            return path
    raise FileNotFoundError(f"no local cache directory found for chunk {chunk_identifier(chunk)}")


def sample_index(path: Path) -> int | None:
    match = SAMPLE_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def cache_sample_indexes(cache_dir: Path) -> set[int]:
    return {index for path in cache_dir.rglob("*") if path.is_file() for index in [sample_index(path)] if index is not None}


def cache_sample_prefix(cache_dir: Path, chunk: dict) -> str:
    num_samples = int(chunk["num_samples"])
    actual = cache_sample_indexes(cache_dir)
    if not actual:
        raise ValueError(f"no cache sample files found for chunk {chunk_identifier(chunk)}")
    actual_start = min(actual)
    actual_end = max(actual)
    expected = set(range(actual_start, actual_end + 1))
    missing = sorted(expected - actual)
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(f"missing cache sample index(es) for chunk {chunk_identifier(chunk)}: {preview}{suffix}")
    if len(actual) != num_samples:
        raise ValueError(f"chunk {chunk_identifier(chunk)} has {len(actual)} cache sample(s), expected {num_samples}")
    return f"{actual_start}-{actual_end}"


def discover_resolution_dirs(cache_dir: Path) -> list[tuple[str, Path]]:
    matches = []
    for path in sorted(cache_dir.rglob("*")):
        if path.is_dir() and RESOLUTION_RE.fullmatch(path.name) and any(child.is_file() for child in path.rglob("*")):
            matches.append((path.name, path))
    if not matches:
        raise ValueError(f"no resolution directories like 1024x1024 found under {cache_dir}")
    return matches


def create_tar(source_dir: Path, tar_path: Path):
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(source_dir))


def upload_tar(tar_path: Path, remote_path: str, repo_id: str, repo_type: str, token: str, max_upload_workers: int | None):
    command = [
        "modelscope",
        "upload",
        "--repo-type",
        repo_type,
        "--token",
        token,
    ]
    if max_upload_workers is not None:
        command.extend(["--max-workers", str(max_upload_workers)])
    command.extend([repo_id, str(tar_path), remote_path])
    subprocess.run(command, check=True)


def upload_model_train_dir(
    local_path: Path,
    remote_path: str,
    repo_id: str,
    repo_type: str,
    token: str,
    max_upload_workers: int | None,
):
    command = [
        "modelscope",
        "upload",
        "--repo-type",
        repo_type,
        "--token",
        token,
    ]
    if max_upload_workers is not None:
        command.extend(["--max-workers", str(max_upload_workers)])
    command.extend([repo_id, str(local_path), remote_path])
    subprocess.run(command, check=True)


def remote_cache_path(prefix: str, resolution: str, remote_prefix: str | None) -> str:
    parts = [part.strip("/") for part in [remote_prefix, prefix, resolution, f"{resolution}.tar"] if part and part.strip("/")]
    return "/".join(parts)


def remote_model_train_path(chunk_name: str, remote_prefix: str | None) -> str:
    parts = [part.strip("/") for part in [remote_prefix, chunk_name] if part and part.strip("/")]
    return "/".join(parts)


def should_process_chunk(chunk: dict, selected_chunk_ids: set[str] | None) -> bool:
    chunk_id = chunk_identifier(chunk)
    if selected_chunk_ids is not None and str(chunk_id) not in selected_chunk_ids:
        return False
    return chunk.get("status") == "complete"


def is_already_done(chunk: dict) -> bool:
    return bool(chunk.get("cache_uploaded")) or bool(chunk.get("cache_deleted"))


def is_complete_model_train_chunk(chunk_dir: Path) -> bool:
    return (chunk_dir / "xpred-adapter-checkpoint.safetensors").is_file() and (chunk_dir / "train-summary.json").is_file()


def process_model_train_chunks(
    *,
    model_train_dir: Path,
    repo_id: str,
    repo_type: str = "model",
    token: str | None = None,
    token_env: str = "MODELSCOPE_TOKEN",
    dry_run: bool = False,
    max_upload_workers: int | None = None,
    delete_model_after_upload: bool = False,
    preserve_last_model_chunk: bool = False,
    remote_prefix: str | None = "train",
    upload_func=upload_model_train_dir,
) -> ProcessResult:
    model_train_dir = Path(model_train_dir)
    result = ProcessResult()
    token = token or os.environ.get(token_env)

    if not model_train_dir.exists():
        print(f"skip model train upload: directory does not exist: {model_train_dir}", flush=True)
        return result
    if not model_train_dir.is_dir():
        raise ValueError(f"model train path is not a directory: {model_train_dir}")

    chunk_dirs = sorted(path for path in model_train_dir.iterdir() if path.is_dir())
    preserved_dir = chunk_dirs[-1] if preserve_last_model_chunk and chunk_dirs else None

    for chunk_dir in chunk_dirs:
        chunk_name = chunk_dir.name
        if preserved_dir is not None and chunk_dir == preserved_dir:
            result.skipped_model_chunks.append(chunk_name)
            print(f"skip model train chunk {chunk_name}: preserving last name-sorted chunk", flush=True)
            continue
        if not is_complete_model_train_chunk(chunk_dir):
            result.skipped_model_chunks.append(chunk_name)
            print(f"skip model train chunk {chunk_name}: missing checkpoint or summary", flush=True)
            continue
        if not dry_run and not token:
            raise ValueError(f"{token_env} is not set")

        remote_path = remote_model_train_path(chunk_name, remote_prefix)
        print(f"model train chunk {chunk_name}: {chunk_dir} -> {repo_id}:{remote_path}", flush=True)
        if not dry_run:
            upload_func(chunk_dir, remote_path, repo_id, repo_type, token, max_upload_workers)
            if delete_model_after_upload:
                shutil.rmtree(chunk_dir)
        result.uploaded_model_chunks.append(chunk_name)

    return result


def process_completed_chunks(
    *,
    chunk_root: Path,
    repo_id: str,
    repo_type: str = "dataset",
    token: str | None = None,
    token_env: str = "MODELSCOPE_TOKEN",
    dry_run: bool = False,
    staging_dir: Path | None = None,
    keep_tar: bool = False,
    chunk_ids: list[str] | None = None,
    max_upload_workers: int | None = None,
    delete_cache_after_upload: bool = False,
    remote_prefix: str | None = "tag",
) -> ProcessResult:
    chunk_root = Path(chunk_root)
    staging_dir = Path(staging_dir) if staging_dir is not None else chunk_root / "upload-staging"
    manifest_path = chunk_root / "chunk-manifest.json"
    manifest, chunks = load_manifest(manifest_path)
    selected_chunk_ids = {str(chunk_id) for chunk_id in chunk_ids} if chunk_ids else None
    result = ProcessResult()
    manifest_changed = False

    token = token or os.environ.get(token_env)

    for chunk in chunks:
        if not should_process_chunk(chunk, selected_chunk_ids):
            continue
        chunk_id = chunk_identifier(chunk)
        if is_already_done(chunk):
            result.skipped_chunks.append(chunk_id)
            print(f"skip chunk {chunk_id}: cache already uploaded or deleted", flush=True)
            continue

        cache_dir = resolve_cache_dir(chunk_root, chunk)
        prefix = cache_sample_prefix(cache_dir, chunk)
        resolution_dirs = discover_resolution_dirs(cache_dir)
        if not dry_run and not token:
            raise ValueError(f"{token_env} is not set")
        print(f"chunk {chunk_id}: package {len(resolution_dirs)} resolution cache dir(s) from {cache_dir}", flush=True)

        for resolution, source_dir in resolution_dirs:
            tar_path = staging_dir / prefix / resolution / f"{resolution}.tar"
            remote_path = remote_cache_path(prefix, resolution, remote_prefix)
            print(f"chunk {chunk_id}: {source_dir} -> {repo_id}:{remote_path}", flush=True)
            if dry_run:
                continue
            create_tar(source_dir, tar_path)
            upload_tar(tar_path, remote_path, repo_id, repo_type, token, max_upload_workers)
            if not keep_tar:
                tar_path.unlink()

        if not dry_run:
            if delete_cache_after_upload:
                shutil.rmtree(cache_dir)
                chunk["cache_deleted"] = True
            chunk["cache_uploaded"] = True
            chunk["cache_upload_repo"] = repo_id
            chunk["cache_upload_prefix"] = prefix
            chunk["cache_upload_remote_prefix"] = remote_prefix or ""
            manifest_changed = True
        result.uploaded_chunks.append(chunk_id)

    if manifest_changed:
        save_manifest(manifest_path, manifest)
    return result


def run_once(args):
    result = process_completed_chunks(
        chunk_root=args.chunk_root,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        token=args.token,
        token_env=args.token_env,
        dry_run=args.dry_run,
        staging_dir=args.staging_dir,
        keep_tar=args.keep_tar,
        chunk_ids=args.chunk_id,
        max_upload_workers=args.max_upload_workers,
        delete_cache_after_upload=args.delete_cache_after_upload,
        remote_prefix=args.remote_prefix,
    )
    if args.upload_model_train_chunks:
        if args.model_train_dir is None:
            raise ValueError("--model-train-dir is required with --upload-model-train-chunks")
        if not args.model_repo_id:
            raise ValueError("--model-repo-id is required with --upload-model-train-chunks")
        model_result = process_model_train_chunks(
            model_train_dir=args.model_train_dir,
            repo_id=args.model_repo_id,
            repo_type=args.model_repo_type,
            token=args.token,
            token_env=args.token_env,
            dry_run=args.dry_run,
            max_upload_workers=args.max_upload_workers,
            delete_model_after_upload=args.delete_model_after_upload,
            preserve_last_model_chunk=args.preserve_last_model_chunk,
            remote_prefix=args.model_remote_prefix,
        )
        result.uploaded_model_chunks.extend(model_result.uploaded_model_chunks)
        result.skipped_model_chunks.extend(model_result.skipped_model_chunks)
    return result


def watch_completed_chunks(args, sleep_func=time.sleep, max_scans: int | None = None):
    if args.scan_interval_seconds < 1:
        raise ValueError("--scan-interval-seconds must be >= 1")
    scans = 0
    while True:
        scans += 1
        print(f"starting cache chunk scan {scans}", flush=True)
        try:
            run_once(args)
        except Exception as exc:
            print(f"cache chunk scan {scans} failed: {exc}", flush=True)
        if max_scans is not None and scans >= max_scans:
            return
        print(f"sleeping {args.scan_interval_seconds} seconds before next scan", flush=True)
        sleep_func(args.scan_interval_seconds)


def main():
    args = parse_args()
    if args.watch:
        watch_completed_chunks(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
