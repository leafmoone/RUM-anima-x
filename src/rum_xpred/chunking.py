from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
from pathlib import Path

from rum_xpred.cache_batches import bucket_cache_path, choose_cache_bucket


@dataclasses.dataclass(frozen=True)
class ChunkPlan:
    chunk_id: int
    start_index: int
    num_samples: int


def make_chunk_plan(
    *,
    start_index: int,
    total_samples: int,
    chunk_size: int,
    max_chunks: int | None,
) -> list[ChunkPlan]:
    if total_samples < 1:
        raise ValueError("total_samples must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    remaining = total_samples
    plans: list[ChunkPlan] = []
    chunk_id = 0
    current_index = start_index
    while remaining > 0 and (max_chunks is None or chunk_id < max_chunks):
        num_samples = min(chunk_size, remaining)
        plans.append(ChunkPlan(chunk_id=chunk_id, start_index=current_index, num_samples=num_samples))
        current_index += num_samples
        remaining -= num_samples
        chunk_id += 1
    return plans


def make_prompt_set_chunk_plan(
    *,
    prompt_sets: list[dict],
    chunk_size: int,
    max_chunks: int | None,
) -> list[ChunkPlan]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    totals = []
    for index, prompt_set in enumerate(prompt_sets):
        num_samples = prompt_set.get("num_samples")
        if num_samples is None:
            raise ValueError(f"prompt_sets[{index}] requires num_samples when chunked_rum.total_samples is omitted")
        totals.append(int(num_samples) * int(prompt_set.get("repeat", 1)))
    total_samples = max(totals) if totals else 0
    return make_chunk_plan(start_index=0, total_samples=total_samples, chunk_size=chunk_size, max_chunks=max_chunks)


def prompt_set_effective_total(prompt_set: dict) -> int:
    set_total = prompt_set.get("num_samples")
    if set_total is None:
        raise ValueError(f"prompt set {prompt_set.get('name', '<unnamed>')} requires num_samples")
    return int(set_total) * int(prompt_set.get("repeat", 1))


def prompt_set_slices_for_plan(prompt_set: dict, plan: ChunkPlan) -> list[dict]:
    set_start = int(prompt_set.get("start_index", 0))
    set_total = prompt_set.get("num_samples")
    if set_total is None:
        raise ValueError(f"prompt set {prompt_set.get('name', '<unnamed>')} requires num_samples")
    set_total = int(set_total)
    effective_total = prompt_set_effective_total(prompt_set)
    effective_offset = plan.start_index
    remaining = effective_total - effective_offset
    if remaining <= 0:
        return []
    to_take = min(plan.num_samples, remaining)
    slices: list[dict] = []
    while to_take > 0:
        offset_in_set = effective_offset % set_total
        repeat_cycle = effective_offset // set_total
        take = min(to_take, set_total - offset_in_set)
        adjusted = dict(prompt_set)
        adjusted["start_index"] = set_start + offset_in_set
        adjusted["num_samples"] = take
        adjusted["cache_chunk_offset"] = 0
        adjusted["_repeat_cycle"] = repeat_cycle
        slices.append(adjusted)
        effective_offset += take
        to_take -= take
    return slices


def prompt_set_slice_for_plan(prompt_set: dict, plan: ChunkPlan) -> dict | None:
    slices = prompt_set_slices_for_plan(prompt_set, plan)
    return slices[0] if slices else None


def load_chunk_manifest(path: Path) -> dict:
    if not path.exists():
        return {"chunks": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
        raise ValueError(f"invalid chunk manifest: {path}")
    return data


def write_chunk_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def update_chunk_manifest(path: Path, plan: ChunkPlan, **values) -> None:
    manifest = load_chunk_manifest(path)
    chunks = manifest.setdefault("chunks", [])
    existing = next((chunk for chunk in chunks if chunk.get("chunk_id") == plan.chunk_id), None)
    if existing is None:
        existing = {
            "chunk_id": plan.chunk_id,
            "start_index": plan.start_index,
            "num_samples": plan.num_samples,
        }
        chunks.append(existing)
    else:
        existing["start_index"] = plan.start_index
        existing["num_samples"] = plan.num_samples
    existing.update(values)
    write_chunk_manifest(path, manifest)


def completed_chunk_ids(manifest: dict) -> set[int]:
    return {
        int(chunk["chunk_id"])
        for chunk in manifest.get("chunks", [])
        if isinstance(chunk, dict) and chunk.get("status") == "complete" and "chunk_id" in chunk
    }


def chunk_train_steps(chunk: dict, fallback_output_dir: Path | None, previous_chunk: dict | None = None) -> int:
    value = chunk.get("train_steps")
    if isinstance(value, int) and value >= 0:
        return value
    total_completed_steps = chunk.get("total_completed_steps")
    if isinstance(total_completed_steps, int) and total_completed_steps >= 0:
        previous_total = previous_chunk.get("total_completed_steps") if previous_chunk else None
        if isinstance(previous_total, int) and 0 <= previous_total <= total_completed_steps:
            return total_completed_steps - previous_total
    output_dir = Path(chunk.get("output_dir") or fallback_output_dir)
    summary_path = output_dir / "train-summary.json"
    if not summary_path.exists():
        return 0
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    losses = data.get("losses", [])
    return len(losses) if isinstance(losses, list) else 0


def resolve_chunk_student_init(args: argparse.Namespace, previous_checkpoint: str | None) -> str | None:
    return previous_checkpoint or args.student_init


def resolve_chunk_optimizer_state(args: argparse.Namespace, previous_optimizer_state: str | None) -> str | None:
    return previous_optimizer_state or getattr(args, "optimizer_state", None)


def normalize_prompt_sets(args: argparse.Namespace) -> list[dict] | None:
    prompt_sets = getattr(args, "prompt_sets", None)
    if not prompt_sets:
        return None
    if not isinstance(prompt_sets, list):
        raise ValueError("prompt_sets must be a list of tables")
    normalized: list[dict] = []
    for index, prompt_set in enumerate(prompt_sets):
        if not isinstance(prompt_set, dict):
            raise ValueError("each prompt_sets entry must be a table")
        prompts = prompt_set.get("prompts")
        cache_dir = prompt_set.get("cache_dir")
        if not prompts or not cache_dir:
            raise ValueError(f"prompt_sets[{index}] requires prompts and cache_dir")
        normalized.append(
            {
                "name": prompt_set.get("name") or f"set-{index}",
                "prompts": prompts,
                "cache_dir": cache_dir,
                "start_index": int(prompt_set.get("start_index", getattr(args, "start_index", 0))),
                "num_samples": prompt_set.get("num_samples", getattr(args, "num_samples", None)),
                "cache_chunk_offset": int(prompt_set.get("cache_chunk_offset", 0)),
                "repeat": int(prompt_set.get("repeat", 1)),
            }
        )
        if normalized[-1]["repeat"] < 1:
            raise ValueError(f"prompt_sets[{index}].repeat must be >= 1")
    return normalized


def prompt_set_mix_weights(prompt_sets: list[dict]) -> list[float]:
    return [float(prompt_set_effective_total(prompt_set)) for prompt_set in prompt_sets]


def prompt_set_chunk_name(prompt_set: dict, training_chunk_id: int) -> str:
    return f"chunk-{training_chunk_id + int(prompt_set.get('cache_chunk_offset', 0)):04d}"


def link_or_symlink_cache_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)


def link_repeated_prompt_set_cache(
    *,
    prompt_set: dict,
    adjusted: dict,
    chunk_size: int,
    seed: int,
    bucket_enabled: bool,
    width: int,
    height: int,
    skip_existing: bool,
) -> tuple[int, int]:
    base_start = int(prompt_set.get("start_index", 0))
    base_cache_dir = Path(prompt_set["cache_dir"])
    dst_cache_dir = Path(adjusted["cache_dir"])
    linked = 0
    skipped = 0
    for sample_index in range(int(adjusted["start_index"]), int(adjusted["start_index"]) + int(adjusted["num_samples"])):
        offset = sample_index - base_start
        if offset < 0:
            raise ValueError(f"repeat sample index {sample_index} is before prompt set start_index {base_start}")
        source_chunk_id = offset // chunk_size
        source_cache_dir = base_cache_dir / prompt_set_chunk_name(prompt_set, source_chunk_id)
        if bucket_enabled:
            sample_width, sample_height = choose_cache_bucket(sample_index, seed)
        else:
            sample_width, sample_height = width, height
        source = bucket_cache_path(
            source_cache_dir,
            sample_index,
            width=sample_width,
            height=sample_height,
            bucket_enabled=bucket_enabled,
        )
        target = bucket_cache_path(
            dst_cache_dir,
            sample_index,
            width=sample_width,
            height=sample_height,
            bucket_enabled=bucket_enabled,
        )
        if skip_existing and target.exists():
            skipped += 1
            continue
        if not source.exists():
            raise FileNotFoundError(
                f"repeat cache source is missing: {source}. "
                "Run earlier chunks first or disable repeat for this prompt set."
            )
        if target.exists():
            target.unlink()
        link_or_symlink_cache_file(source, target)
        linked += 1
    return linked, skipped


def chunk_cache_dirs_for_prompt_sets(prompt_sets: list[dict], training_chunk_id: int) -> list[str]:
    return [str(Path(prompt_set["cache_dir"]) / prompt_set_chunk_name(prompt_set, training_chunk_id)) for prompt_set in prompt_sets]
