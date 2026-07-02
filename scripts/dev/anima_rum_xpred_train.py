#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import random
import shutil
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (str(SRC_ROOT), str(REPO_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from rum_xpred.anima import (
    CacheMetadata,
    DEFAULT_EPS_FLOOR,
    DEFAULT_FLOW_SHIFT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SIGMA_MIN_TRAIN,
    DEFAULT_TEACHER_CFG,
    DEFAULT_TEACHER_STEPS,
    ToyXPredNet,
    cache_file_sort_key,
    load_object,
    load_xpred_cache_sample,
    make_shifted_sigma_schedule,
    make_toy_teacher_endpoint,
    reflow_training_target,
    sample_train_sigmas,
    sample_with_vpred_student,
    sample_with_xpred_student,
    save_xpred_cache_sample,
)
from rum_xpred.config import config_to_namespace, load_toml_config


DEFAULT_CACHE_BUCKETS: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (832, 1216),
    (1216, 832),
    (896, 1152),
    (1152, 896),
    (768, 1344),
    (1344, 768),
)


@dataclasses.dataclass(frozen=True)
class CacheBucket:
    name: str
    files: list[Path]


@dataclasses.dataclass(frozen=True)
class ChunkPlan:
    chunk_id: int
    start_index: int
    num_samples: int


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError("mixed_precision must be bf16, fp16, or fp32")


def read_prompts(path: str | Path, limit: int | None) -> list[str]:
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def chunked(items: list, batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def make_seeded_eps_batch(
    sample_indices: list[int],
    *,
    seed: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    latents = []
    for sample_index in sample_indices:
        generator = torch.Generator(device=device).manual_seed(seed + sample_index)
        latents.append(
            torch.randn(
                1,
                16,
                height // 8,
                width // 8,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
    return torch.cat(latents, dim=0)


def choose_cache_bucket(sample_index: int, seed: int) -> tuple[int, int]:
    rng = random.Random((int(seed) << 32) + int(sample_index))
    return DEFAULT_CACHE_BUCKETS[rng.randrange(len(DEFAULT_CACHE_BUCKETS))]


def bucket_name(width: int, height: int) -> str:
    return f"{int(width)}x{int(height)}"


def bucket_cache_path(cache_dir: Path, sample_index: int, *, width: int, height: int, bucket_enabled: bool) -> Path:
    filename = f"sample-{sample_index:06d}.safetensors"
    if not bucket_enabled:
        return cache_dir / filename
    return cache_dir / bucket_name(width, height) / filename


def collect_cache_buckets(cache_dir: str | Path) -> list[CacheBucket]:
    root = Path(cache_dir)
    buckets: list[CacheBucket] = []
    root_files = sorted(root.glob("*.safetensors"), key=cache_file_sort_key)
    if root_files:
        buckets.append(CacheBucket(name="root", files=root_files))
    for child in sorted(root.iterdir() if root.exists() else [], key=lambda path: path.name):
        if not child.is_dir():
            continue
        files = sorted(child.glob("*.safetensors"), key=cache_file_sort_key)
        if files:
            buckets.append(CacheBucket(name=child.name, files=files))
    return buckets


def merge_cache_samples(paths: list[Path], device: torch.device, dtype: torch.dtype) -> dict:
    samples = [load_xpred_cache_sample(path, device=device, dtype=dtype) for path in paths]
    shapes = {tuple(sample["x_teacher_latent"].shape[1:]) for sample in samples}
    if len(shapes) != 1:
        raise ValueError(f"mixed latent shapes in one train batch are not supported: {sorted(shapes)}")
    text_keys = set(samples[0]["text_conditioning"])
    if any(set(sample["text_conditioning"]) != text_keys for sample in samples):
        raise ValueError("all samples in a train batch must have the same text conditioning keys")
    return {
        "x_teacher_latent": torch.cat([sample["x_teacher_latent"] for sample in samples], dim=0),
        "eps_latent": torch.cat([sample["eps_latent"] for sample in samples], dim=0),
        "text_conditioning": {
            key: torch.cat([sample["text_conditioning"][key] for sample in samples], dim=0)
            for key in text_keys
        },
    }


class CacheBatchCursor:
    def __init__(self, cache_files: list[Path], batch_size: int, *, shuffle: bool, seed: int, drop_last: bool) -> None:
        if batch_size < 1:
            raise ValueError("train_batch_size must be >= 1")
        self.cache_files = list(cache_files)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = random.Random(seed)
        self.position = 0
        if self.shuffle:
            self.rng.shuffle(self.cache_files)

    def next(self) -> list[Path]:
        selected = []
        while len(selected) < self.batch_size:
            remaining = len(self.cache_files) - self.position
            need = self.batch_size - len(selected)
            take = min(remaining, need)
            if take:
                selected.extend(self.cache_files[self.position : self.position + take])
                self.position += take
            if len(selected) == self.batch_size:
                return selected
            self.position = 0
            if self.shuffle:
                self.rng.shuffle(self.cache_files)
            if self.drop_last and selected:
                selected = []
        return selected


class CacheBucketBatchCursor:
    def __init__(self, buckets: list[CacheBucket], batch_size: int, *, shuffle: bool, seed: int, drop_last: bool) -> None:
        if not buckets:
            raise ValueError("at least one cache bucket is required")
        self.cursors = [
            CacheBatchCursor(bucket.files, batch_size, shuffle=shuffle, seed=seed + index, drop_last=drop_last)
            for index, bucket in enumerate(buckets)
        ]
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        self.position = 0
        self.order = list(range(len(self.cursors)))
        if self.shuffle:
            self.rng.shuffle(self.order)

    def next(self) -> list[Path]:
        if self.shuffle:
            index = self.rng.randrange(len(self.cursors))
            return self.cursors[index].next()
        index = self.order[self.position]
        self.position = (self.position + 1) % len(self.order)
        return self.cursors[index].next()


def prune_checkpoints(output_dir: Path, keep: int | None, *, prediction_type: str = "x") -> None:
    if keep is None or keep < 1:
        return
    checkpoints = sorted(output_dir.glob(f"{prediction_type}pred-checkpoint-step-*.safetensors"), key=cache_file_sort_key)
    for path in checkpoints[:-keep]:
        path.unlink()


def lr_scale_for_step(args: argparse.Namespace, step: int) -> float:
    if step < 1:
        raise ValueError("step is 1-indexed and must be >= 1")
    if args.lr_warmup_steps > 0 and step <= args.lr_warmup_steps:
        return step / args.lr_warmup_steps
    if args.lr_scheduler == "constant":
        return 1.0
    if args.lr_scheduler == "cosine":
        total_steps = args.lr_scheduler_total_steps or getattr(args, "resolved_max_train_steps", None) or args.max_train_steps
        decay_steps = max(total_steps - args.lr_warmup_steps, 1)
        decay_step = min(max(step - args.lr_warmup_steps, 0), decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
        return args.lr_cosine_min + (1.0 - args.lr_cosine_min) * cosine
    raise ValueError(f"unsupported lr_scheduler: {args.lr_scheduler}")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def resolve_max_train_steps(args: argparse.Namespace, cache_sample_count: int) -> int:
    if args.max_train_steps is not None:
        return args.max_train_steps
    effective_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(cache_sample_count / effective_batch_size)
    return max(1, math.ceil(steps_per_epoch * args.num_train_epochs))


def planned_chunk_train_steps(train_args: argparse.Namespace, chunk_args: argparse.Namespace, plans: list[ChunkPlan]) -> int:
    if chunk_args.train_steps_per_chunk is not None:
        return chunk_args.train_steps_per_chunk * len(plans)
    if train_args.max_train_steps is not None:
        return train_args.max_train_steps * len(plans)
    return sum(resolve_max_train_steps(train_args, plan.num_samples) for plan in plans)


def enable_model_gradient_checkpointing(student: torch.nn.Module, args: argparse.Namespace) -> None:
    if not args.gradient_checkpointing:
        return
    if not hasattr(student, "enable_gradient_checkpointing"):
        print("gradient_checkpointing requested, but student does not expose enable_gradient_checkpointing(); ignored")
        return
    try:
        student.enable_gradient_checkpointing(
            cpu_offload=args.gradient_checkpointing_cpu_offload,
            unsloth_offload=args.gradient_checkpointing_unsloth_offload,
        )
    except TypeError:
        try:
            student.enable_gradient_checkpointing(cpu_offload=args.gradient_checkpointing_cpu_offload)
        except TypeError:
            student.enable_gradient_checkpointing()


def init_wandb(args: argparse.Namespace):
    if not getattr(args, "wandb_enabled", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb_enabled=true but wandb is not installed. Install requirements.txt or disable wandb.") from exc
    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "mode": args.wandb_mode,
        "tags": args.wandb_tags,
        "notes": args.wandb_notes,
    }
    init_kwargs = {key: value for key, value in init_kwargs.items() if value not in (None, [], "")}
    if args.wandb_log_config:
        init_kwargs["config"] = serializable_args(args)
    return wandb.init(**init_kwargs)


def log_sample_images_to_wandb(wandb_run, image_paths: list[Path], prompt: str, step: int | None = None) -> None:
    if wandb_run is None or not image_paths:
        return
    import wandb

    images = [wandb.Image(str(path), caption=prompt) for path in image_paths]
    payload = {"sample/images": images}
    if step is None:
        wandb_run.log(payload)
    else:
        wandb_run.log(payload, step=step)


def read_wandb_external_metrics(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"wandb metrics file must contain a JSON object: {metrics_path}")
    aliases = {
        "fid": "eval/fid",
        "FID": "eval/fid",
        "is": "eval/is",
        "IS": "eval/is",
        "inception_score": "eval/is",
        "Inception Score": "eval/is",
    }
    metrics: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            if "value" not in value:
                continue
            value = value["value"]
        if not isinstance(value, (int, float)):
            continue
        metric_key = aliases.get(key, key if "/" in key else f"eval/{key}")
        metrics[metric_key] = float(value)
    return metrics


def serializable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        if key == "func" or callable(value):
            continue
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            out[key] = value
        elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
            out[key] = value
    return out


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
    existing.update(values)
    write_chunk_manifest(path, manifest)


def completed_chunk_ids(manifest: dict) -> set[int]:
    return {
        int(chunk["chunk_id"])
        for chunk in manifest.get("chunks", [])
        if isinstance(chunk, dict) and chunk.get("status") == "complete" and "chunk_id" in chunk
    }


def chunk_train_steps(chunk: dict, fallback_output_dir: Path) -> int:
    value = chunk.get("train_steps")
    if isinstance(value, int) and value >= 0:
        return value
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


def make_stage_args(config_path: str, command: str) -> argparse.Namespace:
    args = config_to_namespace(load_toml_config(config_path), command_override=command)
    args.config = str(Path(config_path).resolve())
    args.func = {"build_cache": build_cache, "train_xpred": train_xpred, "sample_xpred": sample_xpred}[command]
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def final_checkpoint_for_train_args(args: argparse.Namespace) -> Path:
    prediction_type = getattr(args, "prediction_type", "x")
    if args.toy_smoke:
        return Path(args.output_dir) / ("xpred-toy-smoke.pt" if prediction_type == "x" else "vpred-toy-smoke.pt")
    return Path(args.output_dir) / (
        "xpred-adapter-checkpoint.safetensors" if prediction_type == "x" else "vpred-adapter-checkpoint.safetensors"
    )


def final_optimizer_state_for_train_args(args: argparse.Namespace) -> Path:
    prediction_type = getattr(args, "prediction_type", "x")
    return Path(args.output_dir) / f"{prediction_type}pred-train-state.pt"


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def should_run_train_sample(args: argparse.Namespace, global_step: int) -> bool:
    every = getattr(args, "sample_every_steps", 0) or 0
    total_step = getattr(args, "global_step_offset", 0) + global_step
    return every > 0 and total_step > 0 and total_step % every == 0


@torch.no_grad()
def sample_from_training_student(
    *,
    student,
    adapter,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    wandb_run=None,
) -> list[Path]:
    total_step = getattr(args, "global_step_offset", 0) + global_step
    sample_width = args.sample_width or getattr(args, "width", None) or 1024
    sample_height = args.sample_height or getattr(args, "height", None) or 1024
    sample_seed = args.sample_seed if args.sample_seed is not None else args.seed
    sigmas = make_shifted_sigma_schedule(args.sample_steps, args.flow_shift, device=device, dtype=dtype)
    eps_latent = make_seeded_eps_batch(
        list(range(args.sample_num_samples)),
        seed=sample_seed,
        height=sample_height,
        width=sample_width,
        device=device,
        dtype=dtype,
    )
    was_training = student.training
    student.eval()
    try:
        if args.toy_smoke:
            if args.prediction_type == "x":
                x_latent = sample_with_xpred_student(student, eps_latent, sigmas, eps_floor=args.sample_eps_floor)
            else:
                x_latent = sample_with_vpred_student(student, eps_latent, sigmas)
        else:
            if not args.sample_prompt:
                raise ValueError("training sample requires sample_prompt for real Anima runs")
            original_prompt = getattr(args, "prompt", None)
            args.prompt = args.sample_prompt
            try:
                student_forward = lambda z, sigma: adapter.student_forward_xpred(student, z, sigma, {})
                if args.prediction_type == "x":
                    x_latent = sample_with_xpred_student(student_forward, eps_latent, sigmas, eps_floor=args.sample_eps_floor)
                else:
                    x_latent = sample_with_vpred_student(student_forward, eps_latent, sigmas)
            finally:
                if original_prompt is None:
                    try:
                        delattr(args, "prompt")
                    except AttributeError:
                        pass
                else:
                    args.prompt = original_prompt
    finally:
        if was_training:
            student.train()

    sample_dir = Path(args.sample_output_dir) if args.sample_output_dir else Path(args.output_dir) / "train-samples"
    latent_dir = sample_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    latent_path = latent_dir / f"{args.prediction_type}pred-step-{total_step:06d}.pt"
    torch.save(
        {"x_latent": x_latent.detach().cpu(), "sigmas": sigmas.detach().cpu(), "step": total_step, "local_step": global_step},
        latent_path,
    )
    image_paths: list[Path] = []
    if args.sample_decode_images:
        if args.toy_smoke:
            print("sample_decode_images=true ignored for toy_smoke training samples")
        else:
            image_dir = sample_dir / "images" / f"step-{total_step:06d}"
            image_paths = adapter.decode_latents_to_images(x_latent, image_dir, prefix=args.sample_image_prefix)
            print(f"saved {len(image_paths)} training sample image(s) to {image_dir}")
            if args.sample_wandb_log_images:
                log_sample_images_to_wandb(wandb_run, image_paths, args.sample_prompt, step=total_step)
    print(f"saved training sample latent tensor to {latent_path}")
    del x_latent, sigmas, eps_latent
    release_cuda_memory()
    return image_paths


def build_cache(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.mixed_precision)
    all_prompts = read_prompts(args.prompts, None)
    prompts = all_prompts[args.start_index :]
    if args.num_samples is not None:
        prompts = prompts[: args.num_samples]
    if not prompts:
        raise ValueError("no prompts selected for cache build")
    cache_dir = Path(args.cache_dir)
    adapter = None if args.toy_smoke else load_object(args.adapter)(args, device=device, dtype=dtype)
    sigmas = make_shifted_sigma_schedule(args.teacher_steps, args.flow_shift, device=device, dtype=dtype)

    written = 0
    skipped = 0
    selected = [(args.start_index + offset, prompt) for offset, prompt in enumerate(prompts)]
    progress = tqdm(total=len(selected), desc="building x-pred cache", unit="sample")

    def write_batch(width: int, height: int, batch: list[tuple[int, str, Path]]) -> None:
        nonlocal written
        sample_indices = [item[0] for item in batch]
        batch_prompts = [item[1] for item in batch]
        eps_latent = make_seeded_eps_batch(
            sample_indices,
            seed=args.seed,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        if args.toy_smoke:
            x_teacher_latent = torch.cat(
                [make_toy_teacher_endpoint(eps_latent[i : i + 1], sample_index) for i, sample_index in enumerate(sample_indices)],
                dim=0,
            )
            text_conditioning = {"prompt_index": torch.tensor(sample_indices, device=device, dtype=dtype).view(-1, 1)}
        else:
            x_teacher_latent, text_conditioning = adapter.teacher_sample_latent(
                prompt=batch_prompts,
                eps_latent=eps_latent,
                sigmas=sigmas,
                guidance_scale=args.teacher_cfg,
            )
        for batch_index, (sample_index, prompt, path) in enumerate(batch):
            metadata = CacheMetadata(
                prompt=prompt,
                width=width,
                height=height,
                seed=args.seed + sample_index,
                sample_index=sample_index,
                teacher_steps=args.teacher_steps,
                flow_shift=args.flow_shift,
                teacher_cfg=args.teacher_cfg,
                teacher_lora=getattr(args, "teacher_lora", None),
                teacher_lora_weight=getattr(args, "teacher_lora_weight", 1.0),
            )
            save_xpred_cache_sample(
                path,
                x_teacher_latent[batch_index : batch_index + 1],
                eps_latent[batch_index : batch_index + 1],
                {key: value[batch_index : batch_index + 1] for key, value in text_conditioning.items()},
                metadata,
            )
            written += 1
            progress.update(1)
            progress.set_postfix(written=written, skipped=skipped)
        del eps_latent, x_teacher_latent, text_conditioning

    try:
        pending_by_size: dict[tuple[int, int], list[tuple[int, str, Path]]] = {}
        for sample_index, prompt in selected:
            if getattr(args, "bucket_enabled", False):
                width, height = choose_cache_bucket(sample_index, args.seed)
            else:
                width, height = args.width, args.height
            path = bucket_cache_path(
                cache_dir,
                sample_index,
                width=width,
                height=height,
                bucket_enabled=bool(getattr(args, "bucket_enabled", False)),
            )
            if args.skip_existing and path.exists():
                skipped += 1
                progress.update(1)
                progress.set_postfix(written=written, skipped=skipped)
                continue
            pending = pending_by_size.setdefault((width, height), [])
            pending.append((sample_index, prompt, path))
            if len(pending) >= args.cache_batch_size:
                write_batch(width, height, pending)
                pending.clear()
        for (width, height), pending in pending_by_size.items():
            if pending:
                write_batch(width, height, pending)
        print(f"wrote {written} x-pred cache sample(s) to {cache_dir}; skipped {skipped} existing sample(s)")
    finally:
        progress.close()
        del adapter, sigmas
        release_cuda_memory()


def chunked_rum(args: argparse.Namespace) -> None:
    if not getattr(args, "config", None):
        raise ValueError("chunked_rum requires --config so it can reuse [build_cache] and [train_xpred]")
    chunk_root = Path(args.chunk_root)
    chunk_root.mkdir(parents=True, exist_ok=True)
    manifest_path = chunk_root / "chunk-manifest.json"
    plans = make_chunk_plan(
        start_index=args.start_index,
        total_samples=args.total_samples,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
    )
    manifest = load_chunk_manifest(manifest_path)
    completed = completed_chunk_ids(manifest) if args.resume else set()
    previous_checkpoint = None
    previous_optimizer_state = None
    manifest_by_chunk_id = {
        int(chunk["chunk_id"]): chunk
        for chunk in manifest.get("chunks", [])
        if isinstance(chunk, dict) and "chunk_id" in chunk
    }
    global_step_offset = 0
    base_train_args = make_stage_args(args.config, "train_xpred")
    planned_total_train_steps = planned_chunk_train_steps(base_train_args, args, plans)

    print(
        f"chunked_rum: {len(plans)} planned chunk(s), chunk_size={args.chunk_size}, "
        f"total_samples={args.total_samples}, resume={args.resume}"
    )
    for plan in plans:
        chunk_name = f"chunk-{plan.chunk_id:04d}"
        chunk_cache_dir = chunk_root / "cache" / chunk_name
        chunk_output_dir = chunk_root / "train" / chunk_name
        if plan.chunk_id in completed:
            print(f"{chunk_name}: already complete, skipping")
            completed_chunk = manifest_by_chunk_id.get(plan.chunk_id, {})
            checkpoint = completed_chunk.get("checkpoint")
            if checkpoint:
                previous_checkpoint = str(checkpoint)
            optimizer_state = completed_chunk.get("optimizer_state")
            if optimizer_state:
                previous_optimizer_state = str(optimizer_state)
            global_step_offset += chunk_train_steps(completed_chunk, chunk_output_dir)
            continue

        cache_args = make_stage_args(args.config, "build_cache")
        cache_args.cache_dir = str(chunk_cache_dir)
        cache_args.start_index = plan.start_index
        cache_args.num_samples = plan.num_samples
        update_chunk_manifest(manifest_path, plan, status="building_cache", cache_dir=str(chunk_cache_dir))
        print(f"{chunk_name}: building cache start_index={plan.start_index} num_samples={plan.num_samples}")
        build_cache(cache_args)
        release_cuda_memory()
        update_chunk_manifest(manifest_path, plan, status="cache_built", cache_dir=str(chunk_cache_dir))

        train_args = make_stage_args(args.config, "train_xpred")
        train_args.cache_dir = str(chunk_cache_dir)
        train_args.output_dir = str(chunk_output_dir)
        train_args.student_init = resolve_chunk_student_init(args, previous_checkpoint)
        train_args.optimizer_state = resolve_chunk_optimizer_state(args, previous_optimizer_state)
        train_args.global_step_offset = global_step_offset
        if train_args.lr_scheduler_total_steps is None:
            train_args.lr_scheduler_total_steps = planned_total_train_steps
        if args.train_steps_per_chunk is not None:
            train_args.max_train_steps = args.train_steps_per_chunk
        update_chunk_manifest(manifest_path, plan, status="training", output_dir=str(chunk_output_dir))
        print(
            f"{chunk_name}: training cache_dir={chunk_cache_dir} "
            f"student_init={train_args.student_init or '<default>'} "
            f"optimizer_state={train_args.optimizer_state or '<none>'}"
        )
        try:
            train_xpred(train_args)
        finally:
            release_cuda_memory()

        checkpoint = final_checkpoint_for_train_args(train_args)
        if not checkpoint.exists():
            raise FileNotFoundError(f"expected chunk checkpoint was not written: {checkpoint}")
        previous_checkpoint = str(checkpoint)
        optimizer_state = final_optimizer_state_for_train_args(train_args)
        if not optimizer_state.exists():
            raise FileNotFoundError(f"expected chunk optimizer state was not written: {optimizer_state}")
        previous_optimizer_state = str(optimizer_state)
        update_chunk_manifest(
            manifest_path,
            plan,
            status="complete",
            output_dir=str(chunk_output_dir),
            checkpoint=str(checkpoint),
            optimizer_state=str(optimizer_state),
            train_steps=getattr(train_args, "completed_train_steps", None) or getattr(train_args, "resolved_max_train_steps", None),
        )
        global_step_offset += getattr(train_args, "completed_train_steps", 0) or getattr(train_args, "resolved_max_train_steps", 0)
        if args.delete_cache_after_train:
            shutil.rmtree(chunk_cache_dir, ignore_errors=True)
            update_chunk_manifest(manifest_path, plan, cache_deleted=True)
            print(f"{chunk_name}: deleted cache {chunk_cache_dir}")
    print(f"chunked_rum complete; manifest: {manifest_path}")


def train_xpred(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.mixed_precision)
    cache_buckets = collect_cache_buckets(args.cache_dir)
    cache_files = [path for bucket in cache_buckets for path in bucket.files]
    if not cache_files:
        raise ValueError(f"no safetensors cache files found in {args.cache_dir}")
    adapter = None if args.toy_smoke else load_object(args.adapter)(args, device=device, dtype=dtype)
    if args.toy_smoke:
        student = ToyXPredNet().to(device=device, dtype=dtype)
        if args.student_init and Path(args.student_init).exists():
            state = torch.load(args.student_init, map_location=device)
            student.load_state_dict(state["state_dict"])
    else:
        student = adapter.load_student_xpred(init_checkpoint=args.student_init)
        student.to(device=device, dtype=dtype)
    enable_model_gradient_checkpointing(student, args)
    student.train()
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    optimizer_state_path = getattr(args, "optimizer_state", None)
    if optimizer_state_path:
        state_path = Path(optimizer_state_path)
        if state_path.exists():
            state = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(state["optimizer"])
            print(f"loaded optimizer state from {state_path}")
        else:
            print(f"optimizer_state was set but not found, starting optimizer fresh: {state_path}")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_cursor = CacheBucketBatchCursor(
        cache_buckets,
        args.train_batch_size,
        shuffle=args.shuffle_cache,
        seed=args.seed,
        drop_last=args.drop_last,
    )
    resolved_max_train_steps = resolve_max_train_steps(args, len(cache_files))
    args.resolved_max_train_steps = resolved_max_train_steps
    if args.max_train_steps is None:
        print(
            "auto max_train_steps="
            f"{resolved_max_train_steps} from cache_samples={len(cache_files)}, "
            f"cache_buckets={len(cache_buckets)}, "
            f"effective_batch_size={args.train_batch_size * args.gradient_accumulation_steps}, "
            f"num_train_epochs={args.num_train_epochs}"
        )
    wandb_run = init_wandb(args)

    losses: list[float] = []
    completed_train_steps = 0
    try:
        for step in range(resolved_max_train_steps):
            global_step = step + 1
            total_step = getattr(args, "global_step_offset", 0) + global_step
            current_lr = args.learning_rate * lr_scale_for_step(args, total_step)
            set_optimizer_lr(optimizer, current_lr)
            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []
            for _ in range(args.gradient_accumulation_steps):
                sample = merge_cache_samples(batch_cursor.next(), device=device, dtype=dtype)
                sigma = sample_train_sigmas(
                    sample["x_teacher_latent"].shape[0],
                    sigma_min_train=args.sigma_min_train,
                    flow_shift=args.flow_shift,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                z = (1 - sigma) * sample["x_teacher_latent"] + sigma * sample["eps_latent"]
                if args.toy_smoke:
                    prediction = student(z, sigma)
                else:
                    prediction = adapter.student_forward_xpred(student, z, sigma, sample["text_conditioning"])
                target = reflow_training_target(args.prediction_type, sample["x_teacher_latent"], sample["eps_latent"])
                loss_value = torch.nn.functional.mse_loss(prediction.float(), target.float())
                if not torch.isfinite(loss_value):
                    raise FloatingPointError(f"non-finite {args.prediction_type}-pred loss")
                (loss_value / args.gradient_accumulation_steps).backward()
                micro_losses.append(float(loss_value.detach()))
            grad_norm = None
            if args.max_grad_norm and args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            for parameter in student.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("non-finite x-pred gradient")
            optimizer.step()
            completed_train_steps = global_step
            args.completed_train_steps = completed_train_steps
            loss = sum(micro_losses) / len(micro_losses)
            losses.append(float(loss))
            if wandb_run is not None:
                grad_norm_value = None if grad_norm is None else float(grad_norm)
                wandb_metrics = {
                    "train/loss": float(loss),
                    "train/lr": current_lr,
                    "train/grad_norm": grad_norm_value,
                    "grad_norm": grad_norm_value,
                    "train/effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
                }
                if args.wandb_metrics_log_every > 0 and global_step % args.wandb_metrics_log_every == 0:
                    wandb_metrics.update(read_wandb_external_metrics(args.wandb_metrics_file))
                wandb_run.log(wandb_metrics, step=total_step)
            if global_step % args.log_every == 0:
                grad_text = "" if grad_norm is None else f" grad_norm={float(grad_norm):.6f}"
                offset_text = "" if getattr(args, "global_step_offset", 0) == 0 else f" total_step={total_step}"
                print(f"step={global_step}{offset_text} loss={float(loss):.6f} lr={current_lr:.8g}{grad_text}")
            if should_run_train_sample(args, global_step):
                sample_from_training_student(
                    student=student,
                    adapter=adapter,
                    args=args,
                    device=device,
                    dtype=dtype,
                    global_step=global_step,
                    wandb_run=wandb_run,
                )
            if args.dry_run:
                print("dry_run=true: completed one optimizer step and skipped checkpoint saving")
                break
            if args.save_every_steps and global_step % args.save_every_steps == 0:
                step_checkpoint = output_dir / f"{args.prediction_type}pred-checkpoint-step-{global_step:06d}.safetensors"
                if args.toy_smoke:
                    torch.save({"state_dict": student.state_dict(), "args": serializable_args(args), "losses": losses}, step_checkpoint.with_suffix(".pt"))
                else:
                    adapter.save_student_xpred(student, step_checkpoint)
                    prune_checkpoints(output_dir, args.checkpoints_total_limit, prediction_type=args.prediction_type)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    if args.dry_run:
        return

    checkpoint_path = final_checkpoint_for_train_args(args)
    if args.toy_smoke:
        torch.save({"state_dict": student.state_dict(), "args": serializable_args(args), "losses": losses}, checkpoint_path)
    else:
        adapter.save_student_xpred(student, checkpoint_path)
    train_state_path = final_optimizer_state_for_train_args(args)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "args": serializable_args(args),
            "prediction_type": args.prediction_type,
            "global_step_offset": getattr(args, "global_step_offset", 0),
            "completed_train_steps": completed_train_steps,
            "total_completed_steps": getattr(args, "global_step_offset", 0) + completed_train_steps,
        },
        train_state_path,
    )
    (output_dir / "train-summary.json").write_text(
        json.dumps(
            {
                "losses": losses,
                "prediction_type": args.prediction_type,
                "global_step_offset": getattr(args, "global_step_offset", 0),
                "completed_train_steps": completed_train_steps,
                "total_completed_steps": getattr(args, "global_step_offset", 0) + completed_train_steps,
                "optimizer_state": str(train_state_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.optimizer_state_output = str(train_state_path)
    print(f"saved {args.prediction_type}-pred checkpoint to {checkpoint_path}")
    print(f"saved optimizer state to {train_state_path}")
    del student, optimizer, batch_cursor
    if not args.toy_smoke:
        del adapter
    release_cuda_memory()


def sample_xpred(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.mixed_precision)
    sigmas = make_shifted_sigma_schedule(args.steps, args.flow_shift, device=device, dtype=dtype)
    eps_latent = make_seeded_eps_batch(
        list(range(args.num_samples)),
        seed=args.seed,
        height=args.height,
        width=args.width,
        device=device,
        dtype=dtype,
    )
    adapter = None if args.toy_smoke else load_object(args.adapter)(args, device=device, dtype=dtype)
    if args.toy_smoke:
        student = ToyXPredNet().to(device=device, dtype=dtype)
        state = torch.load(args.checkpoint, map_location=device)
        student.load_state_dict(state["state_dict"])
        student.eval()
        if args.prediction_type == "x":
            x_latent = sample_with_xpred_student(student, eps_latent, sigmas, eps_floor=args.eps_floor)
        else:
            x_latent = sample_with_vpred_student(student, eps_latent, sigmas)
    else:
        student = adapter.load_student_xpred(init_checkpoint=args.checkpoint)
        student.to(device=device, dtype=dtype).eval()
        student_forward = lambda z, sigma: adapter.student_forward_xpred(student, z, sigma, {})
        if args.prediction_type == "x":
            x_latent = sample_with_xpred_student(student_forward, eps_latent, sigmas, eps_floor=args.eps_floor)
        else:
            x_latent = sample_with_vpred_student(student_forward, eps_latent, sigmas)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x_latent": x_latent.detach().cpu(), "sigmas": sigmas.detach().cpu()}, output)
    print(f"saved sampled latent tensor to {output}")
    image_paths: list[Path] = []
    if args.decode_sample_images:
        if args.toy_smoke:
            raise ValueError("decode_sample_images requires the real Anima adapter and VAE; toy_smoke only produces latent tensors")
        image_dir = Path(args.sample_image_dir) if args.sample_image_dir else output.with_suffix("")
        image_paths = adapter.decode_latents_to_images(x_latent, image_dir, prefix=args.sample_image_prefix)
        print(f"saved {len(image_paths)} decoded sample image(s) to {image_dir}")

    wandb_run = init_wandb(args)
    try:
        if args.wandb_log_sample_images:
            log_sample_images_to_wandb(wandb_run, image_paths, args.prompt)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    del eps_latent, sigmas, x_latent
    if "student" in locals():
        del student
    if adapter is not None:
        del adapter
    release_cuda_memory()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="TOML config file. When set, normal experiment options are loaded from the file.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed_precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--flow_shift", type=float, default=DEFAULT_FLOW_SHIFT)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--toy_smoke", action="store_true", help="Run a local toy model instead of requiring an Anima adapter.")
    parser.add_argument(
        "--adapter",
        default="rum_xpred.adapters.anima_sd_scripts:create_adapter",
        help="Adapter factory in module:function form for real Anima integration.",
    )
    parser.add_argument("--dit", default=None, help="Anima DiT checkpoint for real adapter runs.")
    parser.add_argument("--student_init", default=None, help="Initial x-pred student checkpoint; defaults to --dit.")
    parser.add_argument("--text_encoder", default=None, help="Qwen3 text encoder path for real adapter runs.")
    parser.add_argument("--vae", default=None, help="Optional Anima/Qwen VAE path, reserved for preview decode.")
    parser.add_argument("--vae_spatial_chunk_size", type=int, default=None)
    parser.add_argument("--vae_disable_cache", action="store_true")
    parser.add_argument("--negative_prompt", default="", help="Negative prompt for real adapter CFG.")
    parser.add_argument("--attn_mode", default="torch", choices=["torch", "flash", "sageattn", "xformers"])
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8_scaled", action="store_true")
    parser.add_argument("--text_encoder_cpu", action="store_true")


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--wandb_project", default="rum-anima-xpred")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_mode", default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_notes", default=None)
    parser.add_argument("--wandb_log_config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb_metrics_file", default=None)
    parser.add_argument("--wandb_metrics_log_every", type=int, default=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RUM reflow experiment for converting Anima FM v-pred DiT to latent x-pred.")
    parser.add_argument("--config", help="TOML config file. If no subcommand is given, the config command is used.")
    subparsers = parser.add_subparsers(dest="command")

    cache = subparsers.add_parser("build_cache", help="Generate eps_latent -> x_teacher_latent cache.")
    add_common_args(cache)
    cache.add_argument("--prompts")
    cache.add_argument("--cache_dir")
    cache.add_argument("--num_samples", type=int, default=None)
    cache.add_argument("--start_index", type=int, default=0)
    cache.add_argument("--cache_batch_size", type=int, default=1)
    cache.add_argument("--skip_existing", action="store_true", default=True)
    cache.add_argument("--bucket_enabled", action="store_true", help="Randomly assign samples to built-in fixed resolution buckets.")
    cache.add_argument("--width", type=int, default=1024)
    cache.add_argument("--height", type=int, default=1024)
    cache.add_argument("--teacher_steps", type=int, default=DEFAULT_TEACHER_STEPS)
    cache.add_argument("--teacher_cfg", type=float, default=DEFAULT_TEACHER_CFG)
    cache.add_argument("--teacher_lora", default=None)
    cache.add_argument("--teacher_lora_weight", type=float, default=1.0)
    cache.set_defaults(func=build_cache)

    train = subparsers.add_parser("train_xpred", help="Train student to predict cached clean Anima latents.")
    add_common_args(train)
    train.add_argument("--cache_dir")
    train.add_argument("--output_dir")
    train.add_argument("--prediction_type", default="x", choices=["x", "v"])
    train.add_argument("--global_step_offset", type=int, default=0, help=argparse.SUPPRESS)
    train.add_argument("--optimizer_state", default=None, help="Optional optimizer state checkpoint to resume AdamW moments.")
    train.add_argument("--max_train_steps", type=int, default=None)
    train.add_argument("--num_train_epochs", type=float, default=1.0)
    train.add_argument("--train_batch_size", type=int, default=1)
    train.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    train.add_argument("--lr_scheduler", default="constant", choices=["constant", "cosine"])
    train.add_argument("--lr_warmup_steps", type=int, default=0)
    train.add_argument("--lr_cosine_min", type=float, default=0.1)
    train.add_argument("--lr_scheduler_total_steps", type=int, default=None)
    train.add_argument("--weight_decay", type=float, default=0.0)
    train.add_argument("--adam_beta1", type=float, default=0.9)
    train.add_argument("--adam_beta2", type=float, default=0.999)
    train.add_argument("--adam_epsilon", type=float, default=1e-8)
    train.add_argument("--max_grad_norm", type=float, default=1.0)
    train.add_argument("--sigma_min_train", type=float, default=DEFAULT_SIGMA_MIN_TRAIN)
    train.add_argument("--shuffle_cache", action="store_true", default=True)
    train.add_argument("--drop_last", action="store_true")
    train.add_argument("--log_every", type=int, default=1)
    train.add_argument("--save_every_steps", type=int, default=None)
    train.add_argument("--checkpoints_total_limit", type=int, default=None)
    train.add_argument("--gradient_checkpointing", action="store_true")
    train.add_argument("--gradient_checkpointing_cpu_offload", action="store_true")
    train.add_argument("--gradient_checkpointing_unsloth_offload", action="store_true")
    train.add_argument("--sample_every_steps", type=int, default=0)
    train.add_argument("--sample_prompt", default="")
    train.add_argument("--sample_steps", type=int, default=DEFAULT_TEACHER_STEPS)
    train.add_argument("--sample_num_samples", type=int, default=1)
    train.add_argument("--sample_eps_floor", type=float, default=DEFAULT_EPS_FLOOR)
    train.add_argument("--sample_width", type=int, default=None)
    train.add_argument("--sample_height", type=int, default=None)
    train.add_argument("--sample_seed", type=int, default=None)
    train.add_argument("--sample_output_dir", default=None)
    train.add_argument("--sample_decode_images", action="store_true")
    train.add_argument("--sample_image_prefix", default="train-sample")
    train.add_argument("--sample_wandb_log_images", action=argparse.BooleanOptionalAction, default=True)
    add_wandb_args(train)
    train.add_argument("--dry_run", action="store_true")
    train.set_defaults(func=train_xpred)

    sample = subparsers.add_parser("sample_xpred", help="Sample with the dedicated latent x-pred Euler sampler.")
    add_common_args(sample)
    sample.add_argument("--checkpoint")
    sample.add_argument("--output")
    sample.add_argument("--prediction_type", default="x", choices=["x", "v"])
    sample.add_argument("--prompt", default="", help="Prompt for real adapter x-pred sampling.")
    sample.add_argument("--num_samples", type=int, default=1)
    sample.add_argument("--steps", type=int, default=DEFAULT_TEACHER_STEPS)
    sample.add_argument("--width", type=int, default=1024)
    sample.add_argument("--height", type=int, default=1024)
    sample.add_argument("--eps_floor", type=float, default=DEFAULT_EPS_FLOOR)
    sample.add_argument("--decode_sample_images", action="store_true")
    sample.add_argument("--sample_image_dir", default=None)
    sample.add_argument("--sample_image_prefix", default="sample")
    sample.add_argument("--wandb_log_sample_images", action=argparse.BooleanOptionalAction, default=True)
    add_wandb_args(sample)
    sample.set_defaults(func=sample_xpred)

    chunked_stage = subparsers.add_parser("chunked_rum", help="Build cache chunks and train after each chunk with resume manifest.")
    add_common_args(chunked_stage)
    add_wandb_args(chunked_stage)
    chunked_stage.add_argument("--chunk_root", default=None, help="Root directory for rolling cache chunks, train outputs, and manifest.")
    chunked_stage.add_argument("--total_samples", type=int, default=None, help="Total samples to process across all chunks.")
    chunked_stage.add_argument("--chunk_size", type=int, default=1024, help="Samples per cache/train chunk.")
    chunked_stage.add_argument("--start_index", type=int, default=0, help="Global prompt start index for the first chunk.")
    chunked_stage.add_argument("--max_chunks", type=int, default=None, help="Optional cap for debugging/resume.")
    chunked_stage.add_argument("--train_steps_per_chunk", type=int, default=None, help="Override [train_xpred].max_train_steps per chunk.")
    chunked_stage.add_argument("--delete_cache_after_train", action="store_true", help="Delete each chunk cache after its training finishes.")
    chunked_stage.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    chunked_stage.set_defaults(func=chunked_rum)

    args = parser.parse_args()
    if args.config:
        config_path = args.config
        config_args = config_to_namespace(load_toml_config(config_path), command_override=args.command)
        args = config_args
        args.config = str(Path(config_path).resolve())
        args.func = {
            "build_cache": build_cache,
            "train_xpred": train_xpred,
            "sample_xpred": sample_xpred,
            "chunked_rum": chunked_rum,
        }[args.command]
    elif not args.command:
        parser.error("a subcommand is required unless --config provides command")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.toy_smoke and (not args.dit and args.command in {"build_cache", "chunked_rum"}):
        parser.error("real Anima build_cache/chunked_rum requires --dit; use --toy_smoke for local smoke tests")
    if not args.toy_smoke and not args.text_encoder:
        parser.error("real Anima runs require --text_encoder; use --toy_smoke for local smoke tests")
    if args.command == "build_cache" and (not args.prompts or not args.cache_dir):
        parser.error("build_cache requires prompts and cache_dir")
    if args.command == "train_xpred" and (not args.cache_dir or not args.output_dir):
        parser.error("train_xpred requires cache_dir and output_dir")
    if args.command == "sample_xpred" and (not args.checkpoint or not args.output):
        parser.error("sample_xpred requires checkpoint and output")
    if hasattr(args, "prediction_type") and args.prediction_type not in {"x", "v"}:
        parser.error("prediction_type must be 'x' or 'v'")
    if hasattr(args, "global_step_offset") and args.global_step_offset < 0:
        parser.error("global_step_offset must be >= 0")
    if args.command == "chunked_rum":
        if not args.config:
            parser.error("chunked_rum requires --config")
        if not args.chunk_root:
            parser.error("chunked_rum requires chunk_root")
        if args.total_samples is None:
            parser.error("chunked_rum requires total_samples")
        if args.total_samples < 1:
            parser.error("total_samples must be >= 1")
        if args.chunk_size < 1:
            parser.error("chunk_size must be >= 1")
        if args.max_chunks is not None and args.max_chunks < 1:
            parser.error("max_chunks must be >= 1")
        if args.train_steps_per_chunk is not None and args.train_steps_per_chunk < 1:
            parser.error("train_steps_per_chunk must be >= 1")
    if hasattr(args, "max_train_steps") and args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error("max_train_steps must be >= 1")
    if hasattr(args, "num_train_epochs") and args.num_train_epochs <= 0:
        parser.error("num_train_epochs must be > 0")
    if hasattr(args, "cache_batch_size") and args.cache_batch_size < 1:
        parser.error("cache_batch_size must be >= 1")
    if hasattr(args, "start_index") and args.start_index < 0:
        parser.error("start_index must be >= 0")
    if hasattr(args, "train_batch_size") and args.train_batch_size < 1:
        parser.error("train_batch_size must be >= 1")
    if hasattr(args, "gradient_accumulation_steps") and args.gradient_accumulation_steps < 1:
        parser.error("gradient_accumulation_steps must be >= 1")
    if hasattr(args, "lr_warmup_steps") and args.lr_warmup_steps < 0:
        parser.error("lr_warmup_steps must be >= 0")
    if hasattr(args, "lr_cosine_min") and not 0 <= args.lr_cosine_min <= 1:
        parser.error("lr_cosine_min must be in [0, 1]")
    if hasattr(args, "lr_scheduler_total_steps") and args.lr_scheduler_total_steps is not None and args.lr_scheduler_total_steps < 1:
        parser.error("lr_scheduler_total_steps must be >= 1")
    if hasattr(args, "wandb_metrics_log_every") and args.wandb_metrics_log_every < 0:
        parser.error("wandb_metrics_log_every must be >= 0")
    if hasattr(args, "sample_every_steps") and args.sample_every_steps < 0:
        parser.error("sample_every_steps must be >= 0")
    if hasattr(args, "sample_steps") and args.sample_steps < 1:
        parser.error("sample_steps must be >= 1")
    if hasattr(args, "sample_num_samples") and args.sample_num_samples < 1:
        parser.error("sample_num_samples must be >= 1")
    if hasattr(args, "sample_eps_floor") and args.sample_eps_floor <= 0:
        parser.error("sample_eps_floor must be > 0")
    if getattr(args, "sample_width", None) is not None and args.sample_width < 8:
        parser.error("sample_width must be >= 8")
    if getattr(args, "sample_height", None) is not None and args.sample_height < 8:
        parser.error("sample_height must be >= 8")
    if getattr(args, "vae_spatial_chunk_size", None) is not None and args.vae_spatial_chunk_size < 1:
        parser.error("vae_spatial_chunk_size must be >= 1")
    if (
        args.command == "train_xpred"
        and getattr(args, "sample_every_steps", 0) > 0
        and getattr(args, "sample_decode_images", False)
        and not args.toy_smoke
        and not args.vae
    ):
        parser.error("sample_decode_images during train_xpred requires --vae / [common].vae")
    if args.command == "sample_xpred" and getattr(args, "decode_sample_images", False) and not args.toy_smoke and not args.vae:
        parser.error("decode_sample_images requires --vae / [common].vae")
    if hasattr(args, "num_samples") and args.num_samples is not None and args.num_samples < 1:
        parser.error("num_samples must be >= 1")
    if hasattr(args, "teacher_lora_weight") and args.teacher_lora_weight < 0:
        parser.error("teacher_lora_weight must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
