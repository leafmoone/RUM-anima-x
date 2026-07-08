#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
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
    load_object,
    make_shifted_sigma_schedule,
    make_toy_teacher_endpoint,
    reflow_training_target,
    sample_train_sigmas,
    sample_with_mixed_velocity,
    sample_with_vpred_student,
    sample_with_xpred_student,
    save_xpred_cache_sample,
    xpred_to_anima_v,
)
from rum_xpred.cache_batches import (
    DEFAULT_CACHE_BUCKETS,
    CacheBucketBatchCursor,
    MultiCacheBucketBatchCursor,
    allocate_weighted_counts,
    bucket_cache_path,
    cache_source_indices_for_paths,
    cache_source_name,
    chunked,
    choose_cache_bucket,
    collect_cache_buckets,
    loss_by_cache_source,
    make_seeded_eps_batch,
    merge_cache_samples,
    prediction_to_x,
    prune_checkpoints,
    read_prompts,
    reflow_loss,
    unique_cache_source_names,
    x_mse_by_cache_source,
)
from rum_xpred.chunking import (
    ChunkPlan,
    chunk_cache_dirs_for_prompt_sets,
    chunk_train_steps,
    completed_chunk_ids,
    link_repeated_prompt_set_cache,
    load_chunk_manifest,
    make_chunk_plan,
    make_prompt_set_chunk_plan,
    normalize_prompt_sets,
    prompt_set_cache_dir_for_plan,
    prompt_set_cache_scope,
    prompt_set_chunk_name,
    prompt_set_effective_total,
    prompt_set_mix_weights,
    prompt_set_weight_total,
    prompt_set_slice_for_plan,
    prompt_set_slices_for_plan,
    resolve_chunk_optimizer_state,
    resolve_chunk_student_init,
    update_chunk_manifest,
)
from rum_xpred.config import config_to_namespace, load_toml_config
from rum_xpred.train_schedule import (
    lr_scale_for_step,
    planned_chunk_train_steps,
    planned_prompt_set_cache_sample_count,
    planned_prompt_set_train_steps,
    resolve_max_train_steps,
    set_optimizer_lr,
)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError("mixed_precision must be bf16, fp16, or fp32")




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


def init_tracker(args: argparse.Namespace):
    return None


def log_sample_images_to_tracker(
    tracker_run,
    image_paths: list[Path],
    prompt: str,
    step: int | None = None,
    *,
    key: str = "sample/images",
) -> None:
    if tracker_run is None or not image_paths:
        return

    images = [{"path": str(path), "caption": sample_image_caption(path, prompt)} for path in image_paths]
    payload = {key: images}
    if step is not None:
        step_key = f"{key[:-len('/images')]}/step" if key.endswith("/images") else f"{key}/step"
        payload[step_key] = step
    current_step = getattr(tracker_run, "step", None)
    if step is None or (current_step is not None and step < current_step):
        if step is not None and current_step is not None and step < current_step:
            print(f"tracker media step {step} is behind current step {current_step}; logging without explicit step")
        tracker_run.log(payload)
    else:
        tracker_run.log(payload, step=step)


def sample_image_caption(path: Path, prompt: str) -> str:
    parent = path.parent.name
    if parent == "lora":
        mode = "x+lora"
    elif parent == "fm_lora":
        mode = "fm+lora"
    elif parent == "x_lora":
        mode = "x+lora"
    elif parent in {"fm", "x", "teacher_sanity"}:
        mode = parent
    elif path.name.startswith("teacher-baseline"):
        mode = "teacher_baseline"
    else:
        mode = "x"
    sample_match = re.search(r"-(\d+)\.[^.]+$", path.name)
    sample_suffix = f" #{sample_match.group(1)}" if sample_match else ""
    label = f"{mode}{sample_suffix}"
    return f"{label}\n{prompt}" if prompt else label


def find_image_files(path: str | Path | None) -> list[Path]:
    if not path:
        return []
    root = Path(path)
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else []
    return sorted(
        file
        for file in root.rglob("*")
        if file.is_file()
        and file.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        and ".ipynb_checkpoints" not in file.parts
    )


def maybe_import_compare_baseline(args: argparse.Namespace, tracker_run=None) -> list[Path]:
    if getattr(args, "sample_compare_baseline_imported", False):
        return []
    source_dir = getattr(args, "sample_compare_baseline_source_dir", None)
    if not source_dir:
        args.sample_compare_baseline_imported = True
        return []
    source_images = find_image_files(source_dir)
    if not source_images:
        print(f"sample_compare baseline source has no images: {source_dir}")
        args.sample_compare_baseline_imported = True
        return []
    output_root = Path(getattr(args, "sample_compare_baseline_output_dir", None) or REPO_ROOT / "compare-baseline")
    output_root.mkdir(parents=True, exist_ok=True)
    marker_path = output_root / ".tracker_uploaded"
    copied: list[Path] = []
    for index, source in enumerate(source_images):
        target = output_root / f"teacher-baseline-{index:04d}{source.suffix.lower()}"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(target)
    print(f"imported {len(copied)} compare baseline image(s) to {output_root}")
    if getattr(args, "sample_compare_baseline_tracker_log_images", True) and not marker_path.exists():
        prompt = getattr(args, "sample_compare_prompt", None) or getattr(args, "sample_prompt", "")
        log_sample_images_to_tracker(tracker_run, copied, prompt, key="sample_compare/baseline")
        marker_path.write_text("uploaded\n", encoding="utf-8")
    args.sample_compare_baseline_imported = True
    return copied


def read_tracker_metrics_file(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"tracker metrics file must contain a JSON object: {metrics_path}")
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


def make_stage_args(config_path: str, command: str) -> argparse.Namespace:
    args = config_to_namespace(load_toml_config(config_path), command_override=command)
    args.config = str(Path(config_path).resolve())
    args.func = {
        "build_cache": build_cache,
        "train_xpred": train_xpred,
        "sample_xpred": sample_xpred,
        "sample_compare": sample_compare,
    }[command]
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


def prepare_memory_for_training_generation(optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    release_cuda_memory()


def clear_adapter_teacher(adapter) -> None:
    if adapter is not None and getattr(adapter, "teacher", None) is not None:
        adapter.teacher.to("cpu")
        adapter.teacher = None


def load_compare_teacher(adapter, args: argparse.Namespace, *, checkpoint: str | None, lora: str | None, lora_weight: float | None):
    original_dit = getattr(args, "dit", None)
    original_lora = getattr(args, "teacher_lora", None)
    original_lora_weight = getattr(args, "teacher_lora_weight", 1.0)
    if checkpoint:
        args.dit = checkpoint
    args.teacher_lora = lora or None
    args.teacher_lora_weight = 1.0 if lora_weight is None else float(lora_weight)
    clear_adapter_teacher(adapter)
    release_cuda_memory()
    try:
        return adapter._load_teacher()
    finally:
        args.dit = original_dit
        args.teacher_lora = original_lora
        args.teacher_lora_weight = original_lora_weight


def load_lora_student_copy(
    *,
    student,
    adapter,
    args: argparse.Namespace,
    checkpoint_path: Path,
    lora: str | None,
    lora_weight: float | None,
):
    if not lora:
        return None
    original_lora = getattr(args, "teacher_lora", None)
    original_lora_weight = getattr(args, "teacher_lora_weight", 1.0)
    original_adapter_student = getattr(adapter, "student", None)
    args.teacher_lora = lora
    args.teacher_lora_weight = 1.0 if lora_weight is None else float(lora_weight)
    release_cuda_memory()
    try:
        adapter.save_student_xpred(student, checkpoint_path)
        reference_parameter = next(student.parameters(), None)
        reference_device = reference_parameter.device if reference_parameter is not None else torch.device(getattr(args, "device", "cpu"))
        reference_dtype = reference_parameter.dtype if reference_parameter is not None else dtype_from_name(getattr(args, "mixed_precision", "fp32"))
        lora_student = adapter.load_student_xpred(init_checkpoint=str(checkpoint_path))
        lora_student.to(device=reference_device, dtype=reference_dtype).eval()
        return lora_student
    finally:
        args.teacher_lora = original_lora
        args.teacher_lora_weight = original_lora_weight
        adapter.student = original_adapter_student


def should_run_train_sample(args: argparse.Namespace, global_step: int) -> bool:
    every = getattr(args, "sample_every_steps", 0) or 0
    total_step = getattr(args, "global_step_offset", 0) + global_step
    return every > 0 and total_step > 0 and total_step % every == 0


def should_run_train_compare(args: argparse.Namespace, global_step: int) -> bool:
    every = getattr(args, "sample_compare_every_steps", 0) or 0
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
    tracker_run=None,
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
    lora_student = None
    lora_checkpoint_path: Path | None = None
    lora_latent = None
    lora_sigmas = None
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
                sample_cfg = getattr(args, "sample_cfg", 1.0)
                cond_embed, uncond_embed = adapter._encode_prompts([args.sample_prompt], cfg=sample_cfg, anima_model=student)
                sample_conditioning = {
                    "prompt_embeds": cond_embed.detach().to(dtype=dtype),
                    "negative_prompt_embeds": uncond_embed.detach().to(dtype=dtype),
                }
                student_forward = lambda z, sigma: adapter.student_forward_xpred(
                    student,
                    z,
                    sigma,
                    sample_conditioning,
                    guidance_scale=sample_cfg,
                )
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
            sample_lora = getattr(args, "sample_lora", None)
            if sample_lora == "__inherit__":
                sample_lora = None
            if sample_lora:
                lora_steps = args.sample_lora_steps if args.sample_lora_steps is not None else args.sample_steps
                lora_eps_floor = args.sample_lora_eps_floor if args.sample_lora_eps_floor is not None else args.sample_eps_floor
                lora_sigmas = make_shifted_sigma_schedule(lora_steps, args.flow_shift, device=device, dtype=dtype)
                sample_dir_for_lora = Path(args.sample_output_dir) if args.sample_output_dir else Path(args.output_dir) / "train-samples"
                lora_checkpoint_path = sample_dir_for_lora / "_tmp" / f"student-step-{total_step:06d}.safetensors"
                lora_student = load_lora_student_copy(
                    student=student,
                    adapter=adapter,
                    args=args,
                    checkpoint_path=lora_checkpoint_path,
                    lora=sample_lora,
                    lora_weight=getattr(args, "sample_lora_weight", None),
                )
                if lora_student is not None:
                    lora_cfg = getattr(args, "sample_lora_cfg", None)
                    if lora_cfg is None:
                        lora_cfg = sample_cfg
                    lora_cond_embed, lora_uncond_embed = adapter._encode_prompts([args.sample_prompt], cfg=lora_cfg, anima_model=lora_student)
                    lora_conditioning = {
                        "prompt_embeds": lora_cond_embed.detach().to(dtype=dtype),
                        "negative_prompt_embeds": lora_uncond_embed.detach().to(dtype=dtype),
                    }
                    lora_forward = lambda z, sigma: adapter.student_forward_xpred(
                        lora_student,
                        z,
                        sigma,
                        lora_conditioning,
                        guidance_scale=lora_cfg,
                    )
                    if args.prediction_type == "x":
                        lora_latent = sample_with_xpred_student(lora_forward, eps_latent.clone(), lora_sigmas, eps_floor=lora_eps_floor)
                    else:
                        lora_latent = sample_with_vpred_student(lora_forward, eps_latent.clone(), lora_sigmas)
    finally:
        if lora_student is not None:
            lora_student.to("cpu")
            del lora_student
        if lora_checkpoint_path is not None and lora_checkpoint_path.exists():
            lora_checkpoint_path.unlink()
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
    if lora_latent is not None:
        lora_latent_path = latent_dir / f"{args.prediction_type}pred-lora-step-{total_step:06d}.pt"
        torch.save(
            {
                "x_latent": lora_latent.detach().cpu(),
                "sigmas": (lora_sigmas if lora_sigmas is not None else sigmas).detach().cpu(),
                "step": total_step,
                "local_step": global_step,
            },
            lora_latent_path,
        )
    image_paths: list[Path] = []
    if args.sample_decode_images:
        if args.toy_smoke:
            print("sample_decode_images=true ignored for toy_smoke training samples")
        else:
            image_dir = sample_dir / "images" / f"step-{total_step:06d}"
            image_paths = adapter.decode_latents_to_images(x_latent, image_dir, prefix=args.sample_image_prefix)
            if lora_latent is not None:
                image_paths.extend(adapter.decode_latents_to_images(lora_latent, image_dir / "lora", prefix=f"{args.sample_image_prefix}-lora"))
            print(f"saved {len(image_paths)} training sample image(s) to {image_dir}")
            if args.sample_tracker_log_images:
                log_sample_images_to_tracker(tracker_run, image_paths, args.sample_prompt, step=total_step)
    print(f"saved training sample latent tensor to {latent_path}")
    del x_latent, sigmas, eps_latent
    if lora_latent is not None:
        del lora_latent
    if lora_sigmas is not None:
        del lora_sigmas
    release_cuda_memory()
    return image_paths


@torch.no_grad()
def sample_compare_from_training_student(
    *,
    student,
    adapter,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
    tracker_run=None,
) -> list[Path]:
    if args.prediction_type != "x":
        raise ValueError("training sample_compare currently requires prediction_type='x'")
    total_step = getattr(args, "global_step_offset", 0) + global_step
    prompt = getattr(args, "sample_compare_prompt", None) or getattr(args, "sample_prompt", "")
    width = args.sample_compare_width or args.sample_width or getattr(args, "width", None) or 1024
    height = args.sample_compare_height or args.sample_height or getattr(args, "height", None) or 1024
    seed = args.sample_compare_seed if args.sample_compare_seed is not None else (
        args.sample_seed if args.sample_seed is not None else args.seed
    )
    sigmas = make_shifted_sigma_schedule(args.sample_compare_steps, args.flow_shift, device=device, dtype=dtype)
    eps_latent = make_seeded_eps_batch(
        list(range(args.sample_compare_num_samples)),
        seed=seed,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
    )
    was_training = student.training
    student.eval()
    release_cuda_memory()
    original_prompt = None
    results: dict[str, torch.Tensor] = {}
    lora_student = None
    lora_checkpoint_path: Path | None = None
    sanity_teacher = None
    try:
        if args.toy_smoke:
            student_forward = student
        else:
            if not prompt:
                raise ValueError("training sample_compare requires sample_compare_prompt or sample_prompt for real Anima runs")
            original_prompt = getattr(args, "prompt", None)
            args.prompt = prompt
            compare_cfg = getattr(args, "sample_compare_cfg", 1.0)
            cond_embed, uncond_embed = adapter._encode_prompts([prompt], cfg=compare_cfg, anima_model=student)
            student_conditioning = {
                "prompt_embeds": cond_embed.detach().to(dtype=dtype),
                "negative_prompt_embeds": uncond_embed.detach().to(dtype=dtype),
            }
            student_forward = lambda z, sigma: adapter.student_forward_xpred(
                student,
                z,
                sigma,
                student_conditioning,
                guidance_scale=compare_cfg,
            )
        results["fm"] = sample_with_vpred_student(student_forward, eps_latent.clone(), sigmas)
        results["x"] = sample_with_xpred_student(
            student_forward,
            eps_latent.clone(),
            sigmas,
            eps_floor=args.sample_compare_eps_floor,
        )
        compare_lora = getattr(args, "sample_compare_lora", None)
        if compare_lora == "__inherit__":
            compare_lora = None
        if compare_lora and not args.toy_smoke:
            compare_dir = Path(args.sample_compare_output_dir) if args.sample_compare_output_dir else Path(args.output_dir) / "train-compare-samples"
            lora_checkpoint_path = compare_dir / "_tmp" / f"student-step-{total_step:06d}.safetensors"
            lora_student = load_lora_student_copy(
                student=student,
                adapter=adapter,
                args=args,
                checkpoint_path=lora_checkpoint_path,
                lora=compare_lora,
                lora_weight=getattr(args, "sample_compare_lora_weight", None),
            )
            if lora_student is not None:
                compare_lora_cfg = getattr(args, "sample_compare_lora_cfg", None)
                if compare_lora_cfg is None:
                    compare_lora_cfg = compare_cfg
                lora_cond_embed, lora_uncond_embed = adapter._encode_prompts([prompt], cfg=compare_lora_cfg, anima_model=lora_student)
                lora_conditioning = {
                    "prompt_embeds": lora_cond_embed.detach().to(dtype=dtype),
                    "negative_prompt_embeds": lora_uncond_embed.detach().to(dtype=dtype),
                }
                lora_student_forward = lambda z, sigma: adapter.student_forward_xpred(
                    lora_student,
                    z,
                    sigma,
                    lora_conditioning,
                    guidance_scale=compare_lora_cfg,
                )
                results["fm_lora"] = sample_with_vpred_student(lora_student_forward, eps_latent.clone(), sigmas)
                results["x_lora"] = sample_with_xpred_student(
                    lora_student_forward,
                    eps_latent.clone(),
                    sigmas,
                    eps_floor=args.sample_compare_eps_floor,
                )
        if getattr(args, "sample_compare_teacher_sanity", False) and not args.toy_smoke:
            sanity_lora = getattr(args, "sample_compare_teacher_sanity_lora", None)
            if sanity_lora == "__inherit__":
                sanity_lora = None
            sanity_teacher = load_compare_teacher(
                adapter,
                args,
                checkpoint=getattr(args, "dit", None),
                lora=sanity_lora,
                lora_weight=getattr(args, "sample_compare_teacher_sanity_lora_weight", None),
            )
            sanity_teacher.to(device=device, dtype=dtype).eval()
            sanity_latents = []
            for sample_index in range(eps_latent.shape[0]):
                sanity_latent, _ = adapter.teacher_sample_latent(
                    prompt,
                    eps_latent[sample_index : sample_index + 1],
                    sigmas,
                    guidance_scale=1.0,
                )
                sanity_latents.append(sanity_latent)
            results["teacher_sanity"] = torch.cat(sanity_latents, dim=0)
    finally:
        if not args.toy_smoke:
            if original_prompt is None:
                try:
                    delattr(args, "prompt")
                except AttributeError:
                    pass
                else:
                    args.prompt = original_prompt
        if lora_student is not None:
            lora_student.to("cpu")
            del lora_student
        if lora_checkpoint_path is not None and lora_checkpoint_path.exists():
            lora_checkpoint_path.unlink()
        if sanity_teacher is not None:
            sanity_teacher.to("cpu")
            del sanity_teacher
        clear_adapter_teacher(adapter)
        if was_training:
            student.train()

    compare_dir = Path(args.sample_compare_output_dir) if args.sample_compare_output_dir else Path(args.output_dir) / "train-compare-samples"
    step_dir = compare_dir / f"step-{total_step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    latent_path = step_dir / "compare-latents.pt"
    torch.save(
        {
            "latents": {key: value.detach().cpu() for key, value in results.items()},
            "sigmas": sigmas.detach().cpu(),
            "modes": list(results),
            "step": total_step,
            "local_step": global_step,
        },
        latent_path,
    )
    image_paths: list[Path] = []
    if args.sample_compare_decode_images:
        if args.toy_smoke:
            print("sample_compare_decode_images=true ignored for toy_smoke training compare samples")
        else:
            for key, latent in results.items():
                image_paths.extend(
                    adapter.decode_latents_to_images(
                        latent,
                        step_dir / "images" / key,
                        prefix=f"{args.sample_compare_image_prefix}-{key}",
                    )
                )
            print(f"saved {len(image_paths)} training compare image(s) to {step_dir / 'images'}")
            if args.sample_compare_tracker_log_images:
                log_sample_images_to_tracker(tracker_run, image_paths, prompt, step=total_step, key="sample_compare/images")
    print(f"saved training compare latent tensor to {latent_path}")
    del results, sigmas, eps_latent
    release_cuda_memory()
    return image_paths


def build_cache(args: argparse.Namespace) -> None:
    prompt_sets = normalize_prompt_sets(args)
    if prompt_sets is not None:
        for prompt_set in prompt_sets:
            set_args = argparse.Namespace(**vars(args))
            set_args.prompt_sets = None
            set_args.prompts = prompt_set["prompts"]
            set_args.cache_dir = prompt_set["cache_dir"]
            set_args.start_index = prompt_set["start_index"]
            set_args.num_samples = prompt_set["num_samples"]
            print(
                f"building prompt set {prompt_set['name']}: "
                f"prompts={set_args.prompts} cache_dir={set_args.cache_dir} "
                f"start_index={set_args.start_index} num_samples={set_args.num_samples}"
            )
            build_cache(set_args)
        return

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


def cache_sample_index_from_path(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("sample-"):
        return None
    try:
        return int(stem.removeprefix("sample-"))
    except ValueError:
        return None


def validate_prepared_cache_dirs(cache_dirs: list[str], expected_sample_indexes: dict[str, set[int]] | None = None) -> None:
    expected_sample_indexes = expected_sample_indexes or {}
    for cache_dir in cache_dirs:
        path = Path(cache_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"prepared cache dir is missing: {path}")
        buckets = collect_cache_buckets(path)
        if not any(bucket.files for bucket in buckets):
            raise ValueError(f"prepared cache dir has no safetensors files: {path}")
        expected = expected_sample_indexes.get(cache_dir)
        if expected is None:
            continue
        actual = {
            index
            for bucket in buckets
            for sample_path in bucket.files
            for index in [cache_sample_index_from_path(sample_path)]
            if index is not None
        }
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details = []
            if missing:
                preview = ", ".join(str(index) for index in missing[:10])
                details.append(f"missing: {preview}{' ...' if len(missing) > 10 else ''}")
            if extra:
                preview = ", ".join(str(index) for index in extra[:10])
                details.append(f"extra: {preview}{' ...' if len(extra) > 10 else ''}")
            raise ValueError(
                f"prepared cache dir is incomplete for expected range "
                f"{min(expected)}-{max(expected)}: {path} ({'; '.join(details)})"
            )


def chunked_rum(args: argparse.Namespace) -> None:
    if not getattr(args, "config", None):
        raise ValueError("chunked_rum requires --config so it can reuse [build_cache] and [train_xpred]")
    chunk_root = Path(args.chunk_root)
    chunk_root.mkdir(parents=True, exist_ok=True)
    manifest_path = chunk_root / "chunk-manifest.json"
    base_train_args = make_stage_args(args.config, "train_xpred")
    base_cache_args = make_stage_args(args.config, "build_cache")
    prompt_sets = normalize_prompt_sets(base_cache_args)
    if (
        prompt_sets is not None
        and any(prompt_set_cache_scope(prompt_set) == "all" for prompt_set in prompt_sets)
        and not getattr(args, "prepared_cache_only", False)
    ):
        raise ValueError("prompt_sets cache_scope='all' requires chunked_rum.prepared_cache_only=true")
    if prompt_sets is not None and args.total_samples is None:
        plans = make_prompt_set_chunk_plan(prompt_sets=prompt_sets, chunk_size=args.chunk_size, max_chunks=args.max_chunks)
        plan_source = "prompt_sets"
    else:
        if args.total_samples is None:
            raise ValueError("chunked_rum requires total_samples when build_cache.prompt_sets is not enabled")
        plans = make_chunk_plan(
            start_index=args.start_index,
            total_samples=args.total_samples,
            chunk_size=args.chunk_size,
            max_chunks=args.max_chunks,
        )
        plan_source = "chunked_rum"
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
    if prompt_sets is not None and args.total_samples is None:
        planned_total_train_steps = planned_prompt_set_train_steps(base_train_args, args, plans, prompt_sets)
    else:
        cache_multiplier = len(prompt_sets) if prompt_sets is not None else 1
        planned_total_train_steps = planned_chunk_train_steps(base_train_args, args, plans, cache_multiplier=cache_multiplier)

    print(
        f"chunked_rum: {len(plans)} planned chunk(s), chunk_size={args.chunk_size}, "
        f"total_samples={args.total_samples if args.total_samples is not None else '<from prompt_sets>'}, "
        f"plan_source={plan_source}, resume={args.resume}"
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
            previous_manifest_chunk = manifest_by_chunk_id.get(plan.chunk_id - 1)
            global_step_offset += chunk_train_steps(completed_chunk, chunk_output_dir, previous_chunk=previous_manifest_chunk)
            continue

        cache_args = make_stage_args(args.config, "build_cache")
        cache_prompt_sets = normalize_prompt_sets(cache_args)
        if cache_prompt_sets is None:
            cache_args.cache_dir = str(chunk_cache_dir)
            cache_args.start_index = plan.start_index
            cache_args.num_samples = plan.num_samples
            cache_dirs = [str(chunk_cache_dir)]
            manifest_cache_value: str | list[str] = str(chunk_cache_dir)
            expected_cache_indexes = {str(chunk_cache_dir): set(range(plan.start_index, plan.start_index + plan.num_samples))}
        else:
            adjusted_sets = []
            repeat_link_slices: list[tuple[dict, dict]] = []
            for prompt_set in cache_prompt_sets:
                if prompt_set_cache_scope(prompt_set) == "all":
                    adjusted = dict(prompt_set)
                    adjusted["_cache_scope_all"] = True
                    slices = [adjusted]
                elif args.total_samples is None:
                    slices = prompt_set_slices_for_plan(prompt_set, plan)
                else:
                    adjusted = dict(prompt_set)
                    adjusted["start_index"] = int(prompt_set["start_index"]) + (plan.start_index - args.start_index)
                    adjusted["num_samples"] = plan.num_samples
                    adjusted["cache_chunk_offset"] = 0
                    slices = [adjusted]
                for adjusted in slices:
                    adjusted["cache_dir"] = prompt_set_cache_dir_for_plan(prompt_set, plan.chunk_id)
                    if adjusted.get("_cache_scope_all"):
                        adjusted_sets.append(adjusted)
                        continue
                    if int(adjusted.get("_repeat_cycle", 0)) > 0:
                        repeat_link_slices.append((prompt_set, adjusted))
                    else:
                        adjusted_sets.append(adjusted)
            if not adjusted_sets and not repeat_link_slices:
                print(f"{chunk_name}: no prompt set has samples for this chunk, skipping")
                continue
            cache_args.prompt_sets = adjusted_sets
            cache_dirs = list(
                dict.fromkeys(
                    [str(Path(prompt_set["cache_dir"])) for prompt_set in adjusted_sets]
                    + [str(Path(adjusted["cache_dir"])) for _prompt_set, adjusted in repeat_link_slices]
                )
            )
            expected_cache_indexes = {
                str(Path(prompt_set["cache_dir"])): set(
                    range(int(prompt_set["start_index"]), int(prompt_set["start_index"]) + int(prompt_set["num_samples"]))
                )
                for prompt_set in adjusted_sets
                if not prompt_set.get("_cache_scope_all")
            }
            manifest_cache_value = cache_dirs
        if getattr(args, "prepared_cache_only", False):
            validate_prepared_cache_dirs(cache_dirs, expected_cache_indexes)
            print(f"{chunk_name}: prepared cache ready; skipping build_cache")
        else:
            update_chunk_manifest(manifest_path, plan, status="building_cache", cache_dir=manifest_cache_value)
            print(f"{chunk_name}: building cache start_index={plan.start_index} num_samples={plan.num_samples}")
            if cache_prompt_sets is None:
                build_cache(cache_args)
                release_cuda_memory()
            else:
                if adjusted_sets:
                    build_cache(cache_args)
                    release_cuda_memory()
                for prompt_set, adjusted in repeat_link_slices:
                    linked, skipped_links = link_repeated_prompt_set_cache(
                        prompt_set=prompt_set,
                        adjusted=adjusted,
                        chunk_size=args.chunk_size,
                        seed=cache_args.seed,
                        bucket_enabled=bool(getattr(cache_args, "bucket_enabled", False)),
                        width=cache_args.width,
                        height=cache_args.height,
                        skip_existing=cache_args.skip_existing,
                    )
                    print(
                        f"{chunk_name}: linked repeat cache set={prompt_set['name']} "
                        f"start_index={adjusted['start_index']} num_samples={adjusted['num_samples']} "
                        f"linked={linked} skipped={skipped_links}"
                    )
        update_chunk_manifest(manifest_path, plan, status="cache_built", cache_dir=manifest_cache_value)

        train_args = make_stage_args(args.config, "train_xpred")
        if len(cache_dirs) == 1:
            train_args.cache_dir = cache_dirs[0]
            train_args.cache_dirs = None
        else:
            train_args.cache_dir = None
            train_args.cache_dirs = cache_dirs
            if getattr(train_args, "cache_mix_mode", "single") == "single":
                train_args.cache_mix_mode = "batch_weighted"
            if prompt_sets is not None and args.total_samples is None and getattr(train_args, "cache_mix_weights", None) is None:
                weight_by_dir = {
                    prompt_set_cache_dir_for_plan(prompt_set, plan.chunk_id): prompt_set_weight_total(prompt_set)
                    for prompt_set in cache_prompt_sets
                }
                train_args.cache_mix_weights = [float(weight_by_dir[cache_dir]) for cache_dir in cache_dirs]
        train_args.output_dir = str(chunk_output_dir)
        train_args.student_init = resolve_chunk_student_init(args, previous_checkpoint)
        train_args.optimizer_state = resolve_chunk_optimizer_state(args, previous_optimizer_state)
        train_args.global_step_offset = global_step_offset
        if train_args.lr_scheduler_total_steps is None:
            train_args.lr_scheduler_total_steps = planned_total_train_steps
        if args.train_steps_per_chunk is not None:
            train_args.max_train_steps = args.train_steps_per_chunk
        elif (
            prompt_sets is not None
            and args.total_samples is None
            and train_args.max_train_steps is None
            and any(prompt_set_cache_scope(prompt_set) == "all" for prompt_set in cache_prompt_sets)
        ):
            planned_cache_samples = planned_prompt_set_cache_sample_count(plan, plans, cache_prompt_sets)
            train_args.max_train_steps = resolve_max_train_steps(train_args, planned_cache_samples)
            print(
                f"{chunk_name}: planned train steps from proportional global cache samples="
                f"{planned_cache_samples} -> {train_args.max_train_steps}"
            )
        update_chunk_manifest(manifest_path, plan, status="training", output_dir=str(chunk_output_dir))
        print(
            f"{chunk_name}: training cache_dir={cache_dirs if len(cache_dirs) > 1 else cache_dirs[0]} "
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
            total_completed_steps=global_step_offset
            + (getattr(train_args, "completed_train_steps", 0) or getattr(train_args, "resolved_max_train_steps", 0)),
        )
        global_step_offset += getattr(train_args, "completed_train_steps", 0) or getattr(train_args, "resolved_max_train_steps", 0)
        if args.delete_cache_after_train:
            for cache_dir in cache_dirs:
                shutil.rmtree(cache_dir, ignore_errors=True)
            update_chunk_manifest(manifest_path, plan, cache_deleted=True)
            print(f"{chunk_name}: deleted cache {cache_dirs}")
    print(f"chunked_rum complete; manifest: {manifest_path}")


def train_xpred(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.mixed_precision)
    cache_dirs = list(getattr(args, "cache_dirs", None) or [])
    if cache_dirs:
        cache_bucket_sets = [collect_cache_buckets(cache_dir) for cache_dir in cache_dirs]
        cache_files = [path for buckets in cache_bucket_sets for bucket in buckets for path in bucket.files]
        cache_source_names = unique_cache_source_names(cache_dirs)
    else:
        cache_buckets = collect_cache_buckets(args.cache_dir)
        cache_bucket_sets = [cache_buckets]
        cache_files = [path for bucket in cache_buckets for path in bucket.files]
        cache_source_names = []
    if not cache_files:
        cache_label = cache_dirs if cache_dirs else args.cache_dir
        raise ValueError(f"no safetensors cache files found in {cache_label}")
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
    if cache_dirs:
        if getattr(args, "cache_mix_mode", "single") == "single":
            args.cache_mix_mode = "batch_weighted"
        if getattr(args, "cache_mix_mode", "batch_weighted") != "batch_weighted":
            raise ValueError("cache_dirs currently require cache_mix_mode='batch_weighted'")
        batch_cursor = MultiCacheBucketBatchCursor(
            cache_bucket_sets,
            args.train_batch_size,
            weights=getattr(args, "cache_mix_weights", None),
            shuffle=args.shuffle_cache,
            seed=args.seed,
            drop_last=args.drop_last,
        )
        cache_bucket_count = sum(len(buckets) for buckets in cache_bucket_sets)
    else:
        batch_cursor = CacheBucketBatchCursor(
            cache_bucket_sets[0],
            args.train_batch_size,
            shuffle=args.shuffle_cache,
            seed=args.seed,
            drop_last=args.drop_last,
        )
        cache_bucket_count = len(cache_bucket_sets[0])
    resolved_max_train_steps = resolve_max_train_steps(args, len(cache_files))
    args.resolved_max_train_steps = resolved_max_train_steps
    if args.max_train_steps is None:
        print(
            "auto max_train_steps="
            f"{resolved_max_train_steps} from cache_samples={len(cache_files)}, "
            f"cache_buckets={cache_bucket_count}, "
            f"effective_batch_size={args.train_batch_size * args.gradient_accumulation_steps}, "
            f"num_train_epochs={args.num_train_epochs}"
        )
    tracker_run = init_tracker(args)

    losses: list[float] = []
    completed_train_steps = 0
    try:
        for step in range(resolved_max_train_steps):
            global_step = step + 1
            total_step = getattr(args, "global_step_offset", 0) + global_step
            should_log_console = global_step % args.log_every == 0
            should_log_tracker = tracker_run is not None
            should_compute_source_metrics = bool(cache_dirs) and (should_log_console or should_log_tracker)
            current_lr = args.learning_rate * lr_scale_for_step(args, total_step)
            set_optimizer_lr(optimizer, current_lr)
            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []
            micro_x_mses: list[float] = []
            micro_source_losses: dict[str, list[float]] = {name: [] for name in cache_source_names}
            micro_source_x_mses: dict[str, list[float]] = {name: [] for name in cache_source_names}
            for _ in range(args.gradient_accumulation_steps):
                batch_paths = batch_cursor.next()
                source_indices = cache_source_indices_for_paths(batch_paths, cache_dirs) if cache_dirs else []
                sample = merge_cache_samples(batch_paths, device=device, dtype=dtype)
                sigma = sample_train_sigmas(
                    sample["x_teacher_latent"].shape[0],
                    sigma_min_train=args.sigma_min_train,
                    flow_shift=args.flow_shift,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                    time_sampling=args.time_sampling,
                    logit_mean=args.time_sampling_logit_mean,
                    logit_std=args.time_sampling_logit_std,
                )
                z = (1 - sigma) * sample["x_teacher_latent"] + sigma * sample["eps_latent"]
                if args.toy_smoke:
                    prediction = student(z, sigma)
                else:
                    prediction = adapter.student_forward_xpred(student, z, sigma, sample["text_conditioning"])
                target = reflow_training_target(args.prediction_type, sample["x_teacher_latent"], sample["eps_latent"])
                loss_value = reflow_loss(
                    args.prediction_type,
                    args.loss_weighting,
                    prediction,
                    target,
                    z,
                    sigma,
                    args.loss_eps_floor,
                )
                if not torch.isfinite(loss_value):
                    raise FloatingPointError(f"non-finite {args.prediction_type}-pred loss")
                x_prediction = prediction_to_x(args.prediction_type, prediction, z, sigma)
                x_mse = torch.nn.functional.mse_loss(x_prediction.detach().float(), sample["x_teacher_latent"].detach().float())
                micro_x_mses.append(float(x_mse.detach()))
                if should_compute_source_metrics:
                    for source_index, source_loss in loss_by_cache_source(
                        prediction_type=args.prediction_type,
                        loss_weighting=args.loss_weighting,
                        prediction=prediction,
                        target=target,
                        z=z,
                        sigma=sigma,
                        eps_floor=args.loss_eps_floor,
                        source_indices=source_indices,
                        source_count=len(cache_dirs),
                    ).items():
                        micro_source_losses[cache_source_names[source_index]].append(source_loss)
                    for source_index, source_x_mse in x_mse_by_cache_source(
                        x_pred=x_prediction,
                        x_target=sample["x_teacher_latent"],
                        source_indices=source_indices,
                        source_count=len(cache_dirs),
                    ).items():
                        micro_source_x_mses[cache_source_names[source_index]].append(source_x_mse)
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
            x_mse = sum(micro_x_mses) / len(micro_x_mses)
            source_losses = {
                name: sum(values) / len(values)
                for name, values in micro_source_losses.items()
                if values
            }
            source_x_mses = {
                name: sum(values) / len(values)
                for name, values in micro_source_x_mses.items()
                if values
            }
            losses.append(float(loss))
            if tracker_run is not None:
                grad_norm_value = None if grad_norm is None else float(grad_norm)
                tracker_metrics = {
                    "train/loss": float(loss),
                    "train/x_mse": float(x_mse),
                    "train/lr": current_lr,
                    "train/grad_norm": grad_norm_value,
                    "grad_norm": grad_norm_value,
                    "train/effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
                }
                for name, source_loss in source_losses.items():
                    tracker_metrics[f"train/loss_by_cache/{name}"] = float(source_loss)
                for name, source_x_mse in source_x_mses.items():
                    tracker_metrics[f"train/x_mse_by_cache/{name}"] = float(source_x_mse)
                    if args.tracker_metrics_log_every > 0 and global_step % args.tracker_metrics_log_every == 0:
                        tracker_metrics.update(read_tracker_metrics_file(args.tracker_metrics_file))
                    tracker_run.log(tracker_metrics, step=total_step)
            if should_log_console:
                grad_text = "" if grad_norm is None else f" grad_norm={float(grad_norm):.6f}"
                offset_text = "" if getattr(args, "global_step_offset", 0) == 0 else f" total_step={total_step}"
                source_text = ""
                if source_losses:
                    source_text = " " + " ".join(f"loss/{name}={value:.6f}" for name, value in source_losses.items())
                source_x_text = ""
                if source_x_mses:
                    source_x_text = " " + " ".join(f"x_mse/{name}={value:.6f}" for name, value in source_x_mses.items())
                print(
                    f"step={global_step}{offset_text} loss={float(loss):.6f} x_mse={float(x_mse):.6f}"
                    f"{source_text}{source_x_text} lr={current_lr:.8g}{grad_text}"
                )
            run_train_sample = should_run_train_sample(args, global_step)
            run_train_compare = should_run_train_compare(args, global_step)
            if run_train_sample or run_train_compare:
                prepare_memory_for_training_generation(optimizer)
            if run_train_sample:
                sample_from_training_student(
                    student=student,
                    adapter=adapter,
                    args=args,
                    device=device,
                    dtype=dtype,
                    global_step=global_step,
                    tracker_run=tracker_run,
                )
            if run_train_compare:
                maybe_import_compare_baseline(args, tracker_run=tracker_run)
                sample_compare_from_training_student(
                    student=student,
                    adapter=adapter,
                    args=args,
                    device=device,
                    dtype=dtype,
                    global_step=global_step,
                    tracker_run=tracker_run,
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
        if tracker_run is not None:
            tracker_run.finish()

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

    tracker_run = init_tracker(args)
    try:
        if args.tracker_log_sample_images:
            log_sample_images_to_tracker(tracker_run, image_paths, args.prompt)
    finally:
        if tracker_run is not None:
            tracker_run.finish()
    del eps_latent, sigmas, x_latent
    if "student" in locals():
        del student
    if adapter is not None:
        del adapter
    release_cuda_memory()


def sample_compare(args: argparse.Namespace) -> None:
    if args.prediction_type != "x":
        raise ValueError("sample_compare currently compares an x-pred student against an FM teacher; set prediction_type='x'")
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = None if args.toy_smoke else load_object(args.adapter)(args, device=device, dtype=dtype)
    results: dict[str, torch.Tensor] = {}
    if args.toy_smoke:
        student = ToyXPredNet().to(device=device, dtype=dtype)
        state = torch.load(args.student_checkpoint, map_location=device)
        student.load_state_dict(state["state_dict"])
        student.eval()

        def teacher_forward(z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            x_teacher = torch.cat(
                [make_toy_teacher_endpoint(z[i : i + 1], i) for i in range(z.shape[0])],
                dim=0,
            )
            return xpred_to_anima_v(z, x_teacher, sigma, args.eps_floor)

        student_forward = student
        for alpha in args.alphas:
            alpha = float(alpha)
            if alpha == 0.0:
                teacher_latent, _ = adapter.teacher_sample_latent(args.prompt, eps_latent, sigmas, guidance_scale=args.teacher_cfg)
                results[f"alpha-{alpha:g}"] = teacher_latent
                continue
            results[f"alpha-{alpha:g}"] = sample_with_mixed_velocity(
                teacher_forward,
                student_forward,
                eps_latent,
                sigmas,
                alpha=alpha,
                eps_floor=args.eps_floor,
            )
    else:
        if not args.prompt:
            raise ValueError("sample_compare requires prompt for real Anima runs")
        teacher = load_compare_teacher(
            adapter,
            args,
            checkpoint=args.teacher_checkpoint,
            lora=getattr(args, "teacher_lora", None),
            lora_weight=getattr(args, "teacher_lora_weight", None),
        )
        student = adapter.load_student_xpred(init_checkpoint=args.student_checkpoint)
        student.to(device=device, dtype=dtype).eval()
        teacher.to(device=device, dtype=dtype).eval()
        args.prompt = args.prompt
        cond_embed, uncond_embed = adapter._encode_prompts([args.prompt], cfg=args.teacher_cfg, anima_model=teacher)
        teacher_conditioning = {
            "prompt_embeds": cond_embed.detach().to(dtype=dtype),
            "negative_prompt_embeds": uncond_embed.detach().to(dtype=dtype),
        }
        student_conditioning = {"prompt_embeds": cond_embed.detach().to(dtype=dtype)}
        teacher_forward = lambda z, sigma: adapter.teacher_forward_vpred(
            teacher,
            z,
            sigma,
            teacher_conditioning,
            guidance_scale=args.teacher_cfg,
        )
        student_forward = lambda z, sigma: adapter.student_forward_xpred(student, z, sigma, student_conditioning)
        for alpha in args.alphas:
            results[f"alpha-{alpha:g}"] = sample_with_mixed_velocity(
                teacher_forward,
                student_forward,
                eps_latent,
                sigmas,
                alpha=float(alpha),
                eps_floor=args.eps_floor,
            )

    torch.save(
        {"latents": {key: value.detach().cpu() for key, value in results.items()}, "sigmas": sigmas.detach().cpu(), "alphas": list(args.alphas)},
        output_dir / "compare-latents.pt",
    )
    print(f"saved compare latent tensor to {output_dir / 'compare-latents.pt'}")

    image_paths: list[Path] = []
    if args.decode_sample_images:
        if args.toy_smoke:
            print("decode_sample_images=true ignored for toy_smoke sample_compare")
        else:
            for key, latent in results.items():
                image_paths.extend(adapter.decode_latents_to_images(latent, output_dir / "images" / key, prefix=f"{args.sample_image_prefix}-{key}"))
            print(f"saved {len(image_paths)} compare image(s) to {output_dir / 'images'}")
            tracker_run = init_tracker(args)
            if args.tracker_log_sample_images:
                log_sample_images_to_tracker(tracker_run, image_paths, args.prompt, step=None, key="sample_compare/images")
            if tracker_run is not None:
                tracker_run.finish()
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


def add_tracker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracker_enabled", action="store_true")
    parser.add_argument("--tracker_project", default="rum-anima-xpred")
    parser.add_argument("--tracker_entity", default=None)
    parser.add_argument("--tracker_run_name", default=None)
    parser.add_argument("--tracker_run_id", default=None)
    parser.add_argument("--tracker_resume", default=None)
    parser.add_argument("--tracker_mode", default=None)
    parser.add_argument("--tracker_tags", nargs="*", default=[])
    parser.add_argument("--tracker_notes", default=None)
    parser.add_argument("--tracker_log_config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tracker_metrics_file", default=None)
    parser.add_argument("--tracker_metrics_log_every", type=int, default=1)


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
    train.add_argument("--cache_dirs", nargs="*", default=None)
    train.add_argument("--cache_mix_mode", default="single", choices=["single", "batch_weighted"])
    train.add_argument("--cache_mix_weights", nargs="*", type=float, default=None)
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
    train.add_argument("--time_sampling", default="uniform_shifted", choices=["uniform_shifted", "jlt_logit_normal"])
    train.add_argument("--time_sampling_logit_mean", type=float, default=-0.8)
    train.add_argument("--time_sampling_logit_std", type=float, default=0.8)
    train.add_argument("--loss_weighting", default="none", choices=["none", "jlt_velocity_readout"])
    train.add_argument("--loss_eps_floor", type=float, default=5e-2)
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
    train.add_argument("--sample_cfg", type=float, default=1.0)
    train.add_argument("--sample_eps_floor", type=float, default=DEFAULT_EPS_FLOOR)
    train.add_argument("--sample_width", type=int, default=None)
    train.add_argument("--sample_height", type=int, default=None)
    train.add_argument("--sample_seed", type=int, default=None)
    train.add_argument("--sample_output_dir", default=None)
    train.add_argument("--sample_decode_images", action="store_true")
    train.add_argument("--sample_image_prefix", default="train-sample")
    train.add_argument("--sample_tracker_log_images", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--sample_lora", default="__inherit__")
    train.add_argument("--sample_lora_weight", type=float, default=None)
    train.add_argument("--sample_lora_steps", type=int, default=None)
    train.add_argument("--sample_lora_cfg", type=float, default=None)
    train.add_argument("--sample_lora_eps_floor", type=float, default=None)
    train.add_argument("--sample_compare_every_steps", type=int, default=0)
    train.add_argument("--sample_compare_prompt", default="")
    train.add_argument("--sample_compare_steps", type=int, default=DEFAULT_TEACHER_STEPS)
    train.add_argument("--sample_compare_num_samples", type=int, default=1)
    train.add_argument("--sample_compare_cfg", type=float, default=1.0)
    train.add_argument("--sample_compare_eps_floor", type=float, default=DEFAULT_EPS_FLOOR)
    train.add_argument("--sample_compare_width", type=int, default=None)
    train.add_argument("--sample_compare_height", type=int, default=None)
    train.add_argument("--sample_compare_seed", type=int, default=None)
    train.add_argument("--sample_compare_output_dir", default=None)
    train.add_argument("--sample_compare_baseline_source_dir", default=None)
    train.add_argument("--sample_compare_baseline_output_dir", default=None)
    train.add_argument("--sample_compare_baseline_tracker_log_images", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--sample_compare_lora", default="__inherit__")
    train.add_argument("--sample_compare_lora_weight", type=float, default=None)
    train.add_argument("--sample_compare_lora_cfg", type=float, default=None)
    train.add_argument("--sample_compare_teacher_sanity", action="store_true")
    train.add_argument("--sample_compare_teacher_sanity_lora", default="__inherit__")
    train.add_argument("--sample_compare_teacher_sanity_lora_weight", type=float, default=None)
    train.add_argument("--sample_compare_decode_images", action="store_true")
    train.add_argument("--sample_compare_image_prefix", default="compare")
    train.add_argument("--sample_compare_tracker_log_images", action=argparse.BooleanOptionalAction, default=True)
    add_tracker_args(train)
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
    sample.add_argument("--tracker_log_sample_images", action=argparse.BooleanOptionalAction, default=True)
    add_tracker_args(sample)
    sample.set_defaults(func=sample_xpred)

    compare = subparsers.add_parser("sample_compare", help="Compare teacher FM, mixed velocity, and student x-pred sampling.")
    add_common_args(compare)
    compare.add_argument("--student_checkpoint")
    compare.add_argument("--teacher_checkpoint", default=None)
    compare.add_argument("--teacher_lora", default="__inherit__")
    compare.add_argument("--teacher_lora_weight", type=float, default=None)
    compare.add_argument("--output_dir")
    compare.add_argument("--prediction_type", default="x", choices=["x"])
    compare.add_argument("--prompt", default="")
    compare.add_argument("--num_samples", type=int, default=1)
    compare.add_argument("--steps", type=int, default=DEFAULT_TEACHER_STEPS)
    compare.add_argument("--width", type=int, default=1024)
    compare.add_argument("--height", type=int, default=1024)
    compare.add_argument("--eps_floor", type=float, default=DEFAULT_EPS_FLOOR)
    compare.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    compare.add_argument("--teacher_cfg", type=float, default=DEFAULT_TEACHER_CFG)
    compare.add_argument("--decode_sample_images", action="store_true")
    compare.add_argument("--sample_image_prefix", default="compare")
    compare.add_argument("--tracker_log_sample_images", action=argparse.BooleanOptionalAction, default=True)
    add_tracker_args(compare)
    compare.set_defaults(func=sample_compare)

    chunked_stage = subparsers.add_parser("chunked_rum", help="Build cache chunks and train after each chunk with resume manifest.")
    add_common_args(chunked_stage)
    add_tracker_args(chunked_stage)
    chunked_stage.add_argument("--chunk_root", default=None, help="Root directory for rolling cache chunks, train outputs, and manifest.")
    chunked_stage.add_argument("--total_samples", type=int, default=None, help="Total samples to process across all chunks.")
    chunked_stage.add_argument("--chunk_size", type=int, default=1024, help="Samples per cache/train chunk.")
    chunked_stage.add_argument("--start_index", type=int, default=0, help="Global prompt start index for the first chunk.")
    chunked_stage.add_argument("--max_chunks", type=int, default=None, help="Optional cap for debugging/resume.")
    chunked_stage.add_argument("--train_steps_per_chunk", type=int, default=None, help="Override [train_xpred].max_train_steps per chunk.")
    chunked_stage.add_argument("--delete_cache_after_train", action="store_true", help="Delete each chunk cache after its training finishes.")
    chunked_stage.add_argument("--prepared_cache_only", action="store_true", help="Skip build_cache and require all chunk cache directories to already exist.")
    chunked_stage.add_argument("--optimizer_state", default=None, help="Initial optimizer state for the first chunk.")
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
            "sample_compare": sample_compare,
            "chunked_rum": chunked_rum,
        }[args.command]
    elif not args.command:
        parser.error("a subcommand is required unless --config provides command")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.toy_smoke and (not args.dit and args.command in {"build_cache", "chunked_rum", "sample_compare"}):
        parser.error("real Anima build_cache/chunked_rum/sample_compare requires --dit; use --toy_smoke for local smoke tests")
    if not args.toy_smoke and not args.text_encoder:
        parser.error("real Anima runs require --text_encoder; use --toy_smoke for local smoke tests")
    if args.command == "build_cache" and not getattr(args, "prompt_sets", None) and (not args.prompts or not args.cache_dir):
        parser.error("build_cache requires prompts/cache_dir or prompt_sets")
    if args.command == "train_xpred" and (not (getattr(args, "cache_dirs", None) or args.cache_dir) or not args.output_dir):
        parser.error("train_xpred requires cache_dir or cache_dirs, plus output_dir")
    if args.command == "train_xpred" and getattr(args, "cache_dirs", None):
        if len(args.cache_dirs) < 2:
            parser.error("cache_dirs requires at least two directories")
        if args.cache_mix_mode == "single":
            args.cache_mix_mode = "batch_weighted"
        if args.cache_mix_mode != "batch_weighted":
            parser.error("cache_dirs requires cache_mix_mode='batch_weighted'")
        if args.cache_mix_weights is not None and len(args.cache_mix_weights) != len(args.cache_dirs):
            parser.error("cache_mix_weights length must match cache_dirs length")
        if args.cache_mix_weights is not None and (any(weight < 0 for weight in args.cache_mix_weights) or sum(args.cache_mix_weights) <= 0):
            parser.error("cache_mix_weights must be non-negative and sum to > 0")
    if args.command == "sample_xpred" and (not args.checkpoint or not args.output):
        parser.error("sample_xpred requires checkpoint and output")
    if args.command == "sample_compare" and (not args.student_checkpoint or not args.output_dir):
        parser.error("sample_compare requires student_checkpoint and output_dir")
    if hasattr(args, "prediction_type") and args.prediction_type not in {"x", "v"}:
        parser.error("prediction_type must be 'x' or 'v'")
    if hasattr(args, "loss_weighting") and args.loss_weighting == "jlt_velocity_readout" and args.prediction_type != "x":
        parser.error("loss_weighting='jlt_velocity_readout' requires prediction_type='x'")
    if hasattr(args, "loss_eps_floor") and args.loss_eps_floor <= 0:
        parser.error("loss_eps_floor must be > 0")
    if hasattr(args, "time_sampling_logit_std") and args.time_sampling_logit_std <= 0:
        parser.error("time_sampling_logit_std must be > 0")
    if hasattr(args, "global_step_offset") and args.global_step_offset < 0:
        parser.error("global_step_offset must be >= 0")
    if args.command == "chunked_rum":
        if not args.config:
            parser.error("chunked_rum requires --config")
        if not args.chunk_root:
            parser.error("chunked_rum requires chunk_root")
        if args.total_samples is None and not getattr(make_stage_args(args.config, "build_cache"), "prompt_sets", None):
            parser.error("chunked_rum requires total_samples unless build_cache.prompt_sets is enabled")
        if args.total_samples is not None and args.total_samples < 1:
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
    if hasattr(args, "tracker_metrics_log_every") and args.tracker_metrics_log_every < 0:
        parser.error("tracker_metrics_log_every must be >= 0")
    if hasattr(args, "sample_every_steps") and args.sample_every_steps < 0:
        parser.error("sample_every_steps must be >= 0")
    if hasattr(args, "sample_steps") and args.sample_steps < 1:
        parser.error("sample_steps must be >= 1")
    if hasattr(args, "sample_num_samples") and args.sample_num_samples < 1:
        parser.error("sample_num_samples must be >= 1")
    if hasattr(args, "sample_cfg") and args.sample_cfg < 0:
        parser.error("sample_cfg must be >= 0")
    if hasattr(args, "sample_eps_floor") and args.sample_eps_floor <= 0:
        parser.error("sample_eps_floor must be > 0")
    if getattr(args, "sample_lora_steps", None) is not None and args.sample_lora_steps < 1:
        parser.error("sample_lora_steps must be >= 1")
    if getattr(args, "sample_lora_cfg", None) is not None and args.sample_lora_cfg < 0:
        parser.error("sample_lora_cfg must be >= 0")
    if getattr(args, "sample_lora_eps_floor", None) is not None and args.sample_lora_eps_floor <= 0:
        parser.error("sample_lora_eps_floor must be > 0")
    if hasattr(args, "eps_floor") and args.eps_floor <= 0:
        parser.error("eps_floor must be > 0")
    if hasattr(args, "alphas") and any(alpha < 0 or alpha > 1 for alpha in args.alphas):
        parser.error("alphas must be in [0, 1]")
    if getattr(args, "sample_width", None) is not None and args.sample_width < 8:
        parser.error("sample_width must be >= 8")
    if getattr(args, "sample_height", None) is not None and args.sample_height < 8:
        parser.error("sample_height must be >= 8")
    if hasattr(args, "sample_compare_every_steps") and args.sample_compare_every_steps < 0:
        parser.error("sample_compare_every_steps must be >= 0")
    if hasattr(args, "sample_compare_steps") and args.sample_compare_steps < 1:
        parser.error("sample_compare_steps must be >= 1")
    if hasattr(args, "sample_compare_num_samples") and args.sample_compare_num_samples < 1:
        parser.error("sample_compare_num_samples must be >= 1")
    if hasattr(args, "sample_compare_cfg") and args.sample_compare_cfg < 0:
        parser.error("sample_compare_cfg must be >= 0")
    if hasattr(args, "sample_compare_eps_floor") and args.sample_compare_eps_floor <= 0:
        parser.error("sample_compare_eps_floor must be > 0")
    if getattr(args, "sample_compare_lora_cfg", None) is not None and args.sample_compare_lora_cfg < 0:
        parser.error("sample_compare_lora_cfg must be >= 0")
    if getattr(args, "sample_compare_width", None) is not None and args.sample_compare_width < 8:
        parser.error("sample_compare_width must be >= 8")
    if getattr(args, "sample_compare_height", None) is not None and args.sample_compare_height < 8:
        parser.error("sample_compare_height must be >= 8")
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
    if (
        args.command == "train_xpred"
        and getattr(args, "sample_compare_every_steps", 0) > 0
        and getattr(args, "prediction_type", "x") != "x"
    ):
        parser.error("training sample_compare requires prediction_type='x'")
    if (
        args.command == "train_xpred"
        and getattr(args, "sample_compare_every_steps", 0) > 0
        and getattr(args, "sample_compare_decode_images", False)
        and not args.toy_smoke
        and not args.vae
    ):
        parser.error("sample_compare_decode_images during train_xpred requires --vae / [common].vae")
    if args.command in {"sample_xpred", "sample_compare"} and getattr(args, "decode_sample_images", False) and not args.toy_smoke and not args.vae:
        parser.error("decode_sample_images requires --vae / [common].vae")
    if hasattr(args, "num_samples") and args.num_samples is not None and args.num_samples < 1:
        parser.error("num_samples must be >= 1")
    if hasattr(args, "teacher_lora_weight") and args.teacher_lora_weight is not None and args.teacher_lora_weight < 0:
        parser.error("teacher_lora_weight must be >= 0")
    if hasattr(args, "sample_lora_weight") and args.sample_lora_weight is not None and args.sample_lora_weight < 0:
        parser.error("sample_lora_weight must be >= 0")
    if hasattr(args, "sample_compare_lora_weight") and args.sample_compare_lora_weight is not None and args.sample_compare_lora_weight < 0:
        parser.error("sample_compare_lora_weight must be >= 0")
    if hasattr(args, "sample_compare_teacher_sanity_lora_weight") and args.sample_compare_teacher_sanity_lora_weight is not None and args.sample_compare_teacher_sanity_lora_weight < 0:
        parser.error("sample_compare_teacher_sanity_lora_weight must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
