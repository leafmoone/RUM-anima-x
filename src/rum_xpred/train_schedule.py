from __future__ import annotations

import argparse
import math

import torch

from rum_xpred.chunking import ChunkPlan, prompt_set_effective_total


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


def planned_chunk_train_steps(
    train_args: argparse.Namespace,
    chunk_args: argparse.Namespace,
    plans: list[ChunkPlan],
    *,
    cache_multiplier: int = 1,
) -> int:
    if chunk_args.train_steps_per_chunk is not None:
        return chunk_args.train_steps_per_chunk * len(plans)
    if train_args.max_train_steps is not None:
        return train_args.max_train_steps * len(plans)
    return sum(resolve_max_train_steps(train_args, plan.num_samples * cache_multiplier) for plan in plans)


def planned_prompt_set_train_steps(train_args: argparse.Namespace, chunk_args: argparse.Namespace, plans: list[ChunkPlan], prompt_sets: list[dict]) -> int:
    if chunk_args.train_steps_per_chunk is not None:
        return chunk_args.train_steps_per_chunk * len(plans)
    if train_args.max_train_steps is not None:
        return train_args.max_train_steps * len(plans)
    total = 0
    for plan in plans:
        cache_sample_count = 0
        for prompt_set in prompt_sets:
            remaining = prompt_set_effective_total(prompt_set) - plan.start_index
            if remaining > 0:
                cache_sample_count += min(plan.num_samples, remaining)
        if cache_sample_count > 0:
            total += resolve_max_train_steps(train_args, cache_sample_count)
    return total
