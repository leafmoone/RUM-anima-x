import argparse

import torch

from anima_rum_xpred import CacheMetadata, load_xpred_cache_sample, save_xpred_cache_sample
from scripts.dev.anima_rum_xpred_train import build_cache, final_optimizer_state_for_train_args, resolve_max_train_steps, train_xpred


def test_resolve_max_train_steps_from_epochs():
    args = argparse.Namespace(
        max_train_steps=None,
        train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1.0,
    )

    assert resolve_max_train_steps(args, cache_sample_count=5) == 3


def test_resolve_max_train_steps_explicit_override_wins():
    args = argparse.Namespace(
        max_train_steps=7,
        train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1.0,
    )

    assert resolve_max_train_steps(args, cache_sample_count=5) == 7


def test_train_xpred_saves_and_loads_optimizer_state(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    metadata = CacheMetadata(
        prompt="toy",
        width=64,
        height=64,
        seed=1,
        sample_index=0,
        teacher_steps=1,
        flow_shift=3.0,
        teacher_cfg=1.0,
    )
    save_xpred_cache_sample(
        cache_dir / "sample-000000.safetensors",
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 16, 8, 8),
        {"prompt_embeds": torch.randn(1, 4, 8)},
        metadata,
    )

    first = argparse.Namespace(
        device="cpu",
        mixed_precision="fp32",
        cache_dir=str(cache_dir),
        output_dir=str(tmp_path / "first"),
        toy_smoke=True,
        adapter="",
        student_init=None,
        optimizer_state=None,
        prediction_type="x",
        max_train_steps=1,
        num_train_epochs=1.0,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_scheduler="constant",
        lr_warmup_steps=0,
        lr_cosine_min=0.1,
        lr_scheduler_total_steps=None,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        sigma_min_train=0.02,
        flow_shift=3.0,
        shuffle_cache=False,
        drop_last=False,
        seed=1,
        log_every=10,
        save_every_steps=None,
        checkpoints_total_limit=None,
        gradient_checkpointing=False,
        gradient_checkpointing_cpu_offload=False,
        gradient_checkpointing_unsloth_offload=False,
        sample_every_steps=0,
        dry_run=False,
        wandb_enabled=False,
        wandb_metrics_log_every=0,
        wandb_metrics_file=None,
        global_step_offset=0,
    )
    train_xpred(first)
    first_state = final_optimizer_state_for_train_args(first)
    assert first_state.exists()

    second = argparse.Namespace(**{**vars(first), "output_dir": str(tmp_path / "second"), "optimizer_state": str(first_state)})
    train_xpred(second)
    second_state = torch.load(final_optimizer_state_for_train_args(second), map_location="cpu")

    assert second_state["completed_train_steps"] == 1
    assert second_state["optimizer"]["state"]


def test_build_cache_bucket_enabled_writes_resolution_subdirs(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join([f"prompt {index}" for index in range(10)]), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    args = argparse.Namespace(
        device="cpu",
        mixed_precision="fp32",
        prompts=str(prompts),
        cache_dir=str(cache_dir),
        num_samples=10,
        start_index=0,
        cache_batch_size=2,
        skip_existing=True,
        bucket_enabled=True,
        width=64,
        height=64,
        teacher_steps=1,
        flow_shift=3.0,
        teacher_cfg=1.0,
        teacher_lora=None,
        teacher_lora_weight=1.0,
        seed=11,
        toy_smoke=True,
        adapter="",
    )

    build_cache(args)

    files = sorted(cache_dir.glob("*/*.safetensors"))
    assert len(files) == 10
    assert len({path.parent.name for path in files}) > 1
    sample = load_xpred_cache_sample(files[0], device="cpu", dtype=torch.float32)
    assert f"{sample['width']}x{sample['height']}" == files[0].parent.name
