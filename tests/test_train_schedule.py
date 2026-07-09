import argparse
import sys
import types

import torch

from anima_rum_xpred import CacheMetadata, load_xpred_cache_sample, save_xpred_cache_sample
import scripts.dev.anima_rum_xpred_train as train_mod
from scripts.dev.anima_rum_xpred_train import (
    build_cache,
    final_optimizer_state_for_train_args,
    log_sample_images_to_wandb,
    prepare_memory_for_training_generation,
    reflow_loss,
    resolve_max_train_steps,
    sample_compare_from_training_student,
    sample_from_training_student,
    train_xpred,
)


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


def test_prepare_memory_for_training_generation_clears_gradients():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    parameter.grad = torch.ones_like(parameter)

    prepare_memory_for_training_generation(optimizer)

    assert parameter.grad is None


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
        time_sampling="uniform_shifted",
        time_sampling_logit_mean=-0.8,
        time_sampling_logit_std=0.8,
        loss_weighting="none",
        loss_eps_floor=5e-2,
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


def test_jlt_velocity_readout_loss_matches_sigma_weighted_x_error():
    x_target = torch.tensor([[[[2.0]]]])
    x_pred = torch.tensor([[[[2.25]]]])
    eps = torch.tensor([[[[5.0]]]])
    sigma = torch.tensor([[[[0.5]]]])
    z = (1 - sigma) * x_target + sigma * eps

    loss = reflow_loss("x", "jlt_velocity_readout", x_pred, x_target, z, sigma, eps_floor=5e-2)

    expected = ((x_pred - x_target) / sigma).pow(2).mean()
    torch.testing.assert_close(loss, expected)


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


def test_build_cache_prompt_sets_write_to_separate_dirs(tmp_path):
    tag_prompts = tmp_path / "tag.txt"
    nl_prompts = tmp_path / "nl.txt"
    tag_prompts.write_text("tag a\ntag b\n", encoding="utf-8")
    nl_prompts.write_text("nl a\nnl b\n", encoding="utf-8")
    tag_cache = tmp_path / "tag-cache"
    nl_cache = tmp_path / "nl-cache"
    args = argparse.Namespace(
        device="cpu",
        mixed_precision="fp32",
        prompts=None,
        cache_dir=None,
        prompt_sets=[
            {"name": "tag", "prompts": str(tag_prompts), "cache_dir": str(tag_cache), "start_index": 0, "num_samples": 2},
            {"name": "nl", "prompts": str(nl_prompts), "cache_dir": str(nl_cache), "start_index": 0, "num_samples": 2},
        ],
        num_samples=None,
        start_index=0,
        cache_batch_size=2,
        skip_existing=True,
        bucket_enabled=False,
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

    assert len(list(tag_cache.glob("*.safetensors"))) == 2
    assert len(list(nl_cache.glob("*.safetensors"))) == 2
    assert (tag_cache / "sample-000000.safetensors").exists()
    assert (nl_cache / "sample-000000.safetensors").exists()


def test_train_xpred_accepts_multiple_cache_dirs(tmp_path):
    cache_a = tmp_path / "cache-a"
    cache_b = tmp_path / "cache-b"
    cache_a.mkdir()
    cache_b.mkdir()
    for cache_dir, offset in [(cache_a, 0), (cache_b, 100)]:
        for i in range(2):
            sample_index = offset + i
            metadata = CacheMetadata(
                prompt=f"toy {sample_index}",
                width=64,
                height=64,
                seed=1,
                sample_index=sample_index,
                teacher_steps=1,
                flow_shift=3.0,
                teacher_cfg=1.0,
            )
            save_xpred_cache_sample(
                cache_dir / f"sample-{sample_index:06d}.safetensors",
                torch.randn(1, 16, 8, 8),
                torch.randn(1, 16, 8, 8),
                {"prompt_embeds": torch.randn(1, 4, 8)},
                metadata,
            )
    args = argparse.Namespace(
        device="cpu",
        mixed_precision="fp32",
        cache_dir=None,
        cache_dirs=[str(cache_a), str(cache_b)],
        cache_mix_mode="single",
        cache_mix_weights=[0.5, 0.5],
        output_dir=str(tmp_path / "train"),
        toy_smoke=True,
        adapter="",
        student_init=None,
        optimizer_state=None,
        prediction_type="x",
        max_train_steps=None,
        num_train_epochs=1.0,
        train_batch_size=2,
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
        time_sampling="uniform_shifted",
        time_sampling_logit_mean=-0.8,
        time_sampling_logit_std=0.8,
        loss_weighting="none",
        loss_eps_floor=5e-2,
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

    train_xpred(args)

    assert args.resolved_max_train_steps == 2
    assert args.cache_mix_mode == "batch_weighted"
    assert final_optimizer_state_for_train_args(args).exists()


def test_training_sample_uses_total_step_and_logs_images_to_wandb(tmp_path, monkeypatch):
    class ConstantXStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()))

        def forward(self, z, sigma):
            return torch.zeros_like(z)

    class FakeAdapter:
        def _encode_prompts(self, prompts, *, cfg, anima_model):
            return torch.zeros(1, 4, 8), torch.ones(1, 4, 8)

        def student_forward_xpred(self, student, z, sigma, text_conditioning, *, guidance_scale=1.0):
            return student(z, sigma)

        def decode_latents_to_images(self, latents, image_dir, prefix):
            image_dir.mkdir(parents=True, exist_ok=True)
            path = image_dir / f"{prefix}-0000.png"
            path.write_bytes(b"fake image")
            return [path]

    calls = []

    def fake_log_sample_images_to_wandb(wandb_run, image_paths, prompt, step=None, *, key="sample/images"):
        calls.append(
            {
                "wandb_run": wandb_run,
                "image_paths": list(image_paths),
                "prompt": prompt,
                "step": step,
                "key": key,
            }
        )

    monkeypatch.setattr(train_mod, "log_sample_images_to_wandb", fake_log_sample_images_to_wandb)

    args = argparse.Namespace(
        toy_smoke=False,
        prediction_type="x",
        sample_prompt="preview prompt",
        prompt=None,
        sample_width=64,
        sample_height=64,
        width=64,
        height=64,
        sample_seed=123,
        seed=1,
        sample_steps=2,
        flow_shift=3.0,
        sample_num_samples=1,
        sample_eps_floor=1e-4,
        sample_output_dir=str(tmp_path / "samples"),
        output_dir=str(tmp_path / "train"),
        sample_decode_images=True,
        sample_wandb_log_images=True,
        sample_image_prefix="preview",
        global_step_offset=1999,
    )

    image_paths = sample_from_training_student(
        student=ConstantXStudent(),
        adapter=FakeAdapter(),
        args=args,
        device=torch.device("cpu"),
        dtype=torch.float32,
        global_step=1,
        wandb_run="run",
    )

    assert (tmp_path / "samples" / "latents" / "xpred-step-002000.pt").exists()
    assert image_paths == [tmp_path / "samples" / "images" / "step-002000" / "preview-0000.png"]
    assert calls == [
        {
            "wandb_run": "run",
            "image_paths": image_paths,
            "prompt": "preview prompt",
            "step": 2000,
            "key": "sample/images",
        }
    ]


def test_training_sample_lora_can_use_separate_step_count(tmp_path):
    class ConstantXStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()))

        def forward(self, z, sigma):
            return torch.zeros_like(z)

    class FakeAdapter:
        def _encode_prompts(self, prompts, *, cfg, anima_model):
            return torch.zeros(1, 4, 8), torch.ones(1, 4, 8)

        def student_forward_xpred(self, student, z, sigma, text_conditioning, *, guidance_scale=1.0):
            return student(z, sigma)

        def save_student_xpred(self, student, checkpoint_path):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"checkpoint")

        def load_student_xpred(self, init_checkpoint):
            return ConstantXStudent()

    args = argparse.Namespace(
        toy_smoke=False,
        prediction_type="x",
        sample_prompt="preview prompt",
        prompt=None,
        sample_width=64,
        sample_height=64,
        width=64,
        height=64,
        sample_seed=123,
        seed=1,
        sample_steps=2,
        flow_shift=3.0,
        sample_num_samples=1,
        sample_eps_floor=1e-4,
        sample_output_dir=str(tmp_path / "samples"),
        output_dir=str(tmp_path / "train"),
        sample_decode_images=False,
        sample_wandb_log_images=True,
        sample_image_prefix="preview",
        sample_lora="/models/turbo.safetensors",
        sample_lora_weight=1.0,
        sample_lora_steps=10,
        sample_lora_eps_floor=1e-4,
        teacher_lora=None,
        teacher_lora_weight=1.0,
        mixed_precision="fp32",
        device="cpu",
        global_step_offset=1999,
    )

    sample_from_training_student(
        student=ConstantXStudent(),
        adapter=FakeAdapter(),
        args=args,
        device=torch.device("cpu"),
        dtype=torch.float32,
        global_step=1,
        wandb_run=None,
    )

    main_latent = torch.load(tmp_path / "samples" / "latents" / "xpred-step-002000.pt", map_location="cpu")
    lora_latent = torch.load(tmp_path / "samples" / "latents" / "xpred-lora-step-002000.pt", map_location="cpu")

    assert main_latent["sigmas"].numel() == 3
    assert lora_latent["sigmas"].numel() == 11


def test_training_sample_compare_uses_total_step_and_logs_images_to_wandb(tmp_path, monkeypatch):
    class ConstantXStudent(torch.nn.Module):
        def forward(self, z, sigma):
            return torch.zeros_like(z)

    class FakeAdapter:
        def __init__(self):
            self.teacher = None
            self.encode_calls = 0
            self.encode_cfgs = []
            self.student_conditioning_keys = []
            self.saved_checkpoint = None
            self.loaded_checkpoint = None
            self.loaded_lora = None
            self.loaded_lora_weight = None

        def _load_teacher(self):
            raise AssertionError("training student migration compare must not load teacher")

        def _encode_prompts(self, prompts, *, cfg, anima_model):
            self.encode_calls += 1
            self.encode_cfgs.append(cfg)
            return torch.zeros(1, 4, 8), torch.ones(1, 4, 8)

        def student_forward_xpred(self, student, z, sigma, text_conditioning, *, guidance_scale=1.0):
            self.student_conditioning_keys.append(set(text_conditioning))
            return student(z, sigma)

        def save_student_xpred(self, student, checkpoint_path):
            self.saved_checkpoint = checkpoint_path
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"checkpoint")

        def load_student_xpred(self, init_checkpoint):
            self.loaded_checkpoint = init_checkpoint
            self.loaded_lora = getattr(args, "teacher_lora", None)
            self.loaded_lora_weight = getattr(args, "teacher_lora_weight", None)
            return ConstantXStudent()

        def decode_latents_to_images(self, latents, image_dir, prefix):
            image_dir.mkdir(parents=True, exist_ok=True)
            path = image_dir / f"{prefix}-0000.png"
            path.write_bytes(b"fake image")
            return [path]

    calls = []

    def fake_log_sample_images_to_wandb(wandb_run, image_paths, prompt, step=None, *, key="sample/images"):
        calls.append(
            {
                "wandb_run": wandb_run,
                "image_paths": list(image_paths),
                "prompt": prompt,
                "step": step,
                "key": key,
            }
        )

    monkeypatch.setattr(train_mod, "log_sample_images_to_wandb", fake_log_sample_images_to_wandb)

    args = argparse.Namespace(
        toy_smoke=False,
        prediction_type="x",
        prompt=None,
        dit="/teacher/base.safetensors",
        teacher_lora=None,
        teacher_lora_weight=1.0,
        sample_prompt="fallback prompt",
        sample_width=None,
        sample_height=None,
        width=64,
        height=64,
        sample_seed=None,
        seed=1,
        output_dir=str(tmp_path / "train"),
        global_step_offset=1999,
        flow_shift=3.0,
        sample_compare_prompt="compare prompt",
        sample_compare_steps=2,
            sample_compare_num_samples=2,
            sample_compare_cfg=1.5,
            sample_compare_eps_floor=1e-4,
        sample_compare_width=64,
        sample_compare_height=64,
        sample_compare_seed=123,
        sample_compare_output_dir=str(tmp_path / "compare"),
        sample_compare_decode_images=True,
        sample_compare_image_prefix="compare",
        sample_compare_wandb_log_images=True,
            sample_compare_lora="/teacher/turbo.safetensors",
            sample_compare_lora_weight=0.75,
            sample_compare_lora_cfg=2.5,
        )
    adapter = FakeAdapter()

    image_paths = sample_compare_from_training_student(
        student=ConstantXStudent(),
        adapter=adapter,
        args=args,
        device=torch.device("cpu"),
        dtype=torch.float32,
        global_step=1,
        wandb_run="run",
    )

    assert (tmp_path / "compare" / "step-002000" / "compare-latents.pt").exists()
    assert image_paths == [
        tmp_path / "compare" / "step-002000" / "images" / "fm" / "compare-fm-0000.png",
        tmp_path / "compare" / "step-002000" / "images" / "x" / "compare-x-0000.png",
        tmp_path / "compare" / "step-002000" / "images" / "fm_lora" / "compare-fm_lora-0000.png",
        tmp_path / "compare" / "step-002000" / "images" / "x_lora" / "compare-x_lora-0000.png",
    ]
    assert calls == [
        {
            "wandb_run": "run",
            "image_paths": image_paths,
            "prompt": "compare prompt",
            "step": 2000,
            "key": "sample_compare/images",
        }
    ]
    assert adapter.teacher is None
    assert adapter.encode_calls == 2
    assert adapter.encode_cfgs == [1.5, 2.5]
    assert adapter.saved_checkpoint is not None
    assert adapter.loaded_checkpoint == str(adapter.saved_checkpoint)
    assert adapter.loaded_lora == "/teacher/turbo.safetensors"
    assert adapter.loaded_lora_weight == 0.75
    assert not adapter.saved_checkpoint.exists()
    assert adapter.student_conditioning_keys
    assert all(keys == {"prompt_embeds", "negative_prompt_embeds"} for keys in adapter.student_conditioning_keys)


def test_maybe_import_compare_baseline_copies_and_logs_images(tmp_path, monkeypatch):
    source = tmp_path / "old" / "alpha-0"
    source.mkdir(parents=True)
    (source / "baseline-0000.png").write_bytes(b"image")
    (source / ".ipynb_checkpoints").mkdir()
    (source / ".ipynb_checkpoints" / "ignored.png").write_bytes(b"ignored")
    calls = []

    def fake_log_sample_images_to_wandb(wandb_run, image_paths, prompt, step=None, *, key="sample/images"):
        calls.append(
            {
                "wandb_run": wandb_run,
                "image_paths": list(image_paths),
                "prompt": prompt,
                "step": step,
                "key": key,
            }
        )

    monkeypatch.setattr(train_mod, "log_sample_images_to_wandb", fake_log_sample_images_to_wandb)
    args = argparse.Namespace(
        output_dir=str(tmp_path / "train"),
        sample_prompt="prompt",
        sample_compare_prompt="",
        sample_compare_baseline_source_dir=str(source),
        sample_compare_baseline_output_dir=str(tmp_path / "train" / "compare-baseline"),
        sample_compare_baseline_wandb_log_images=True,
    )

    copied = train_mod.maybe_import_compare_baseline(args, wandb_run="run")
    copied_again = train_mod.maybe_import_compare_baseline(args, wandb_run="run")

    assert copied == [tmp_path / "train" / "compare-baseline" / "teacher-baseline-0000.png"]
    assert copied[0].exists()
    assert copied_again == []
    assert calls == [
        {
            "wandb_run": "run",
            "image_paths": copied,
            "prompt": "prompt",
            "step": None,
            "key": "sample_compare/baseline",
        }
    ]


def test_log_sample_images_to_wandb_omits_stale_explicit_step(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake image")

    class FakeRun:
        step = 2201

        def __init__(self):
            self.calls = []

        def log(self, payload, step=None):
            self.calls.append((payload, step))

    fake_wandb = types.SimpleNamespace(Image=lambda path, caption=None: {"path": path, "caption": caption})
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    run = FakeRun()

    log_sample_images_to_wandb(run, [image_path], "prompt", step=2170, key="sample_compare/images")

    assert run.calls[0][1] is None
    assert run.calls[0][0]["sample_compare/step"] == 2170
