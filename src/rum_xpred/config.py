from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import toml as _toml  # type: ignore

    class tomllib:  # type: ignore
        @staticmethod
        def loads(text: str) -> dict[str, Any]:
            return _toml.loads(text)


COMMANDS = {"build_cache", "train_xpred", "sample_xpred", "chunked_rum"}

COMMON_DEFAULTS: dict[str, Any] = {
    "device": None,
    "mixed_precision": "bf16",
    "flow_shift": 3.0,
    "seed": 20260701,
    "toy_smoke": False,
    "adapter": "rum_xpred.adapters.anima_sd_scripts:create_adapter",
    "dit": None,
    "student_init": None,
    "text_encoder": None,
    "vae": None,
    "vae_spatial_chunk_size": None,
    "vae_disable_cache": False,
    "negative_prompt": "",
    "attn_mode": "torch",
    "fp8": False,
    "fp8_scaled": False,
    "text_encoder_cpu": False,
}

COMMAND_DEFAULTS: dict[str, dict[str, Any]] = {
    "build_cache": {
        "prompts": None,
        "cache_dir": None,
        "num_samples": None,
        "start_index": 0,
        "cache_batch_size": 1,
        "skip_existing": True,
        "width": 1024,
        "height": 1024,
        "teacher_steps": 40,
        "teacher_cfg": 1.0,
        "teacher_lora": None,
        "teacher_lora_weight": 1.0,
    },
    "train_xpred": {
        "cache_dir": None,
        "output_dir": None,
        "prediction_type": "x",
        "global_step_offset": 0,
        "optimizer_state": None,
        "max_train_steps": None,
        "num_train_epochs": 1.0,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-6,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "lr_cosine_min": 0.1,
        "lr_scheduler_total_steps": None,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "sigma_min_train": 0.02,
        "shuffle_cache": True,
        "drop_last": False,
        "log_every": 1,
        "save_every_steps": None,
        "checkpoints_total_limit": None,
        "gradient_checkpointing": False,
        "gradient_checkpointing_cpu_offload": False,
        "gradient_checkpointing_unsloth_offload": False,
        "sample_every_steps": 0,
        "sample_prompt": "",
        "sample_steps": 40,
        "sample_num_samples": 1,
        "sample_eps_floor": 1e-4,
        "sample_width": None,
        "sample_height": None,
        "sample_seed": None,
        "sample_output_dir": None,
        "sample_decode_images": False,
        "sample_image_prefix": "train-sample",
        "sample_wandb_log_images": True,
        "dry_run": False,
    },
    "sample_xpred": {
        "checkpoint": None,
        "output": None,
        "prediction_type": "x",
        "prompt": "",
        "num_samples": 1,
        "steps": 40,
        "width": 1024,
        "height": 1024,
        "eps_floor": 1e-4,
        "decode_sample_images": False,
        "sample_image_dir": None,
        "sample_image_prefix": "sample",
        "wandb_log_sample_images": True,
    },
    "chunked_rum": {
        "chunk_root": None,
        "total_samples": None,
        "chunk_size": 1024,
        "start_index": 0,
        "max_chunks": None,
        "train_steps_per_chunk": None,
        "delete_cache_after_train": False,
        "resume": True,
    },
}

WANDB_DEFAULTS: dict[str, Any] = {
    "wandb_enabled": False,
    "wandb_project": "rum-anima-xpred",
    "wandb_entity": None,
    "wandb_run_name": None,
    "wandb_mode": None,
    "wandb_tags": [],
    "wandb_notes": None,
    "wandb_log_config": True,
    "wandb_metrics_file": None,
    "wandb_metrics_log_every": 1,
}


def load_toml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a TOML table: {config_path}")
    return data


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return dict(value)


def config_to_namespace(config: dict[str, Any], *, command_override: str | None = None) -> argparse.Namespace:
    command = command_override or config.get("command")
    if command not in COMMANDS:
        raise ValueError(f"config command must be one of {sorted(COMMANDS)}, got {command!r}")

    common = dict(COMMON_DEFAULTS)
    common.update(_section(config, "common"))
    command_values = dict(COMMAND_DEFAULTS[command])
    command_values.update(_section(config, command))
    wandb_values = dict(WANDB_DEFAULTS)
    wandb_values.update(_section(config, "wandb"))

    unknown_sections = sorted(
        key
        for key, value in config.items()
        if isinstance(value, dict) and key not in {"common", "wandb", *COMMANDS}
    )
    if unknown_sections:
        raise ValueError(f"unknown config section(s): {', '.join(unknown_sections)}")

    unknown_keys = sorted(
        key
        for key, value in config.items()
        if not isinstance(value, dict) and key != "command"
    )
    if unknown_keys:
        raise ValueError(f"unknown config key(s): {', '.join(unknown_keys)}")

    args = argparse.Namespace(command=command, **common, **command_values, **wandb_values)
    return args


def apply_overrides(args: argparse.Namespace, overrides: dict[str, Any]) -> argparse.Namespace:
    for key, value in overrides.items():
        if value is not None:
            setattr(args, key, value)
    return args
