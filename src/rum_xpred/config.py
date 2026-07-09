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


COMMANDS = {"build_cache", "train_xpred", "sample_xpred", "sample_compare", "chunked_rum"}
LEGACY_WANDB_SECTION = "wa" + "ndb"
LEGACY_PREPARED_CACHE_ONLY = "ex" + "ternal_cache_only"
PROMPT_SET_KEYS = {"name", "prompts", "cache_dir", "start_index", "num_samples", "cache_chunk_offset", "repeat"}

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
        "prompt_sets": None,
        "num_samples": None,
        "start_index": 0,
        "cache_batch_size": 1,
        "skip_existing": True,
        "bucket_enabled": False,
        "width": 1024,
        "height": 1024,
        "teacher_steps": 40,
        "teacher_cfg": 1.0,
        "teacher_lora": None,
        "teacher_lora_weight": 1.0,
    },
    "train_xpred": {
        "cache_dir": None,
        "cache_dirs": None,
        "cache_mix_mode": "single",
        "cache_mix_weights": None,
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
        "time_sampling": "uniform_shifted",
        "time_sampling_logit_mean": -0.8,
        "time_sampling_logit_std": 0.8,
        "loss_weighting": "none",
        "loss_eps_floor": 5e-2,
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
        "sample_cfg": 1.0,
        "sample_eps_floor": 1e-4,
        "sample_width": None,
        "sample_height": None,
        "sample_seed": None,
        "sample_output_dir": None,
        "sample_decode_images": False,
        "sample_image_prefix": "train-sample",
        "sample_wandb_log_images": True,
        "sample_lora": "__inherit__",
        "sample_lora_weight": None,
        "sample_lora_steps": None,
        "sample_lora_cfg": None,
        "sample_lora_eps_floor": None,
        "sample_compare_every_steps": 0,
        "sample_compare_prompt": "",
        "sample_compare_steps": 40,
        "sample_compare_num_samples": 1,
        "sample_compare_cfg": 1.0,
        "sample_compare_eps_floor": 1e-4,
        "sample_compare_width": None,
        "sample_compare_height": None,
        "sample_compare_seed": None,
        "sample_compare_output_dir": None,
        "sample_compare_baseline_source_dir": None,
        "sample_compare_baseline_output_dir": None,
        "sample_compare_baseline_wandb_log_images": True,
        "sample_compare_lora": "__inherit__",
        "sample_compare_lora_weight": None,
        "sample_compare_lora_cfg": None,
        "sample_compare_teacher_sanity": False,
        "sample_compare_teacher_sanity_lora": "__inherit__",
        "sample_compare_teacher_sanity_lora_weight": None,
        "sample_compare_decode_images": False,
        "sample_compare_image_prefix": "compare",
        "sample_compare_wandb_log_images": True,
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
    "sample_compare": {
        "student_checkpoint": None,
        "teacher_checkpoint": None,
        "teacher_lora": "__inherit__",
        "teacher_lora_weight": None,
        "output_dir": None,
        "prediction_type": "x",
        "prompt": "",
        "num_samples": 1,
        "steps": 40,
        "width": 1024,
        "height": 1024,
        "eps_floor": 1e-4,
        "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "teacher_cfg": 1.0,
        "decode_sample_images": False,
        "sample_image_prefix": "compare",
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
    "wandb_run_id": None,
    "wandb_resume": None,
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
    if command == "build_cache" and isinstance(command_values.get("prompt_sets"), list):
        build_cache_global_keys = set(COMMAND_DEFAULTS["build_cache"]) - {"prompt_sets", *PROMPT_SET_KEYS}
        for index, prompt_set in enumerate(command_values["prompt_sets"]):
            if not isinstance(prompt_set, dict):
                continue
            misplaced = sorted(set(prompt_set) & build_cache_global_keys)
            if misplaced:
                raise ValueError(
                    "build_cache.prompt_sets entries contain global build_cache key(s): "
                    f"prompt_sets[{index}] has {', '.join(misplaced)}. "
                    "Move these keys before the first [[build_cache.prompt_sets]] table."
                )
    build_cache_values = _section(config, "build_cache")
    if command == "train_xpred":
        if command_values.get("sample_compare_lora") == "__inherit__":
            command_values["sample_compare_lora"] = build_cache_values.get("teacher_lora")
        if command_values.get("sample_compare_lora_weight") is None:
            command_values["sample_compare_lora_weight"] = build_cache_values.get("teacher_lora_weight", 1.0)
        if command_values.get("sample_lora") == "__inherit__":
            command_values["sample_lora"] = build_cache_values.get("teacher_lora")
        if command_values.get("sample_lora_weight") is None:
            command_values["sample_lora_weight"] = build_cache_values.get("teacher_lora_weight", 1.0)
        if command_values.get("sample_compare_teacher_sanity_lora") == "__inherit__":
            command_values["sample_compare_teacher_sanity_lora"] = build_cache_values.get("teacher_lora")
        if command_values.get("sample_compare_teacher_sanity_lora_weight") is None:
            command_values["sample_compare_teacher_sanity_lora_weight"] = build_cache_values.get("teacher_lora_weight", 1.0)
    elif command == "sample_compare":
        if command_values.get("teacher_lora") == "__inherit__":
            command_values["teacher_lora"] = build_cache_values.get("teacher_lora")
        if command_values.get("teacher_lora_weight") is None:
            command_values["teacher_lora_weight"] = build_cache_values.get("teacher_lora_weight", 1.0)
    wandb_values = dict(WANDB_DEFAULTS)
    wandb_values.update(_section(config, LEGACY_WANDB_SECTION))
    wandb_values.update(_section(config, "wandb"))
    command_config = _section(config, command)
    if command == "chunked_rum" and LEGACY_PREPARED_CACHE_ONLY in command_config:
        command_values["prepared_cache_only"] = command_config[LEGACY_PREPARED_CACHE_ONLY]

    unknown_sections = sorted(
        key
        for key, value in config.items()
        if isinstance(value, dict) and key not in {"common", LEGACY_WANDB_SECTION, "wandb", *COMMANDS}
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
