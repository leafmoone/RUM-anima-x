import pytest

from rum_xpred.config import config_to_namespace, load_toml_config


def test_config_to_namespace_merges_common_and_command_sections():
    args = config_to_namespace(
        {
            "command": "build_cache",
            "common": {
                "toy_smoke": True,
                "mixed_precision": "fp32",
                "flow_shift": 2.0,
            },
            "build_cache": {
                "prompts": "data/prompts/sample_prompts.txt",
                "cache_dir": "/tmp/cache",
                "width": 64,
                "height": 64,
                "teacher_lora": "/models/turbo.safetensors",
                "teacher_lora_weight": 0.75,
            },
        }
    )

    assert args.command == "build_cache"
    assert args.toy_smoke is True
    assert args.mixed_precision == "fp32"
    assert args.flow_shift == 2.0
    assert args.prompts == "data/prompts/sample_prompts.txt"
    assert args.cache_dir == "/tmp/cache"
    assert args.teacher_steps == 40
    assert args.teacher_lora == "/models/turbo.safetensors"
    assert args.teacher_lora_weight == 0.75


def test_config_command_can_be_overridden_by_cli_subcommand():
    args = config_to_namespace(
        {
            "command": "build_cache",
            "common": {"toy_smoke": True},
            "train_xpred": {
                "cache_dir": "/tmp/cache",
                "output_dir": "/tmp/train",
            },
        },
        command_override="train_xpred",
    )

    assert args.command == "train_xpred"
    assert args.cache_dir == "/tmp/cache"
    assert args.output_dir == "/tmp/train"


def test_config_merges_wandb_and_gradient_checkpointing_settings():
    args = config_to_namespace(
        {
            "command": "train_xpred",
            "common": {"toy_smoke": True},
            "wandb": {
                "wandb_enabled": True,
                "wandb_project": "test-project",
                "wandb_mode": "offline",
                "wandb_tags": ["smoke", "xpred"],
                "wandb_metrics_file": "/tmp/eval.json",
                "wandb_metrics_log_every": 5,
            },
            "train_xpred": {
                "cache_dir": "/tmp/cache",
                "output_dir": "/tmp/train",
                "gradient_checkpointing": True,
                "gradient_checkpointing_cpu_offload": True,
                "sample_every_steps": 100,
                "sample_steps": 8,
                "sample_num_samples": 2,
                "sample_eps_floor": 1e-5,
                "sample_prompt": "preview prompt",
                "sample_decode_images": True,
                "sample_wandb_log_images": True,
                "num_train_epochs": 2.0,
                "lr_scheduler": "cosine",
                "lr_warmup_steps": 10,
                "lr_cosine_min": 0.2,
                "dry_run": True,
            },
        }
    )

    assert args.wandb_enabled is True
    assert args.wandb_project == "test-project"
    assert args.wandb_mode == "offline"
    assert args.wandb_tags == ["smoke", "xpred"]
    assert args.wandb_metrics_file == "/tmp/eval.json"
    assert args.wandb_metrics_log_every == 5
    assert args.gradient_checkpointing is True
    assert args.gradient_checkpointing_cpu_offload is True
    assert args.gradient_checkpointing_unsloth_offload is False
    assert args.lr_scheduler == "cosine"
    assert args.num_train_epochs == 2.0
    assert args.lr_warmup_steps == 10
    assert args.lr_cosine_min == 0.2
    assert args.dry_run is True
    assert args.prediction_type == "x"
    assert args.sample_every_steps == 100
    assert args.sample_steps == 8
    assert args.sample_num_samples == 2
    assert args.sample_eps_floor == 1e-5
    assert args.sample_prompt == "preview prompt"
    assert args.sample_decode_images is True
    assert args.sample_wandb_log_images is True


def test_config_loads_v_prediction_type_for_train_and_sample():
    train_args = config_to_namespace(
        {
            "command": "train_xpred",
            "common": {"toy_smoke": True},
            "train_xpred": {
                "cache_dir": "/tmp/cache",
                "output_dir": "/tmp/train",
                "prediction_type": "v",
            },
        }
    )
    sample_args = config_to_namespace(
        {
            "command": "sample_xpred",
            "common": {"toy_smoke": True},
            "sample_xpred": {
                "checkpoint": "/tmp/model.pt",
                "output": "/tmp/out.pt",
                "prediction_type": "v",
            },
        }
    )

    assert train_args.prediction_type == "v"
    assert sample_args.prediction_type == "v"


def test_config_loads_chunked_rum_settings():
    args = config_to_namespace(
        {
            "command": "chunked_rum",
            "common": {"toy_smoke": True},
            "chunked_rum": {
                "chunk_root": "/tmp/chunks",
                "total_samples": 30,
                "chunk_size": 8,
                "start_index": 4,
                "max_chunks": 2,
                "train_steps_per_chunk": 5,
                "delete_cache_after_train": True,
                "resume": False,
            },
        }
    )

    assert args.command == "chunked_rum"
    assert args.chunk_root == "/tmp/chunks"
    assert args.total_samples == 30
    assert args.chunk_size == 8
    assert args.start_index == 4
    assert args.max_chunks == 2
    assert args.train_steps_per_chunk == 5
    assert args.delete_cache_after_train is True
    assert args.resume is False


def test_config_rejects_unknown_sections():
    with pytest.raises(ValueError, match="unknown config section"):
        config_to_namespace({"command": "build_cache", "typo": {}})


def test_config_rejects_unknown_root_keys():
    with pytest.raises(ValueError, match="unknown config key"):
        config_to_namespace({"command": "build_cache", "typo": "value"})


def test_load_toml_config_reads_comments(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
        # Comments are allowed in TOML config files.
        command = "sample_xpred"

        [sample_xpred]
        checkpoint = "/tmp/model.safetensors"
        output = "/tmp/out.pt"
        """,
        encoding="utf-8",
    )

    data = load_toml_config(path)

    assert data["command"] == "sample_xpred"
    assert data["sample_xpred"]["checkpoint"] == "/tmp/model.safetensors"
