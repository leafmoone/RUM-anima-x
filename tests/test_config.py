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
                "bucket_enabled": True,
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
    assert args.bucket_enabled is True
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


def test_config_merges_tracker_and_gradient_checkpointing_settings():
    args = config_to_namespace(
        {
            "command": "train_xpred",
            "common": {"toy_smoke": True},
            "tracker": {
                "tracker_enabled": True,
                "tracker_project": "test-project",
                "tracker_mode": "offline",
                "tracker_run_id": "fixed-run",
                "tracker_resume": "allow",
                "tracker_tags": ["smoke", "xpred"],
                "tracker_metrics_file": "/tmp/eval.json",
                "tracker_metrics_log_every": 5,
            },
            "train_xpred": {
                "cache_dir": "/tmp/cache",
                "output_dir": "/tmp/train",
                "gradient_checkpointing": True,
                "gradient_checkpointing_cpu_offload": True,
                "sample_every_steps": 100,
                "sample_steps": 8,
                "sample_num_samples": 2,
                "sample_cfg": 2.5,
                "sample_eps_floor": 1e-5,
                "sample_prompt": "preview prompt",
                "sample_decode_images": True,
                "sample_tracker_log_images": True,
                "sample_lora": "/models/sample-lora.safetensors",
                "sample_lora_weight": 0.8,
                "sample_lora_steps": 10,
                "sample_lora_cfg": 3.5,
                "sample_lora_eps_floor": 1e-4,
                "sample_compare_every_steps": 200,
                "sample_compare_prompt": "compare prompt",
                "sample_compare_steps": 12,
                "sample_compare_num_samples": 3,
                "sample_compare_cfg": 4.5,
                "sample_compare_eps_floor": 1e-4,
                "sample_compare_width": 768,
                "sample_compare_height": 1344,
                "sample_compare_seed": 20260704,
                "sample_compare_output_dir": "/tmp/compare",
                "sample_compare_baseline_source_dir": "/tmp/alpha-0",
                "sample_compare_baseline_output_dir": "/tmp/baseline",
                "sample_compare_baseline_tracker_log_images": True,
                "sample_compare_lora": "/models/turbo.safetensors",
                "sample_compare_lora_weight": 0.25,
                "sample_compare_lora_cfg": 5.5,
                "sample_compare_teacher_sanity": True,
                "sample_compare_teacher_sanity_lora": "/models/sanity-lora.safetensors",
                "sample_compare_teacher_sanity_lora_weight": 0.6,
                "sample_compare_decode_images": True,
                "sample_compare_image_prefix": "compare",
                "sample_compare_tracker_log_images": True,
                "num_train_epochs": 2.0,
                "lr_scheduler": "cosine",
                "lr_warmup_steps": 10,
                "lr_cosine_min": 0.2,
                "dry_run": True,
            },
        }
    )

    assert args.tracker_enabled is True
    assert args.tracker_project == "test-project"
    assert args.tracker_mode == "offline"
    assert args.tracker_run_id == "fixed-run"
    assert args.tracker_resume == "allow"
    assert args.tracker_tags == ["smoke", "xpred"]
    assert args.tracker_metrics_file == "/tmp/eval.json"
    assert args.tracker_metrics_log_every == 5
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
    assert args.sample_cfg == 2.5
    assert args.sample_eps_floor == 1e-5
    assert args.sample_prompt == "preview prompt"
    assert args.sample_decode_images is True
    assert args.sample_tracker_log_images is True
    assert args.sample_lora == "/models/sample-lora.safetensors"
    assert args.sample_lora_weight == 0.8
    assert args.sample_lora_steps == 10
    assert args.sample_lora_cfg == 3.5
    assert args.sample_lora_eps_floor == 1e-4
    assert args.sample_compare_every_steps == 200
    assert args.sample_compare_prompt == "compare prompt"
    assert args.sample_compare_steps == 12
    assert args.sample_compare_num_samples == 3
    assert args.sample_compare_cfg == 4.5
    assert args.sample_compare_eps_floor == 1e-4
    assert args.sample_compare_width == 768
    assert args.sample_compare_height == 1344
    assert args.sample_compare_seed == 20260704
    assert args.sample_compare_output_dir == "/tmp/compare"
    assert args.sample_compare_baseline_source_dir == "/tmp/alpha-0"
    assert args.sample_compare_baseline_output_dir == "/tmp/baseline"
    assert args.sample_compare_baseline_tracker_log_images is True
    assert args.sample_compare_lora == "/models/turbo.safetensors"
    assert args.sample_compare_lora_weight == 0.25
    assert args.sample_compare_lora_cfg == 5.5
    assert args.sample_compare_teacher_sanity is True
    assert args.sample_compare_teacher_sanity_lora == "/models/sanity-lora.safetensors"
    assert args.sample_compare_teacher_sanity_lora_weight == 0.6
    assert args.sample_compare_decode_images is True
    assert args.sample_compare_image_prefix == "compare"
    assert args.sample_compare_tracker_log_images is True


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


def test_config_loads_sample_compare_settings():
    args = config_to_namespace(
        {
            "command": "sample_compare",
            "common": {"toy_smoke": True},
            "sample_compare": {
                "student_checkpoint": "/tmp/student.pt",
                "teacher_checkpoint": "/tmp/teacher.safetensors",
                "output_dir": "/tmp/compare",
                "prompt": "preview",
                "alphas": [0.0, 0.5, 1.0],
                "steps": 8,
                "decode_sample_images": True,
            },
        }
    )

    assert args.command == "sample_compare"
    assert args.student_checkpoint == "/tmp/student.pt"
    assert args.teacher_checkpoint == "/tmp/teacher.safetensors"
    assert args.output_dir == "/tmp/compare"
    assert args.alphas == [0.0, 0.5, 1.0]
    assert args.decode_sample_images is True


def test_config_inherits_build_cache_teacher_lora_for_compare_paths():
    config = {
        "command": "train_xpred",
        "common": {"toy_smoke": True},
        "build_cache": {
            "teacher_lora": "/models/turbo.safetensors",
            "teacher_lora_weight": 0.75,
        },
        "train_xpred": {
            "cache_dir": "/tmp/cache",
            "output_dir": "/tmp/train",
        },
        "sample_compare": {
            "student_checkpoint": "/tmp/student.pt",
            "output_dir": "/tmp/compare",
        },
    }

    train_args = config_to_namespace(config, command_override="train_xpred")
    compare_args = config_to_namespace(config, command_override="sample_compare")

    assert train_args.sample_compare_lora == "/models/turbo.safetensors"
    assert train_args.sample_compare_lora_weight == 0.75
    assert train_args.sample_lora == "/models/turbo.safetensors"
    assert train_args.sample_lora_weight == 0.75
    assert train_args.sample_compare_teacher_sanity_lora == "/models/turbo.safetensors"
    assert train_args.sample_compare_teacher_sanity_lora_weight == 0.75
    assert compare_args.teacher_lora == "/models/turbo.safetensors"
    assert compare_args.teacher_lora_weight == 0.75


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


def test_config_loads_prompt_sets_and_cache_dirs():
    build_args = config_to_namespace(
        {
            "command": "build_cache",
            "common": {"toy_smoke": True},
            "build_cache": {
                "prompt_sets": [
                    {"name": "tag", "prompts": "/tmp/tag.txt", "cache_dir": "/tmp/tag-cache", "start_index": 10, "num_samples": 30},
                    {"name": "nl", "prompts": "/tmp/nl.txt", "cache_dir": "/tmp/nl-cache", "start_index": 0, "num_samples": 30},
                ]
            },
        }
    )
    train_args = config_to_namespace(
        {
            "command": "train_xpred",
            "common": {"toy_smoke": True},
            "train_xpred": {
                "cache_dirs": ["/tmp/tag-cache", "/tmp/nl-cache"],
                "cache_mix_mode": "batch_weighted",
                "cache_mix_weights": [0.5, 0.5],
                "output_dir": "/tmp/out",
            },
        }
    )

    assert build_args.prompt_sets[0]["name"] == "tag"
    assert build_args.prompt_sets[1]["cache_dir"] == "/tmp/nl-cache"
    assert train_args.cache_dirs == ["/tmp/tag-cache", "/tmp/nl-cache"]
    assert train_args.cache_mix_mode == "batch_weighted"
    assert train_args.cache_mix_weights == [0.5, 0.5]


def test_config_rejects_global_build_cache_keys_inside_prompt_set():
    config = {
        "command": "build_cache",
        "common": {"toy_smoke": True},
        "build_cache": {
            "prompt_sets": [
                {
                    "name": "bad",
                    "prompts": "/tmp/prompts.txt",
                    "cache_dir": "/tmp/cache",
                    "cache_batch_size": 8,
                    "bucket_enabled": True,
                }
            ]
        },
    }

    with pytest.raises(ValueError, match="Move these keys before the first"):
        config_to_namespace(config, command_override="build_cache")


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
