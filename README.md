# RUM Anima X-Pred

Standalone project for converting an Anima Rectified Flow DiT from latent velocity prediction to clean latent `x` prediction with RUM reflow.

This project is only for the x-pred conversion experiment. It does not keep ordinary Anima LoRA or full-finetune training entrypoints.

## Layout

- `src/rum_xpred/` - x-pred formula, cache, sampler, and local Anima adapter.
- `scripts/dev/anima_rum_xpred_train.py` - experiment CLI.
- `scripts/dev/upload_completed_cache_chunks.py` - project-local cache packaging/upload helper.
- `configs/anima_xpred.example.toml` - commented config covering cache, train, and sample.
- `configs/anima_vpred_reflow.example.toml` - optional velocity reflow control config using the same cache format.
- `vendor/sd-scripts/` - copied local Anima/kohya code used by the adapter.
- `docs/` - formula and project notes.
- `agent.md` - current handoff notes for future agents.
- `tests/` - lightweight tests.
- `data/prompts/sample_prompts.txt` - small prompt file for smoke tests.

## Agent Handoff

If another agent continues this project, start with `agent.md`. It records the active root directory, live experiment config, chunked resume behavior, sampling semantics, upload helper, runtime artifacts, and files that must not be committed.

This repository should be treated as independent from `/root/shared-nvme/RUM`; do not route Anima x-pred commands through scripts from that directory.

## Config-First Usage

Edit `configs/anima_xpred.example.toml` for your paths and output directories. All normal experiment options live in the config file.

```bash
python scripts/dev/anima_rum_xpred_train.py --config configs/anima_xpred.example.toml
python scripts/dev/anima_rum_xpred_train.py build_cache --config configs/anima_xpred.example.toml
python scripts/dev/anima_rum_xpred_train.py train_xpred --config configs/anima_xpred.example.toml
python scripts/dev/anima_rum_xpred_train.py sample_xpred --config configs/anima_xpred.example.toml
python scripts/dev/anima_rum_xpred_train.py chunked_rum --config configs/anima_xpred.example.toml
```

Without a subcommand, the script uses `command` from the config. With a subcommand, the subcommand selects the stage. The same config can contain `[build_cache]`, `[train_xpred]`, and `[sample_xpred]` sections.

The example config includes the normal experiment controls: cache start index, cache batch size, skip-existing resume behavior, optional built-in resolution bucketing, train batch size, gradient accumulation, AdamW parameters, LR scheduler, grad clipping, shuffle/drop-last, periodic checkpoints, gradient checkpointing, wandb logging, dry run, and sample count.

In the current workspace, `configs/anima_xpred.example.toml` is also the live experiment config. Inspect it before editing because it contains real paths, W&B run settings, chunk offsets, and sample/compare settings.

Set `[build_cache].bucket_enabled = true` to use the built-in fixed resolution buckets. Cache files are written under `cache_dir/<width>x<height>/`; training discovers those subdirectories automatically and samples each optimizer micro-batch from one bucket so latent shapes do not mix inside a batch.

Training is epoch-driven by default: omit `max_train_steps` and set `num_train_epochs`. The script computes optimizer steps from the number of cache samples and the effective batch size.

The default training mode is still `prediction_type = "x"`. A v-pred reflow control path is available with `prediction_type = "v"` in both `[train_xpred]` and `[sample_xpred]`; use `configs/anima_vpred_reflow.example.toml` for that experiment.

Training-time sample previews are supported but disabled by default. Set `[train_xpred].sample_every_steps` to a positive interval to save preview latents under `<output_dir>/train-samples/`; enable `sample_decode_images` and wandb to also log decoded PNGs as `sample/images`.

## Rolling Chunked RUM

`chunked_rum` automates the original RUM-style loop for large runs:

```text
build cache chunk N -> train on chunk N -> use checkpoint/state N for chunk N+1 -> repeat
```

It writes chunk caches under `[chunked_rum].chunk_root/cache/`, chunk checkpoints and optimizer states under `[chunked_rum].chunk_root/train/`, and resume state to `chunk-manifest.json`. Each next chunk inherits both the previous student checkpoint and AdamW optimizer state. This is sequential on one GPU; it does not run teacher cache generation in parallel with student training.

## Toy Smoke

Use the same config path, but set `[common].toy_smoke = true`, `[common].mixed_precision = "fp32"`, and small sizes such as `64x64`.

```bash
pytest -q
python scripts/dev/anima_rum_xpred_train.py build_cache --config /path/to/toy-config.toml
python scripts/dev/anima_rum_xpred_train.py train_xpred --config /path/to/toy-config.toml
python scripts/dev/anima_rum_xpred_train.py sample_xpred --config /path/to/toy-config.toml
```

## Real Adapter

The default real adapter is local:

```text
rum_xpred.adapters.anima_sd_scripts:create_adapter
```

Real cache build needs these config fields:

```toml
[common]
dit = "/path/to/anima-dit.safetensors"
text_encoder = "/path/to/qwen3-or-qwen3.safetensors"

[build_cache]
prompts = "/path/to/prompts.txt"
cache_dir = "/path/to/cache"
width = 1024
height = 1024
```

The adapter uses only files inside this project plus the model paths you set in the config.

See `docs/runbook.md` for the full build-cache, train, and sample command sequence.

`sample_xpred` can optionally decode sampled latents with the Anima/Qwen VAE and log the PNGs to wandb as `sample/images`. Enable `[sample_xpred].decode_sample_images` and `[wandb].wandb_enabled` in the config.
