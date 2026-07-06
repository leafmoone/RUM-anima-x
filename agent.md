# Agent Handoff

This file is the first stop for the next agent. The active project is:

```bash
cd /root/shared-nvme/RUM-anima-xpred
```

Do not use `/root/shared-nvme/RUM` as the project root for this work. That directory is a separate project/workspace. The Anima x-pred project is intended to be standalone and should not call scripts from `/root/shared-nvme/RUM`.

## Current Intent

The repository converts an Anima Rectified Flow DiT from Anima velocity/FM prediction to clean Anima VAE latent `x` prediction through RUM-style endpoint caching and retraining.

Important contract:

- `x` means clean Anima VAE latent, not RGB pixels.
- Cache stores `eps_latent`, `x_teacher_latent`, text conditioning, and metadata.
- x-pred sampling converts the student output back to Anima velocity with:

```text
v = (z - x_pred) / max(sigma, eps_floor)
z_next = z + v * (sigma_next - sigma)
```

- Never run terminal `sigma=0` through the division.
- Do not load an x-pred checkpoint with the old Anima FM sampler.

## Active Runtime State

At the time this handoff was written, a training process was running:

```text
python scripts/dev/anima_rum_xpred_train.py --config configs/anima_xpred.example.toml
```

Check before touching GPU state:

```bash
ps -eo pid,etime,cmd | grep 'anima_rum_xpred_train.py' | grep -v grep
```

Do not kill or restart it unless the user explicitly asks.

## Source vs Runtime Artifacts

Source files live in:

- `src/rum_xpred/`
- `scripts/dev/`
- `configs/`
- `docs/`
- `tests/`
- `vendor/sd-scripts/`

Runtime/generated artifacts include:

- `anima-jlt-xpred-turbo10-chunks/`
- `anima-xpred-train/`
- `cache/`, `cache_data/`, `*-cache/`, `*cache*/`
- `compare-baseline/`
- `wandb/`
- local PDFs such as `2605.27102v2.pdf`

Do not commit generated cache, checkpoints, W&B logs, or private launchers. The local launcher `watch_upload_completed_cache_chunks.sh` contains a ModelScope token and is ignored by git.

## Active Config

`configs/anima_xpred.example.toml` is currently being used as the live experiment config, not just a clean template. It contains real paths, chunk offsets, W&B settings, and sample settings.

Current highlights:

- default command: `chunked_rum`
- chunk root: `/root/shared-nvme/RUM-anima-xpred/anima-jlt-xpred-turbo10-chunks`
- base Anima model path: `/root/shared-nvme/anima/split_files/diffusion_models/anima-base-v1.0.safetensors`
- teacher Turbo LoRA: `/root/shared-nvme/anima/anima-turbo-lora-v0.2.safetensors`
- teacher steps: `10`
- training prediction type: `x`
- time sampling: `jlt_logit_normal`
- loss weighting: `jlt_velocity_readout`
- ordinary training sample: main branch uses 30 steps; LoRA branch inherits teacher LoRA and uses 10 steps

Before making config changes, inspect the active values instead of assuming defaults:

```bash
sed -n '1,470p' configs/anima_xpred.example.toml
```

## Chunked RUM Resume

`chunked_rum` uses:

```text
<chunk_root>/chunk-manifest.json
```

Completed chunks are skipped when `resume = true`. Each next chunk inherits:

- previous `xpred-adapter-checkpoint.safetensors`
- previous `xpred-train-state.pt` optimizer/scheduler state

In-chunk dataloader position is not restored. If a chunk was interrupted while `training`, it restarts from the previous completed chunk.

When prompt sets are enabled, chunk `N` can map to different physical cache chunk ids per prompt set via `cache_chunk_offset`. This is intentional. For example, tag data can already be at `chunk-0014` while a new natural-language prompt set starts at `chunk-0000`.

## Sampling Semantics

There are two separate training-time visual paths:

1. Ordinary `sample_every_steps`
   - samples the current in-memory x-pred student.
   - saves `sample/images`.
   - optional LoRA branch is controlled by `sample_lora`, `sample_lora_steps`, and `sample_lora_weight`.

2. `sample_compare_every_steps`
   - diagnostic only.
   - runs the same current student in multiple interpretations:
     - `fm`: treat current output as FM/v.
     - `x`: treat current output as x-pred.
     - `fm_lora`: temporarily load current student checkpoint with LoRA, then treat as FM/v.
     - `x_lora`: same temporary LoRA student treated as x-pred.
     - optional `teacher_sanity`: teacher with the same prompt/noise/steps/LoRA.
   - logs to `sample_compare/images` when enabled.

If `x` looks normal and `fm` looks like noise, that can be a valid sign that the student has migrated away from FM/v interpretation. If `teacher_sanity` is also bad, investigate model paths, LoRA, sigma schedule, or decoding first.

Baseline teacher images are project-level artifacts under:

```text
/root/shared-nvme/RUM-anima-xpred/compare-baseline/
```

They should be uploaded to W&B once as `sample_compare/baseline`, guarded by `.wandb_uploaded`.

## Cache Upload Automation

The project-local uploader is:

```bash
python scripts/dev/upload_completed_cache_chunks.py --help
```

The ignored launcher is:

```bash
./watch_upload_completed_cache_chunks.sh
```

Remote ModelScope layout is:

```text
tag/<start-end>/<resolution>/<resolution>.tar
```

Example:

```text
tag/69000-71999/1024x1024/1024x1024.tar
```

The upload script reads completed chunks from `chunk-manifest.json`, packages each resolution directory, uploads it, and can delete the local cache after upload when `--delete-cache-after-upload` is set.

## Verification Commands

Use these before handing off code changes:

```bash
pytest -q
python scripts/dev/anima_rum_xpred_train.py --config configs/anima_xpred.example.toml --help
```

For docs-only changes, tests are not strictly required, but `git status --short` should be checked so generated artifacts and private files are not accidentally included.

## Git Hygiene

The worktree may already contain user/previous-agent changes. Do not revert unrelated changes. Before committing or pushing, inspect:

```bash
git status --short
git diff --stat
git diff --cached --stat
```

Never commit secrets. In particular, do not stage `watch_upload_completed_cache_chunks.sh`.
