# Runbook

All commands assume:

```bash
cd /root/shared-nvme/RUM-anima-xpred
```

Do not run this project from `/root/shared-nvme/RUM`. That is a separate workspace. The Anima x-pred project is intended to be standalone; use the scripts and vendored code inside `/root/shared-nvme/RUM-anima-xpred`.

For agent-to-agent continuity, read the root-level `agent.md` before changing code or config.

## 0. Current Workspace State

`configs/anima_xpred.example.toml` is the active experiment config in this workspace. It contains real model paths, chunk state, W&B run settings, LoRA settings, and sample/compare settings. Treat it as live state, not a disposable template.

Before making operational changes, check for a running training process:

```bash
ps -eo pid,etime,cmd | grep 'anima_rum_xpred_train.py' | grep -v grep
```

Do not interrupt a running process unless the user explicitly asks.

## 1. Verify Install

```bash
python -c "import diffusers, transformers, accelerate, safetensors; print('deps ok')"
PYTHONPATH=src python -c "from rum_xpred.adapters.anima_sd_scripts import create_adapter; print('adapter ok')"
pytest -q
```

## 2. Prepare Config

Use `configs/anima_xpred.example.toml` as the main editable config. It contains every normal option for all three stages:

- `[common]`: shared device, precision, Anima model paths, adapter, seed, attention mode, and FP8/text-encoder placement flags.
- `[wandb]`: optional wandb logging settings. It is disabled by default; set `wandb_enabled = true` and use `wandb_mode = "offline"` for local-only logging. `train/grad_norm` is logged directly by the trainer; FID/IS are read from `wandb_metrics_file` when an external image evaluator writes JSON such as `{"fid": 12.3, "is": 5.6}`.
- `[build_cache]`: prompt file, cache directory, start index, cache batch size, skip-existing resume behavior, resolution, teacher steps, teacher CFG.
- `[train_xpred]`: cache directory, output directory, `prediction_type`, epoch/step count, train batch size, gradient accumulation, AdamW parameters, LR scheduler, grad clipping, sigma lower bound, shuffle/drop-last, logging interval, periodic checkpoints, gradient checkpointing, optional training-time sampling, dry run. If `max_train_steps` is omitted, steps are computed from `num_train_epochs`, cache sample count, and effective batch size.
- `[sample_xpred]`: checkpoint, `prediction_type`, latent output path, prompt, sample count, sampler steps, resolution, epsilon floor.
- `[chunked_rum]`: automatic rolling loop that builds one cache chunk, trains on it, then continues with the next chunk from the previous checkpoint.

You can run the config's `command` directly:

```bash
python scripts/dev/anima_rum_xpred_train.py --config configs/anima_xpred.example.toml
```

Or explicitly select a stage with a subcommand:

```bash
python scripts/dev/anima_rum_xpred_train.py build_cache --config configs/anima_xpred.example.toml
python scripts/dev/anima_rum_xpred_train.py train_xpred --config configs/anima_xpred.example.toml
python scripts/dev/anima_rum_xpred_train.py sample_xpred --config configs/anima_xpred.example.toml
python scripts/dev/anima_rum_xpred_train.py sample_compare --config configs/anima_xpred.example.toml
python scripts/dev/anima_rum_xpred_train.py chunked_rum --config configs/anima_xpred.example.toml
```

For the current machine, the example already points at:

```text
/root/shared-nvme/anima/split_files/diffusion_models/anima-base-v1.0.safetensors
/root/shared-nvme/anima/split_files/text_encoders/qwen_3_06b_base.safetensors
/root/shared-nvme/anima/split_files/vae/qwen_image_vae.safetensors
```

## 3. Toy Smoke

For a fast local smoke test, copy the example config and change these values:

```toml
[common]
toy_smoke = true
mixed_precision = "fp32"

[build_cache]
cache_dir = "/tmp/rum-anima-xpred-cache-smoke"
num_samples = 2
width = 64
height = 64

[train_xpred]
cache_dir = "/tmp/rum-anima-xpred-cache-smoke"
output_dir = "/tmp/rum-anima-xpred-train-smoke"
max_train_steps = 3
learning_rate = 1e-4

[sample_xpred]
checkpoint = "/tmp/rum-anima-xpred-train-smoke/xpred-toy-smoke.pt"
output = "/tmp/rum-anima-xpred-train-smoke/sample-latent.pt"
width = 64
height = 64
steps = 4
```

Then run:

```bash
python scripts/dev/anima_rum_xpred_train.py --config /path/to/toy-config.toml
python scripts/dev/anima_rum_xpred_train.py build_cache --config /path/to/toy-config.toml
python scripts/dev/anima_rum_xpred_train.py train_xpred --config /path/to/toy-config.toml
python scripts/dev/anima_rum_xpred_train.py sample_xpred --config /path/to/toy-config.toml
```

## 4. Real Cache Build

Edit `[build_cache]` in `configs/anima_xpred.example.toml`, then run:

```bash
python scripts/dev/anima_rum_xpred_train.py build_cache --config configs/anima_xpred.example.toml
```

The cache contains latent tensors only:

- `eps_latent`
- `x_teacher_latent`
- text conditioning tensors
- metadata such as seed, resolution, teacher steps, flow shift, and teacher CFG

It does not decode to pixels.

The current example config uses the Anima Turbo LoRA as a 10-step teacher:

```toml
[build_cache]
cache_dir = "/root/shared-nvme/cache_data/anima-xpred-turbo10-cache"
teacher_steps = 10
teacher_lora = "/root/shared-nvme/anima/anima-turbo-lora-v0.2.safetensors"
teacher_lora_weight = 1.0
```

Do not mix this cache directory with the older base 30-step cache. They are different teacher endpoint distributions.

To enable automatic fixed resolution bucketing, set:

```toml
[build_cache]
bucket_enabled = true
```

The concrete bucket list is fixed in code. You do not need to write resolutions into the config. Samples are assigned deterministically from `seed + sample_index` and saved as:

```text
<cache_dir>/1024x1024/sample-000000.safetensors
<cache_dir>/832x1216/sample-000001.safetensors
...
```

`train_xpred` and `chunked_rum` discover these bucket subdirectories automatically. Each optimizer micro-batch is read from a single bucket, so different latent shapes are never concatenated in one batch.

Multiple prompt files can be cached into separate directories with `[[build_cache.prompt_sets]]`. In `chunked_rum`, each set receives its own `chunk-XXXX` directory, so identical sample names from different prompt files do not collide:

```toml
[[build_cache.prompt_sets]]
name = "tag"
prompts = "/root/shared-nvme/RUM/data/prompts/qat_prompts.txt"
cache_dir = "/root/shared-nvme/RUM-anima-xpred/cache_mix/tag"
start_index = 177000
num_samples = 50000
repeat = 1
cache_chunk_offset = 14

[[build_cache.prompt_sets]]
name = "short_nl"
prompts = "/root/shared-nvme/RUM/data/prompts/qat_prompts_chars_short_vibes_few_words_50k.txt"
cache_dir = "/root/shared-nvme/RUM-anima-xpred/cache_mix/short_nl"
start_index = 0
num_samples = 50000
repeat = 1
cache_chunk_offset = 0
```

`cache_chunk_offset` only changes the cache directory number, not the prompt index. With the example above, training `chunk-0000` reads/writes `tag/chunk-0014` and `short_nl/chunk-0000`; training `chunk-0001` uses `tag/chunk-0015` and `short_nl/chunk-0001`.

`repeat` lets a smaller prompt set stay active for more training chunks. Its effective length is:

```text
effective_samples = num_samples * repeat
```

For example, a 10k prompt set with `repeat = 5` contributes across 50k effective samples by cycling through its own prompt range five times. Only the first cycle runs the teacher and writes real cache files. Later repeat cycles reuse those first-cycle cache files by hardlinking them into the current chunk directory, falling back to symlink or copy if hardlinks are unavailable. If the first-cycle source file is missing, `chunked_rum` fails instead of regenerating it.

If a chunk crosses the end of the prompt range, `chunked_rum` splits that set into contiguous slices. The first-cycle slice is generated normally; repeated slices are linked from existing cache. When `chunked_rum.total_samples` is omitted, the chunk plan is derived from the largest effective prompt-set length.

## 5. Rolling Chunked RUM

For large runs where you do not want to cache everything before training, use:

```bash
python scripts/dev/anima_rum_xpred_train.py chunked_rum --config configs/anima_xpred.example.toml
```

The loop is:

```text
cache chunk-0000 -> train chunk-0000 -> cache chunk-0001 -> train chunk-0001 -> ...
```

Each next chunk uses the previous chunk's final student checkpoint as `student_init` and loads the previous AdamW state from `<prediction_type>pred-train-state.pt`. This preserves optimizer moments across chunk boundaries instead of restarting AdamW each chunk. Resume is controlled by:

```toml
[chunked_rum]
chunk_root = "/root/shared-nvme/RUM-anima-xpred/anima-xpred-chunks"
total_samples = 30000
chunk_size = 1024
resume = true
delete_cache_after_train = false
```

Progress is recorded in:

```text
<chunk_root>/chunk-manifest.json
```

The manifest records both:

```text
checkpoint       # xpred-adapter-checkpoint.safetensors or vpred-adapter-checkpoint.safetensors
optimizer_state  # xpred-train-state.pt or vpred-train-state.pt
```

On resume, completed chunks are skipped and their checkpoint/state paths are used for the next incomplete chunk. An incomplete `training` chunk is still restarted from the previous completed chunk; in-chunk dataloader position and partial optimizer steps are not restored.

Set `delete_cache_after_train = true` only when you want to save disk and are comfortable rebuilding a chunk if you need to rerun it. On a single GPU this is sequential, not parallel prefetching.

When `[[build_cache.prompt_sets]]` is enabled, `chunked_rum` trains on all prompt-set cache directories active for the current chunk. If `[train_xpred].cache_mix_weights` is omitted, it sets weights from each prompt set's effective sample count, so a 50k set and a 10k set with `repeat = 5` receive equal sampling weight. Automatic step count uses the combined active sample count. For example, two cache dirs with 3000 samples each and effective batch size 32 produce `ceil(6000 / 32) = 188` optimizer steps for one epoch.

## 6. Real X-Pred Training

Edit `[train_xpred]`, then run:

```bash
python scripts/dev/anima_rum_xpred_train.py train_xpred --config configs/anima_xpred.example.toml
```

To train directly from multiple existing cache directories, use:

```toml
[train_xpred]
cache_dirs = [
  "/path/to/tag/chunk-0000",
  "/path/to/short_nl/chunk-0000",
]
cache_mix_mode = "batch_weighted"
cache_mix_weights = [0.5, 0.5]
```

The trainer draws each micro-batch from one resolution bucket and splits samples across cache directories according to the weights. With `train_batch_size = 8` and weights `[0.5, 0.5]`, each micro-batch contains four samples from each directory when both have the selected bucket.

Training always logs `train/x_mse`, the clean-latent MSE against `x_teacher_latent`, even when the active training objective is JLT velocity-readout loss. When multiple cache directories are active, wandb also logs `train/loss_by_cache/<cache-name>` and `train/x_mse_by_cache/<cache-name>`. For chunked paths such as `tag/chunk-0014`, the cache name is `tag`.

The student starts from `[common].student_init`; if that is empty, it falls back to `[common].dit`.

Default output:

```text
/tmp/anima-xpred-train/xpred-adapter-checkpoint.safetensors
```

Training-time sampling is off by default:

```toml
[train_xpred]
sample_every_steps = 0
```

Set it to a positive optimizer-step interval to sample from the current in-memory student after `optimizer.step()`:

```toml
[train_xpred]
sample_every_steps = 1000
sample_prompt = "hatsune miku, 1girl, ..."
sample_steps = 10
sample_num_samples = 2
sample_cfg = 1.0
sample_eps_floor = 1e-4
sample_decode_images = true
sample_wandb_log_images = true
```

Latents are always saved to:

```text
<output_dir>/train-samples/latents/
```

If `sample_decode_images = true`, PNGs are saved to:

```text
<output_dir>/train-samples/images/step-000000/
```

For ordinary `train_xpred`, the interval is counted from step 1. For `chunked_rum`, training samples and LR scheduling are counted by total optimizer steps across completed chunks; resumed complete chunks are included in the offset. For example, if completed chunks have 1878 total steps and `sample_every_steps = 500`, the next training sample is written at total step 2000.

If `lr_scheduler_total_steps` is omitted in `chunked_rum`, the script estimates the optimizer-step total for all planned chunks and uses that for cosine decay. This keeps warmup and decay continuous across chunks instead of restarting at each chunk.

When `wandb_enabled = true`, `sample_decode_images = true`, and `sample_wandb_log_images = true`, those PNGs are logged to wandb as `sample/images` at the total training step. Toy smoke saves latent previews only; image decode requires the real Anima VAE.

The ordinary training sample can also generate a separate LoRA branch. Its CFG is configured independently from the non-LoRA branch:

```toml
[train_xpred]
sample_cfg = 1.0
sample_lora = "__inherit__"
sample_lora_weight = 1.0
sample_lora_steps = 10
sample_lora_cfg = 1.0
```

If `sample_lora_cfg` is omitted, the LoRA branch inherits `sample_cfg`.

Training-time `sample_compare` is separate from the ordinary training sample and is off by default:

```toml
[train_xpred]
sample_compare_every_steps = 0
```

Enable it with a positive interval:

```toml
[train_xpred]
sample_compare_every_steps = 1000
sample_compare_prompt = "hatsune miku, 1girl, ..."
sample_compare_steps = 30
sample_compare_num_samples = 2
sample_compare_cfg = 1.0
sample_compare_lora = "__inherit__"
sample_compare_lora_weight = 1.0
sample_compare_lora_cfg = 1.0
sample_compare_decode_images = true
sample_compare_wandb_log_images = true
```

This path runs the current in-memory student in multiple interpretations:

- `fm`: treat the current output as the old Anima FM/v prediction.
- `x`: treat the current output as clean latent x and convert it to Anima velocity during sampling.
- `fm_lora`: temporarily load the current student checkpoint with `sample_compare_lora`, then sample it as FM/v.
- `x_lora`: the same temporary LoRA student sampled as x-pred.
- `teacher_sanity`: optional teacher run with the same prompt, noise, steps, and optional LoRA, used to verify that schedule/model/LoRA/decode are sane.

This is intended to show the migration progress: early checkpoints should look better as `fm`, while a successful x-pred conversion should gradually improve `x`. The LoRA variants show whether the acceleration LoRA still helps or harms the current migrated student.

`sample_compare_cfg` applies to `fm` and `x`. `sample_compare_lora_cfg` applies to `fm_lora` and `x_lora`; if omitted, it inherits `sample_compare_cfg`.

If `x` is coherent while `fm` is pure noise, that can be expected after migration: the checkpoint no longer behaves like an FM/v model. If `teacher_sanity` is also bad, debug the teacher path, LoRA path, sigma schedule, or VAE decode before drawing conclusions from student samples.

Outputs are saved under:

```text
<output_dir>/train-compare-samples/step-000000/
```

If `wandb_enabled = true`, `sample_compare_decode_images = true`, and `sample_compare_wandb_log_images = true`, the compare PNGs are also logged to wandb `sample_compare/images` at the same total training step. It currently requires `prediction_type = "x"`.

Teacher baseline images do not need to be regenerated every compare interval. To import an already generated `alpha-0` directory once and upload it to wandb:

```toml
[train_xpred]
sample_compare_baseline_source_dir = "/path/to/train-compare-samples/step-002180/images/alpha-0"
sample_compare_baseline_wandb_log_images = true
```

Those images are imported into the configured baseline directory and logged to `sample_compare/baseline`.
If `sample_compare_baseline_output_dir` is omitted, the project-level directory is used:

```text
<repo>/compare-baseline/
```

The current project-level baseline directory is:

```text
/root/shared-nvme/RUM-anima-xpred/compare-baseline/
```

A `.wandb_uploaded` marker prevents repeated baseline uploads.

The older manual `sample_compare` command still supports teacher/student alpha mixing. For that manual command, teacher LoRA defaults to inheriting `[build_cache].teacher_lora`; set it to `""` only when you intentionally want a pure base teacher baseline:

```toml
[sample_compare]
teacher_lora = ""
teacher_lora_weight = 1.0
```

## Velocity Reflow Control

The main config still defaults to `prediction_type = "x"`. To train a standard velocity reflow student without changing cache format, use:

```bash
python scripts/dev/anima_rum_xpred_train.py train_xpred --config configs/anima_vpred_reflow.example.toml
```

The v-pred target is:

```text
target_v = eps_latent - x_teacher_latent
```

Sampling must also use `prediction_type = "v"`:

```bash
python scripts/dev/anima_rum_xpred_train.py sample_xpred --config configs/anima_vpred_reflow.example.toml
```

Do not sample a v-pred checkpoint with `prediction_type = "x"`, and do not sample an x-pred checkpoint with `prediction_type = "v"`.

## 7. Real X-Pred Sampling

Do not use the old Anima FM sampler with an x-pred checkpoint. Use this dedicated sampler:

```bash
python scripts/dev/anima_rum_xpred_train.py sample_xpred --config configs/anima_xpred.example.toml
```

The current sampler writes the final latent tensor. Pixel preview/decode is intentionally separate.

To log visual samples to wandb, enable image decode and wandb in the config:

```toml
[wandb]
wandb_enabled = true
wandb_mode = "offline"  # or leave empty for online logging

[sample_xpred]
decode_sample_images = true
sample_image_dir = "/tmp/anima-xpred-sample/images"
wandb_log_sample_images = true
```

The sampler will save PNG files locally and log them to wandb as `sample/images`.

## 8. Cache Upload Automation

Completed chunk caches can be packaged and uploaded without using the GPU.

The project-local implementation is:

```bash
python scripts/dev/upload_completed_cache_chunks.py --help
```

The local private launcher is:

```bash
./watch_upload_completed_cache_chunks.sh
```

That launcher is ignored by git because it contains a ModelScope token. Keep secrets there or in environment variables, not in committed docs/configs.

The uploader reads:

```text
<chunk_root>/chunk-manifest.json
```

and uploads only chunks whose status is `complete`. With the current launcher it uploads to the repository's `tag` directory using this remote layout:

```text
tag/<id-min-id-max>/<resolution>/<resolution>.tar
```

For example:

```text
tag/69000-71999/1024x1024/1024x1024.tar
```

With `--delete-cache-after-upload`, each local cache chunk is deleted only after all resolution tar uploads for that chunk succeed.

## 9. Teacher/Student Compare Sampling

Use `sample_compare` to inspect conversion progress with fixed prompt and seed:

```bash
python scripts/dev/anima_rum_xpred_train.py sample_compare --config configs/anima_xpred.example.toml
```

It samples the same initial noise with multiple velocity mixes:

```text
alpha = 0.0  teacher/FM velocity only
alpha = 0.25 mixed velocity
alpha = 0.5  mixed velocity
alpha = 0.75 mixed velocity
alpha = 1.0  student x-pred converted to velocity
```

The update is:

```text
v_teacher = teacher(z, sigma)
v_student = (z - x_student) / max(sigma, eps_floor)
v_mix = (1 - alpha) * v_teacher + alpha * v_student
z_next = z + v_mix * (sigma_next - sigma)
```

This reuses the Anima Euler ODE path while changing only the velocity source. It is slower than normal sampling because each step runs both teacher and student.

## Notes

- `x` means clean Anima VAE latent, not RGB pixels.
- Teacher CFG defaults to `1.0` to avoid double guidance.
- Default training samples the actual shifted Anima sigma directly from `[sigma_min_train, 1]`; `time_sampling="jlt_logit_normal"` uses JLT's `t=sigmoid(N(P_mean,P_std))` and maps to `sigma=1-t`.
- Terminal `sigma=0` is never forwarded through `v=(z-x)/sigma`.
