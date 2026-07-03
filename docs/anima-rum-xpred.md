# Anima FM -> RUM Reflow Latent X-Pred

This note defines the local experiment for converting a trained Anima Rectified Flow DiT from velocity prediction to clean latent prediction.

`x` always means the clean endpoint in Anima VAE latent space. It is not RGB pixels.

## Formula contract

Anima's current flow uses shifted `sigma`:

```text
z_sigma = (1 - sigma) * x0 + sigma * eps
target_v = eps - x0
v_anima = dz/dsigma = eps - x0
```

Sampling runs from `sigma=1` down to `sigma=0`, so Euler updates are:

```text
z_next = z + v_anima * (sigma_next - sigma)
```

The new student predicts the clean latent endpoint:

```text
x_pred ~= x0
```

To use this student in an Anima-style Euler sampler:

```text
v_pred = (z - x_pred) / sigma
z_next = z + v_pred * (sigma_next - sigma)
```

The denominator is `sigma` under Anima variables. If using JLT clean-time `t = 1 - sigma`, the denominator becomes `1 - t`, and the velocity sign flips:

```text
v_jlt = x0 - eps = -v_anima
```

## RUM reflow target

The teacher is frozen. For each prompt and seed:

1. Sample `eps_latent` in Anima latent shape `[B, 16, H/8, W/8]`.
2. Run the original Anima FM teacher from `sigma=1` to `sigma=0` using the actual shifted sigma schedule.
3. Cache the terminal `x_teacher_latent`.
4. Train the student on the straight line between `eps_latent` and `x_teacher_latent`:

```text
sigma ~ shifted Uniform([sigma_min_train, 1])
z = (1 - sigma) * x_teacher_latent + sigma * eps_latent
loss = MSE(student(z, sigma, cond), x_teacher_latent)
```

This reflow target does not replay the teacher's intermediate trajectory. It only distills the teacher endpoint into a new straight flow.

JLT's public implementation uses a JiT-style x-pred loss: the clean prediction is read out to velocity before computing loss. In JLT variables this is `v_pred = (x_pred - z_t) / (1 - t)` and `loss = MSE(v_pred, v)`, which weights clean prediction error by `(1 - t)^-2`. Under Anima variables, `1 - t = sigma` and `v_anima = -v_jlt`, so the equivalent loss is:

```text
v_pred = (z - x_pred) / max(sigma, loss_eps_floor)
v_target = (z - x_teacher_latent) / max(sigma, loss_eps_floor)
loss = MSE(v_pred, v_target)
```

This project exposes that as:

```toml
loss_weighting = "jlt_velocity_readout"
loss_eps_floor = 5e-2
```

The v1 default remains unweighted clean-latent MSE (`loss_weighting = "none"`) for stability and to keep the RUM endpoint experiment simple; use a separate run when comparing against the JLT-style objective.

## Shift consistency

Use the same shifted `sigma` everywhere:

- teacher sampling schedule
- cached metadata
- training `z` construction
- model timestep input
- x-pred sampler schedule
- velocity denominator

The helper implementation uses:

```text
shifted_sigma = flow_shift * sigma / (1 + (flow_shift - 1) * sigma)
```

If upstream Anima changes its sigma shift formula, update the helper and tests before using the experiment.

## Sampling rules

An x-pred checkpoint must not be loaded into the old FM velocity sampler.

Dedicated sampler loop:

```python
sigmas = make_shifted_sigma_schedule(steps=40, flow_shift=3.0)
z = eps_latent
for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
    x_pred = student(z, sigma)
    v = (z - x_pred) / max(sigma, eps_floor)
    z = z + v * (sigma_next - sigma)
```

Do not call `v=(z-x)/sigma` at terminal `sigma=0`. The loop only forwards over `sigmas[:-1]`, and `eps_floor=1e-4` is kept as a NaN guard.

## Prediction type

The default mode remains clean latent x prediction:

```toml
[train_xpred]
prediction_type = "x"
```

Training target:

```text
target = x_teacher_latent
loss = MSE(model(z, sigma, cond), target)
```

For a standard velocity reflow control experiment, use:

```toml
[train_xpred]
prediction_type = "v"

[sample_xpred]
prediction_type = "v"
```

Velocity reflow uses the same cache, but changes the target:

```text
target_v = eps_latent - x_teacher_latent
loss = MSE(model(z, sigma, cond), target_v)
```

Sampling then treats the model output directly as Anima velocity:

```python
v_pred = student(z, sigma)
z = z + v_pred * (sigma_next - sigma)
```

This avoids the x-pred sampler's `v=(z-x)/sigma` conversion and is useful for comparing stability at very low sample step counts. Do not mix x-pred and v-pred checkpoints with the wrong sampler; set `[sample_xpred].prediction_type` to match the checkpoint.

## Training-time samples

`train_xpred` can periodically sample the current student without reloading a checkpoint:

```toml
[train_xpred]
sample_every_steps = 1000
sample_prompt = "..."
sample_steps = 10
sample_num_samples = 2
sample_eps_floor = 1e-4
sample_decode_images = true
```

The hook runs after `optimizer.step()` when the active training step reaches the interval. In `chunked_rum`, this is the total optimizer step across completed chunks, not the local step inside the current chunk. It uses the same `prediction_type` as training. Latent previews are saved under `output_dir/train-samples/latents`; decoded images are saved under `output_dir/train-samples/images` and logged to wandb as `sample/images` when wandb is enabled.

## CFG

Default `teacher_cfg=1.0` avoids baking teacher guidance into the endpoint and then applying guidance again at sample time.

If experimenting with guided teacher endpoints, record `teacher_cfg` in cache metadata and treat sample-time CFG as a separate explicit parameter. For x-pred sampling, mix conditional and unconditional predictions in x-pred space first, then convert the mixed x-pred to velocity.

## Default parameters

```text
teacher_steps = 40
flow_shift = 3.0
teacher_cfg = 1.0
sigma_min_train = 0.02
eps_floor = 1e-4
mixed_precision = bf16
learning_rate = 1e-6
loss_weighting = "none"
student_init = teacher_checkpoint
```

## Config-first CLI

The repository includes a local copy of the Anima `vendor/sd-scripts` code. Ordinary Anima LoRA/full-finetune training entrypoints are intentionally not part of this project.

The RUM x-pred command uses a local adapter because endpoint-cache generation and x-pred student forwarding are new experiment behavior, not a native upstream sd-scripts mode. Use `[common].toy_smoke = true` for local verification without loading Anima model weights.

The main editable example is:

```text
configs/anima_xpred.example.toml
```

Run each stage with the same config:

```bash
python scripts/dev/anima_rum_xpred_train.py build_cache --config configs/anima_xpred.example.toml
python scripts/dev/anima_rum_xpred_train.py train_xpred --config configs/anima_xpred.example.toml
python scripts/dev/anima_rum_xpred_train.py sample_xpred --config configs/anima_xpred.example.toml
python scripts/dev/anima_rum_xpred_train.py chunked_rum --config configs/anima_xpred.example.toml
```

The subcommand selects which config section is executed:

- `[common]`: shared model paths, precision, seed, adapter, attention, and memory flags.
- `[build_cache]`: prompts, cache directory, resolution, teacher steps, teacher CFG.
- `[train_xpred]`: cache directory, output directory, optimizer settings, sigma lower bound.
- `[sample_xpred]`: x-pred checkpoint, latent output, prompt, sampler steps, epsilon floor.
- `[chunked_rum]`: rolling cache/train chunks, resume manifest, optional cache deletion after each chunk.

The v-pred control config is:

```text
configs/anima_vpred_reflow.example.toml
```

For real RUM x-pred integration, the default local adapter is:

```text
rum_xpred.adapters.anima_sd_scripts:create_adapter
```

The factory receives `(args, device, dtype)` and returns an object with:

- `teacher_sample_latent(prompt, eps_latent, sigmas, guidance_scale) -> (x_teacher_latent, text_conditioning)`
- `load_student_xpred(init_checkpoint)`
- `student_forward_xpred(student, z, sigma, text_conditioning)`
- `save_student_xpred(student, checkpoint_path)`

The adapter keeps tensors in Anima VAE latent space and maps the same shifted `sigma` to the model timestep convention used by the original Anima code.

## Turbo LoRA teacher

The cache builder can use an explicit teacher LoRA:

```toml
[build_cache]
teacher_steps = 10
teacher_lora = "/root/shared-nvme/anima/anima-turbo-lora-v0.2.safetensors"
teacher_lora_weight = 1.0
```

This changes the teacher endpoint distribution. Treat cache generated this way as a turbo-teacher RUM dataset, not as a strict base Anima conversion. The cache metadata records `teacher_lora`, `teacher_lora_weight`, and `teacher_steps` for each sample.
