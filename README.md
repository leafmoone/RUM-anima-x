# RUM Anima X-Pred

Standalone project for converting an Anima Rectified Flow DiT from latent velocity prediction to clean latent `x` prediction with teacher-endpoint reflow.

This project is only for the x-pred conversion experiment. It does not keep ordinary Anima LoRA or full-finetune training entrypoints.

## Mathematical Contract

It does not bridge between different VAE latent spaces, and it does not do `teacher latent -> image -> student latent` conversion. The Anima teacher and student live in the same Anima latent space, so the teacher's final latent can be used directly as the student target.

The experiment keeps Anima's Rectified Flow sigma convention. In the equations below, `sigma` means the shifted sigma value actually used by the model, sampler, cache metadata, and training loss:

```text
z_sigma = (1 - sigma) * x0 + sigma * eps
v_anima = dz/dsigma = eps - x0
```

Sampling runs from `sigma=1` to `sigma=0`, so an Euler step is:

```text
z_next = z + v_anima * (sigma_next - sigma)
```

The original Anima model predicts velocity. A direct local conversion from a velocity teacher to an x-pred student would query the teacher at each training point:

```text
v_teacher = teacher_v(z, sigma, cond)
x_target = z - sigma * v_teacher
loss = MSE(student_x(z, sigma, cond), x_target)
```

That would preserve the teacher's local vector field. This project intentionally does something different: it first asks the frozen FM teacher where a sampled noise latent ends up, then trains a new straight reflow path to that endpoint.

For each prompt and seed:

```text
eps, prompt --Anima FM teacher sampler--> x_teacher
z = (1 - sigma) * x_teacher + sigma * eps
x_pred = student_x(z, sigma, cond)
```

With the default clean x-pred loss:

```text
loss = MSE(x_pred, x_teacher)
```

With the JLT-style velocity-readout loss:

```text
v_pred = (z - x_pred) / max(sigma, loss_eps_floor)
v_target = (z - x_teacher) / max(sigma, loss_eps_floor)
loss = MSE(v_pred, v_target)
```

Since `z = (1 - sigma) * x_teacher + sigma * eps`, the readout target is:

```text
v_target = eps - x_teacher
```

So the model output is still clean latent `x`; the JLT loss only changes where the error is measured. It weights x-pred errors approximately by `1 / sigma^2`, which makes the x-pred student behave more like a velocity model after readout.

In short, this is:

```text
teacher endpoint reflow + x-pred parameterization
```

It is not:

```text
local teacher vector-field conversion
```



## Why Train This Way

The motivation inherited from RUM/reflow is to avoid using an arbitrary dataset image as the endpoint for a random noise sample. Instead, the trained teacher defines the endpoint that this prompt/noise pair should reach. This gives a consistent pair:

```text
(eps, x_teacher)
```

and trains the student on the straight path between them.

For this project, that has two intended benefits:

- It converts an existing Anima FM teacher into an x-pred sampler without requiring original image data.
- It reflows the teacher's generated endpoint distribution into a simpler straight-line training problem.

The tradeoff is that the teacher only supervises the endpoint. The intermediate training targets come from the straight-line reflow assumption, not from querying the teacher's true local velocity at each `z, sigma`.

Advantages:

- Same latent space: no VAE decode/encode bridge, no cross-space target mismatch.
- Cacheable endpoints: teacher sampling can be done once and reused.
- x-pred output: sampling remains x-pred; `v = (z - x_pred) / sigma` is only the Euler update readout.
- Reflow target: avoids random dataset endpoint ambiguity for the same prompt/noise pair.

Disadvantages:

- It does not preserve the teacher's original local ODE/vector field.
- A bad or biased teacher endpoint becomes the new ground truth.
- Very small `sigma` can make velocity-readout loss numerically aggressive, so `loss_eps_floor` is required.
- It is not a complete RUM reproduction, because there is no cross-architecture latent-space bridge.

## Layout

- `src/rum_xpred/` - x-pred formula, cache, sampler, and local Anima adapter.
- `scripts/dev/anima_rum_xpred_train.py` - experiment CLI.
- `scripts/dev/upload_completed_cache_chunks.py` - project-local cache packaging/upload helper.
- `configs/anima_xpred.example.toml` - commented config covering cache, train, and sample.
- `configs/anima_vpred_reflow.example.toml` - optional velocity reflow control config using the same cache format.
- `vendor/sd-scripts/` - copied local Anima/kohya code used by the adapter.
- `docs/` - formula and project notes.
- `tests/` - lightweight tests.
- `data/prompts/sample_prompts.txt` - small prompt file for smoke tests.

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



It writes chunk caches under `[chunked_rum].chunk_root/cache/`, chunk checkpoints and optimizer states under `[chunked_rum].chunk_root/train/`, and resume state to `chunk-manifest.json`. Each next chunk inherits both the previous student checkpoint and AdamW optimizer state. This is sequential on one GPU; it does not run teacher cache generation in parallel with student training.

## Acknowledgements

This project is built on several lines of prior work and open-source infrastructure.

- Rectified Flow / reflow: for the straight-path endpoint reflow formulation used to train from teacher-generated endpoints.
- JLT / JiT-style clean prediction: for motivating clean-latent `x` prediction and the velocity-readout loss used to measure x-pred errors
at the sampler's update scale.
- RUM: for the practical teacher-endpoint distillation idea that inspired this experiment, especially replacing arbitrary dataset endpoints
with endpoints generated by a trained teacher. This project does not reproduce RUM's cross-architecture latent-space bridge.
- Anima and the local `sd-scripts` Anima implementation: for the pretrained Flow Matching teacher, model architecture, text-conditioning
path, and latent sampler interface used by this experiment.
- PyTorch, safetensors, and the Hugging Face ecosystem: for the training and model-serialization tooling used throughout the project.
