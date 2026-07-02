# Project Layout

This repository is a standalone Anima latent x-pred RUM conversion experiment.

## Local Experiment Code

- `src/rum_xpred/anima.py` - formula helpers, shifted sigma schedule, x-pred cache format, toy model, and dedicated x-pred sampler.
- `src/rum_xpred/adapters/anima_sd_scripts.py` - real Anima adapter backed by local vendored sd-scripts code.
- `src/rum_xpred/vendor_paths.py` - local vendor path resolver.
- `src/rum_xpred/config.py` - TOML config loading and validation.
- `scripts/dev/anima_rum_xpred_train.py` - single CLI for `build_cache`, `train_xpred`, `sample_xpred`, and `chunked_rum`.
- `configs/anima_xpred.example.toml` - commented config covering all stages and runtime options.
- `tests/test_anima_rum_xpred.py` - lightweight formula/cache/sampler tests.

## Vendored Anima Code

- `vendor/sd-scripts/` - local copy of the Anima/kohya implementation needed for model loading, text encoding, and DiT forward/sampling support.

This project does not depend on `/root/shared-nvme/lora-scripts-next` at runtime.

## Data

- `data/prompts/sample_prompts.txt` - tiny prompt file for toy smoke checks.

Large prompt lists and generated caches are intentionally not included. Put explicit paths in the config when running real experiments.

## Removed Scope

Ordinary Anima LoRA/full-finetune training wrappers are intentionally not part of this project. This repository only targets:

```text
Anima FM teacher -> cached x_teacher_latent -> RUM reflow x-pred student
```
