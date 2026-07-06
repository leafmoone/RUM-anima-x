# ComfyUI RUM Anima XPred

Custom nodes for sampling RUM Anima x-pred checkpoints in ComfyUI.

## Nodes

- `Load Anima XPred Model`
  - Loads the Anima DiT x-pred checkpoint, text encoder, and VAE paths.
  - Use `xpred-adapter-checkpoint.safetensors`, not `xpred-train-state.pt`.

- `Sample Anima XPred`
  - Runs the dedicated x-pred sampler:
    `v = (z - x_pred) / sigma`
  - Decodes the final latent with the Anima/Qwen VAE.
  - Outputs a ComfyUI `IMAGE` plus the raw latent dictionary.

## Minimal Workflow

Connect:

```text
Load Anima XPred Model -> Sample Anima XPred -> Preview Image / Save Image
```

Do not connect an x-pred checkpoint to a regular Anima/FM velocity sampler. The checkpoint predicts clean latent `x`, not velocity.
