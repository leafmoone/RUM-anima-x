from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from rum_xpred.vendor_paths import ensure_local_sd_scripts


@dataclass
class AnimaSdScriptsAdapter:
    """RUM x-pred adapter backed by the vendored Anima sd-scripts code.

    This adapter deliberately exposes only the x-pred experiment surface. It is
    not a LoRA or ordinary full-finetune training entrypoint.
    """

    args: Any
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self) -> None:
        ensure_local_sd_scripts()
        import anima_minimal_inference as anima_inference  # type: ignore
        from library import anima_models  # type: ignore
        from library import strategy_anima, strategy_base  # type: ignore

        self.anima_inference = anima_inference
        self.anima_models = anima_models
        self.strategy_anima = strategy_anima
        self.strategy_base = strategy_base
        self.teacher = None
        self.student = None
        self.vae = None
        self.shared_text_models: dict[str, Any] = {}
        self.conds_cache: dict[str, Any] = {}

        if self.strategy_base.TokenizeStrategy.get_strategy() is None:
            tokenize_strategy = self.strategy_anima.AnimaTokenizeStrategy(
                qwen3_path=self.args.text_encoder,
                t5_tokenizer_path=None,
                qwen3_max_length=512,
                t5_max_length=512,
            )
            self.strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        if self.strategy_base.TextEncodingStrategy.get_strategy() is None:
            self.strategy_base.TextEncodingStrategy.set_strategy(self.strategy_anima.AnimaTextEncodingStrategy())

    def _convert_lora_for_sd_scripts(self, lora_path: str | Path) -> str:
        source = Path(lora_path)
        if not source.exists():
            raise FileNotFoundError(f"teacher_lora not found: {source}")
        sd = load_file(str(source), device="cpu")
        keys = list(sd)
        if any(key.startswith("lora_unet_") for key in keys):
            return str(source)
        if not any(key.startswith("diffusion_model.") and key.endswith((".lora_A.weight", ".lora_B.weight", ".alpha")) for key in keys):
            raise ValueError(
                "teacher_lora must use sd-scripts lora_unet_* keys or diffusion_model.*.lora_A/B keys; "
                f"got first key {keys[0] if keys else '<empty>'}"
            )

        repo_root = Path(__file__).resolve().parents[3]
        converted_dir = repo_root / "tmp"
        converted_dir.mkdir(parents=True, exist_ok=True)
        converted_path = converted_dir / f"{source.stem}.sd-scripts.safetensors"
        if converted_path.exists() and converted_path.stat().st_mtime >= source.stat().st_mtime:
            return str(converted_path)

        converted: dict[str, torch.Tensor] = {}
        for key, value in sd.items():
            if not key.startswith("diffusion_model."):
                continue
            body = key[len("diffusion_model.") :]
            if body.endswith(".lora_A.weight"):
                module = body[: -len(".lora_A.weight")].replace(".", "_")
                converted[f"lora_unet_{module}.lora_down.weight"] = value
            elif body.endswith(".lora_B.weight"):
                module = body[: -len(".lora_B.weight")].replace(".", "_")
                converted[f"lora_unet_{module}.lora_up.weight"] = value
            elif body.endswith(".alpha"):
                module = body[: -len(".alpha")].replace(".", "_")
                converted[f"lora_unet_{module}.alpha"] = value
        if not converted:
            raise ValueError(f"teacher_lora conversion produced no weights: {source}")
        save_file(
            converted,
            str(converted_path),
            metadata={"source": str(source), "format": "converted_for_sd_scripts_lora_utils"},
        )
        return str(converted_path)

    @classmethod
    def from_cli(cls, args: Any, device: torch.device | str, dtype: torch.dtype) -> "AnimaSdScriptsAdapter":
        return cls(args=args, device=torch.device(device), dtype=dtype)

    def _inference_args(
        self,
        *,
        prompt: str | list[str],
        seed: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        require_dit: bool = True,
    ):
        required = ["text_encoder"]
        if require_dit:
            required.append("dit")
        missing = [name for name in required if not getattr(self.args, name, None)]
        if missing:
            raise ValueError(f"real Anima adapter requires CLI argument(s): {', '.join('--' + name for name in missing)}")
        return SimpleNamespace(
            dit=self.args.dit,
            vae=getattr(self.args, "vae", None),
            text_encoder=self.args.text_encoder,
            lora_weight=[self._convert_lora_for_sd_scripts(self.args.teacher_lora)]
            if getattr(self.args, "teacher_lora", None)
            else None,
            lora_multiplier=[float(getattr(self.args, "teacher_lora_weight", 1.0))],
            include_patterns=None,
            exclude_patterns=None,
            guidance_scale=1.0 if cfg is None else cfg,
            prompt=prompt if isinstance(prompt, str) else prompt[0],
            negative_prompt=getattr(self.args, "negative_prompt", ""),
            image_size=[getattr(self.args, "height", 1024), getattr(self.args, "width", 1024)],
            infer_steps=steps or getattr(self.args, "teacher_steps", 40),
            save_path=str(getattr(self.args, "output_dir", "/tmp")),
            seed=seed,
            flow_shift=getattr(self.args, "flow_shift", 3.0),
            fp8=bool(getattr(self.args, "fp8", False)),
            fp8_scaled=bool(getattr(self.args, "fp8_scaled", False)),
            text_encoder_cpu=bool(getattr(self.args, "text_encoder_cpu", False)),
            device=str(self.device),
            attn_mode=getattr(self.args, "attn_mode", "torch"),
            output_type="latent",
            no_metadata=True,
            latent_path=None,
            lycoris=False,
        )

    def _load_teacher(self):
        if self.teacher is None:
            infer_args = self._inference_args(prompt="")
            self.teacher = self.anima_inference.load_dit_model(infer_args, self.device, self.dtype)
        return self.teacher

    def _encode_prompt(self, prompt: str, *, cfg: float, anima_model: torch.nn.Module | None = None):
        if anima_model is None:
            anima_model = self._load_teacher()
            require_dit = True
        else:
            require_dit = False
        infer_args = self._inference_args(prompt=prompt, cfg=cfg, require_dit=require_dit)
        if "conds_cache" not in self.shared_text_models:
            self.shared_text_models["conds_cache"] = self.conds_cache
        if "text_encoder" not in self.shared_text_models:
            text_encoder_device = torch.device("cpu") if infer_args.text_encoder_cpu else self.device
            self.shared_text_models["text_encoder"] = self.anima_inference.load_text_encoder(
                infer_args,
                dtype=torch.bfloat16,
                device=text_encoder_device,
            )
        return self.anima_inference.prepare_text_inputs(infer_args, self.device, anima_model, self.shared_text_models)

    def _ensure_text_encoder(self, *, text_encoder_cpu: bool):
        if "text_encoder" not in self.shared_text_models:
            infer_args = self._inference_args(prompt="", require_dit=False)
            infer_args.text_encoder_cpu = text_encoder_cpu
            text_encoder_device = torch.device("cpu") if text_encoder_cpu else self.device
            self.shared_text_models["text_encoder"] = self.anima_inference.load_text_encoder(
                infer_args,
                dtype=torch.bfloat16,
                device=text_encoder_device,
            )
        return self.shared_text_models["text_encoder"]

    @torch.no_grad()
    def _encode_prompt_batch(self, prompts: list[str], *, cfg: float, anima_model: torch.nn.Module):
        negative_prompt = getattr(self.args, "negative_prompt", "")
        cond_embeddings: list[torch.Tensor | None] = []
        missing_prompts: list[str] = []
        missing_positions: list[int] = []
        for position, prompt in enumerate(prompts):
            cached = self.conds_cache.get(prompt)
            if not isinstance(cached, torch.Tensor):
                cond_embeddings.append(None)
                missing_prompts.append(prompt)
                missing_positions.append(position)
            else:
                cond_embeddings.append(cached)

        text_encoder_cpu = bool(getattr(self.args, "text_encoder_cpu", False))
        text_encoder = self._ensure_text_encoder(text_encoder_cpu=text_encoder_cpu)
        text_encoder_device = torch.device("cpu") if text_encoder_cpu else self.device
        if getattr(text_encoder, "device", None) != text_encoder_device:
            text_encoder.to(text_encoder_device)

        tokenize_strategy = self.strategy_base.TokenizeStrategy.get_strategy()
        encoding_strategy = self.strategy_base.TextEncodingStrategy.get_strategy()
        if tokenize_strategy is None or encoding_strategy is None:
            raise RuntimeError("Anima text strategies are not initialized")

        if missing_prompts:
            tokens = tokenize_strategy.tokenize(missing_prompts)
            embed = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
            crossattn = anima_model._preprocess_text_embeds(
                source_hidden_states=embed[0].to(anima_model.device),
                target_input_ids=embed[2].to(anima_model.device),
                target_attention_mask=embed[3].to(anima_model.device),
                source_attention_mask=embed[1].to(anima_model.device),
            )
            crossattn[~embed[3].to(anima_model.device).bool()] = 0
            crossattn = crossattn.cpu()
            for row, prompt, position in zip(crossattn, missing_prompts, missing_positions):
                cached = row.unsqueeze(0).contiguous()
                self.conds_cache[prompt] = cached
                cond_embeddings[position] = cached

        cached_uncond = self.conds_cache.get(negative_prompt)
        if cached_uncond is None:
            tokens = tokenize_strategy.tokenize(negative_prompt)
            negative_embed = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
            negative_crossattn = anima_model._preprocess_text_embeds(
                source_hidden_states=negative_embed[0].to(anima_model.device),
                target_input_ids=negative_embed[2].to(anima_model.device),
                target_attention_mask=negative_embed[3].to(anima_model.device),
                source_attention_mask=negative_embed[1].to(anima_model.device),
            )
            negative_crossattn[~negative_embed[3].to(anima_model.device).bool()] = 0
            cached_uncond = negative_crossattn.cpu().contiguous()
            self.conds_cache[negative_prompt] = cached_uncond

        cond = torch.cat([embedding for embedding in cond_embeddings if embedding is not None], dim=0)
        uncond = cached_uncond.expand(len(prompts), *cached_uncond.shape[1:]).contiguous()
        return cond.to(self.device, dtype=torch.bfloat16), uncond.to(self.device, dtype=torch.bfloat16)

    def _encode_prompts(self, prompts: list[str], *, cfg: float, anima_model: torch.nn.Module | None = None):
        if anima_model is not None and len(prompts) > 1:
            return self._encode_prompt_batch(prompts, cfg=cfg, anima_model=anima_model)
        cond_embeds = []
        uncond_embeds = []
        for prompt in prompts:
            context, context_null = self._encode_prompt(prompt, cfg=cfg, anima_model=anima_model)
            cond_embeds.append(context["embed"][0].to(self.device, dtype=torch.bfloat16))
            uncond_embeds.append(context_null["embed"][0].to(self.device, dtype=torch.bfloat16))
        return torch.cat(cond_embeds, dim=0), torch.cat(uncond_embeds, dim=0)

    @torch.no_grad()
    def teacher_sample_latent(
        self,
        prompt: str | list[str],
        eps_latent: torch.Tensor,
        sigmas: torch.Tensor,
        guidance_scale: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the frozen Anima teacher from the provided latent noise to sigma=0."""
        teacher = self._load_teacher()
        prompts = [prompt] if isinstance(prompt, str) else prompt
        cond_embed, uncond_embed = self._encode_prompts(prompts, cfg=guidance_scale, anima_model=teacher)

        if eps_latent.ndim == 4:
            latents = eps_latent.unsqueeze(2)
        elif eps_latent.ndim == 5:
            latents = eps_latent
        else:
            raise ValueError(f"expected BCHW or BCFHW eps_latent, got {tuple(eps_latent.shape)}")
        latents = latents.to(self.device, dtype=torch.bfloat16)
        if cond_embed.shape[0] != latents.shape[0]:
            raise ValueError(f"prompt batch size {cond_embed.shape[0]} != latent batch size {latents.shape[0]}")

        padding_mask = torch.zeros(
            latents.shape[0],
            1,
            latents.shape[-2],
            latents.shape[-1],
            dtype=torch.bfloat16,
            device=self.device,
        )
        sigmas = sigmas.to(self.device, dtype=torch.float32)
        do_cfg = guidance_scale != 1.0
        for index, sigma in enumerate(sigmas[:-1]):
            sigma_next = sigmas[index + 1]
            timestep = sigma.reshape(1).expand(latents.shape[0]).to(self.device, dtype=torch.bfloat16)
            noise_pred = teacher(latents, timestep, cond_embed, padding_mask=padding_mask)
            if do_cfg:
                uncond = teacher(latents, timestep, uncond_embed, padding_mask=padding_mask)
                noise_pred = uncond + guidance_scale * (noise_pred - uncond)
            latents = latents + noise_pred * (sigma_next - sigma).to(latents.dtype)

        x_teacher = latents.squeeze(2).to(dtype=self.dtype)
        return x_teacher, {"prompt_embeds": cond_embed.detach().to(dtype=self.dtype)}

    def load_student_xpred(self, init_checkpoint: str | None):
        # The student architecture is the same Anima DiT; the experiment treats
        # its output as clean latent x rather than velocity.
        if init_checkpoint is None:
            init_checkpoint = getattr(self.args, "student_init", None) or getattr(self.args, "dit", None)
        if init_checkpoint is None:
            raise ValueError("load_student_xpred requires --student_init or --dit")
        original_dit = getattr(self.args, "dit", None)
        self.args.dit = init_checkpoint
        try:
            self.student = self._load_teacher() if self.teacher is None and init_checkpoint == original_dit else None
            if self.student is None:
                infer_args = self._inference_args(prompt="")
                self.student = self.anima_inference.load_dit_model(infer_args, self.device, self.dtype)
        finally:
            self.args.dit = original_dit
        self.student.train().requires_grad_(True)
        return self.student

    def _forward_with_conditioning(
        self,
        model: torch.nn.Module,
        z: torch.Tensor,
        sigma: torch.Tensor,
        text_conditioning: dict[str, torch.Tensor],
        *,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        if "prompt_embeds" not in text_conditioning:
            prompt = getattr(self.args, "prompt", "")
            context, context_null = self._encode_prompt(prompt, cfg=guidance_scale, anima_model=model)
            text_conditioning = {
                "prompt_embeds": context["embed"][0].detach().to(dtype=self.dtype),
                "negative_prompt_embeds": context_null["embed"][0].detach().to(dtype=self.dtype),
            }
        if z.ndim == 4:
            model_z = z.unsqueeze(2)
        else:
            model_z = z
        embed = text_conditioning["prompt_embeds"].to(self.device, dtype=torch.bfloat16)
        if embed.shape[0] == 1 and model_z.shape[0] > 1:
            embed = embed.expand(model_z.shape[0], *embed.shape[1:]).contiguous()
        if embed.shape[0] != model_z.shape[0]:
            raise ValueError(f"prompt embed batch size {embed.shape[0]} != latent batch size {model_z.shape[0]}")
        padding_mask = torch.zeros(
            model_z.shape[0],
            1,
            model_z.shape[-2],
            model_z.shape[-1],
            dtype=torch.bfloat16,
            device=self.device,
        )
        timestep = sigma.reshape(sigma.shape[0], -1)[:, 0].to(self.device, dtype=torch.bfloat16)
        model_z = model_z.to(self.device, dtype=torch.bfloat16)
        out = model(model_z, timestep, embed, padding_mask=padding_mask)
        if guidance_scale != 1.0:
            uncond_embed = text_conditioning.get("negative_prompt_embeds")
            if uncond_embed is None:
                _context, context_null = self._encode_prompt(getattr(self.args, "prompt", ""), cfg=guidance_scale, anima_model=model)
                uncond_embed = context_null["embed"][0].detach().to(dtype=self.dtype)
            uncond_embed = uncond_embed.to(self.device, dtype=torch.bfloat16)
            if uncond_embed.shape[0] == 1 and model_z.shape[0] > 1:
                uncond_embed = uncond_embed.expand(model_z.shape[0], *uncond_embed.shape[1:]).contiguous()
            if uncond_embed.shape[0] != model_z.shape[0]:
                raise ValueError(f"negative prompt embed batch size {uncond_embed.shape[0]} != latent batch size {model_z.shape[0]}")
            uncond = model(model_z, timestep, uncond_embed, padding_mask=padding_mask)
            out = uncond + guidance_scale * (out - uncond)
        return out.squeeze(2).to(dtype=z.dtype)

    def student_forward_xpred(
        self,
        student: torch.nn.Module,
        z: torch.Tensor,
        sigma: torch.Tensor,
        text_conditioning: dict[str, torch.Tensor],
        *,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        return self._forward_with_conditioning(student, z, sigma, text_conditioning, guidance_scale=guidance_scale)

    def teacher_forward_vpred(
        self,
        teacher: torch.nn.Module,
        z: torch.Tensor,
        sigma: torch.Tensor,
        text_conditioning: dict[str, torch.Tensor],
        *,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        return self._forward_with_conditioning(teacher, z, sigma, text_conditioning, guidance_scale=guidance_scale)

    def save_student_xpred(self, student: torch.nn.Module, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu().contiguous() for key, value in student.state_dict().items()}
        save_file(state, str(checkpoint_path))

    def _load_vae(self):
        if self.vae is None:
            vae_path = getattr(self.args, "vae", None)
            if not vae_path:
                raise ValueError("decode requires --vae / [common].vae")
            from library import qwen_image_autoencoder_kl  # type: ignore

            self.vae = qwen_image_autoencoder_kl.load_vae(
                vae_path,
                device="cpu",
                disable_mmap=True,
                spatial_chunk_size=getattr(self.args, "vae_spatial_chunk_size", None),
                disable_cache=bool(getattr(self.args, "vae_disable_cache", False)),
            )
            self.vae.to(torch.bfloat16)
            self.vae.eval()
        return self.vae

    @torch.no_grad()
    def decode_latents_to_images(self, latents: torch.Tensor, output_dir: str | Path, prefix: str = "sample") -> list[Path]:
        vae = self._load_vae()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        latents = latents.to(self.device, dtype=vae.dtype)
        vae.to(self.device)
        pixels = vae.decode_to_pixels(latents)
        if pixels.ndim == 5:
            pixels = pixels.squeeze(2)
        pixels = pixels.to("cpu", dtype=torch.float32)
        vae.to("cpu")

        paths: list[Path] = []
        for index, image_tensor in enumerate(pixels):
            image_tensor = torch.clamp(image_tensor, -1.0, 1.0)
            image_tensor = ((image_tensor + 1.0) * 127.5).to(torch.uint8)
            array = image_tensor.permute(1, 2, 0).contiguous().numpy()
            path = output_dir / f"{prefix}-{index:04d}.png"
            Image.fromarray(array).save(path)
            paths.append(path)
        return paths


def create_adapter(args: Any, device: torch.device | str, dtype: torch.dtype) -> AnimaSdScriptsAdapter:
    return AnimaSdScriptsAdapter.from_cli(args, device=device, dtype=dtype)
