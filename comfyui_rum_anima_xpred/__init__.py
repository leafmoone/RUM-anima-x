from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rum_xpred.adapters.anima_sd_scripts import create_adapter
from rum_xpred.anima import make_shifted_sigma_schedule, sample_with_xpred_student, sample_with_vpred_student
from rum_xpred.cache_batches import make_seeded_eps_batch


DEFAULT_CHECKPOINT = str(REPO_ROOT / "anima-jlt-xpred-turbo10-chunks/train/chunk-0047/xpred-adapter-checkpoint.safetensors")
DEFAULT_DIT = "/root/shared-nvme/anima/split_files/diffusion_models/anima-base-v1.0.safetensors"
DEFAULT_TEXT_ENCODER = "/root/shared-nvme/anima/split_files/text_encoders/qwen_3_06b_base.safetensors"
DEFAULT_VAE = "/root/shared-nvme/anima/split_files/vae/qwen_image_vae.safetensors"


def _comfy_device() -> torch.device:
    try:
        import comfy.model_management as model_management

        return torch.device(model_management.get_torch_device())
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _comfy_intermediate_device() -> torch.device:
    try:
        import comfy.model_management as model_management

        return torch.device(model_management.intermediate_device())
    except Exception:
        return torch.device("cpu")


def _soft_empty_cache() -> None:
    gc.collect()
    try:
        import comfy.model_management as model_management

        model_management.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported precision: {name!r}")


@dataclass
class LoadedAnimaXPred:
    args: Any
    adapter: Any
    student: torch.nn.Module
    device: torch.device
    dtype: torch.dtype
    prediction_type: str


class AnimaXPredModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": ("STRING", {"default": DEFAULT_CHECKPOINT}),
                "text_encoder": ("STRING", {"default": DEFAULT_TEXT_ENCODER}),
                "vae": ("STRING", {"default": DEFAULT_VAE}),
                "base_dit": ("STRING", {"default": DEFAULT_DIT}),
                "prediction_type": (["x", "v"], {"default": "x"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "attn_mode": (["torch", "flash", "sageattn", "xformers"], {"default": "flash"}),
                "text_encoder_cpu": ("BOOLEAN", {"default": False}),
                "fp8": ("BOOLEAN", {"default": False}),
                "fp8_scaled": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("RUM_ANIMA_XPRED",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "RUM/Anima XPred"

    def load_model(
        self,
        checkpoint: str,
        text_encoder: str,
        vae: str,
        base_dit: str,
        prediction_type: str,
        precision: str,
        attn_mode: str,
        text_encoder_cpu: bool,
        fp8: bool,
        fp8_scaled: bool,
    ):
        checkpoint_path = Path(checkpoint).expanduser()
        text_encoder_path = Path(text_encoder).expanduser()
        vae_path = Path(vae).expanduser()
        base_dit_path = Path(base_dit).expanduser()
        for label, path in {
            "checkpoint": checkpoint_path,
            "text_encoder": text_encoder_path,
            "vae": vae_path,
            "base_dit": base_dit_path,
        }.items():
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        device = _comfy_device()
        dtype = _dtype_from_name(precision)
        args = SimpleNamespace(
            dit=str(base_dit_path),
            student_init=str(checkpoint_path),
            text_encoder=str(text_encoder_path),
            vae=str(vae_path),
            output_dir=str(REPO_ROOT / "outputs/comfyui-rum-anima-xpred"),
            negative_prompt="",
            flow_shift=3.0,
            teacher_steps=40,
            attn_mode=attn_mode,
            fp8=fp8,
            fp8_scaled=fp8_scaled,
            text_encoder_cpu=text_encoder_cpu,
            teacher_lora=None,
            teacher_lora_weight=1.0,
            vae_spatial_chunk_size=None,
            vae_disable_cache=False,
        )
        adapter = create_adapter(args, device=device, dtype=dtype)
        student = adapter.load_student_xpred(init_checkpoint=str(checkpoint_path))
        student.to(device=device, dtype=dtype).eval().requires_grad_(False)
        return (LoadedAnimaXPred(args=args, adapter=adapter, student=student, device=device, dtype=dtype, prediction_type=prediction_type),)


class AnimaXPredSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("RUM_ANIMA_XPRED",),
                "prompt": ("STRING", {"multiline": True, "default": "hatsune miku, 1girl, masterpiece, best quality"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 20260701, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "eps_floor": ("FLOAT", {"default": 1e-4, "min": 1e-8, "max": 1.0, "step": 1e-5}),
                "offload_after_sample": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "RUM_ANIMA_LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "sample"
    CATEGORY = "RUM/Anima XPred"

    def sample(
        self,
        model: LoadedAnimaXPred,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        width: int,
        height: int,
        batch_size: int,
        flow_shift: float,
        eps_floor: float,
        offload_after_sample: bool,
    ):
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("width and height must be divisible by 8")

        model.args.prompt = prompt
        model.args.negative_prompt = negative_prompt
        model.args.width = width
        model.args.height = height
        model.args.flow_shift = flow_shift
        model.student.to(device=model.device, dtype=model.dtype).eval()

        with torch.no_grad():
            sigmas = make_shifted_sigma_schedule(steps, flow_shift, device=model.device, dtype=model.dtype)
            eps_latent = make_seeded_eps_batch(
                list(range(batch_size)),
                seed=int(seed),
                height=height,
                width=width,
                device=model.device,
                dtype=model.dtype,
            )
            cond_embed, uncond_embed = model.adapter._encode_prompts([prompt] * batch_size, cfg=cfg, anima_model=model.student)
            text_conditioning = {
                "prompt_embeds": cond_embed.detach(),
                "negative_prompt_embeds": uncond_embed.detach(),
            }

            def student_forward(z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                return model.adapter.student_forward_xpred(
                    model.student,
                    z,
                    sigma,
                    text_conditioning,
                    guidance_scale=cfg,
                )

            if model.prediction_type == "x":
                latent = sample_with_xpred_student(student_forward, eps_latent, sigmas, eps_floor=eps_floor)
            else:
                latent = sample_with_vpred_student(student_forward, eps_latent, sigmas)

            vae = model.adapter._load_vae()
            vae.to(model.device)
            pixels = vae.decode_to_pixels(latent.to(model.device, dtype=vae.dtype))
            if pixels.ndim == 5:
                pixels = pixels.squeeze(2)
            images = ((pixels.clamp(-1.0, 1.0) + 1.0) * 0.5).to(torch.float32)
            images = images.movedim(1, -1).to(device=_comfy_intermediate_device())
            latent_out = {
                "samples": latent.detach().to(device=_comfy_intermediate_device(), dtype=torch.float32),
                "sigmas": sigmas.detach().to(device="cpu", dtype=torch.float32),
            }

        if offload_after_sample:
            model.student.to("cpu")
            if model.adapter.vae is not None:
                model.adapter.vae.to("cpu")
            _soft_empty_cache()

        return (images, latent_out)


NODE_CLASS_MAPPINGS = {
    "AnimaXPredModelLoader": AnimaXPredModelLoader,
    "AnimaXPredSampler": AnimaXPredSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaXPredModelLoader": "Load Anima XPred Model",
    "AnimaXPredSampler": "Sample Anima XPred",
}
