from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


DEFAULT_TEACHER_STEPS = 40
DEFAULT_FLOW_SHIFT = 3.0
DEFAULT_TEACHER_CFG = 1.0
DEFAULT_SIGMA_MIN_TRAIN = 0.02
DEFAULT_EPS_FLOOR = 1e-4
DEFAULT_LEARNING_RATE = 1e-6


@dataclass(frozen=True)
class CacheMetadata:
    prompt: str
    width: int
    height: int
    seed: int
    sample_index: int
    teacher_steps: int
    flow_shift: float
    teacher_cfg: float
    teacher_lora: str | None = None
    teacher_lora_weight: float = 1.0


def apply_flow_shift(sigma: torch.Tensor, flow_shift: float) -> torch.Tensor:
    if flow_shift <= 0:
        raise ValueError("flow_shift must be > 0")
    return flow_shift * sigma / (1 + (flow_shift - 1) * sigma)


def make_shifted_sigma_schedule(steps: int, flow_shift: float, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    base = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    shifted = apply_flow_shift(base, flow_shift)
    shifted[0] = 1.0
    shifted[-1] = 0.0
    return shifted


def reflow_latent_z(x0_latent: torch.Tensor, eps_latent: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return (1 - sigma) * x0_latent + sigma * eps_latent


def reflow_training_target(prediction_type: str, x_teacher_latent: torch.Tensor, eps_latent: torch.Tensor) -> torch.Tensor:
    if prediction_type == "x":
        return x_teacher_latent
    if prediction_type == "v":
        return eps_latent - x_teacher_latent
    raise ValueError(f"prediction_type must be 'x' or 'v', got {prediction_type!r}")


def xpred_to_anima_v(z: torch.Tensor, x_pred_latent: torch.Tensor, sigma: torch.Tensor, eps_floor: float = DEFAULT_EPS_FLOOR) -> torch.Tensor:
    if eps_floor <= 0:
        raise ValueError("eps_floor must be > 0")
    return (z - x_pred_latent) / sigma.clamp_min(eps_floor)


def anima_euler_step(z: torch.Tensor, v_anima: torch.Tensor, sigma: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
    return z + v_anima * (sigma_next - sigma)


def sample_train_sigmas(
    batch_size: int,
    sigma_min_train: float,
    flow_shift: float,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not 0 < sigma_min_train < 1:
        raise ValueError("sigma_min_train must be in (0, 1)")
    u = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    base_sigma = sigma_min_train + (1 - sigma_min_train) * u
    shifted_sigma = apply_flow_shift(base_sigma, flow_shift)
    return shifted_sigma.view(batch_size, 1, 1, 1)


def validate_latent_pair(x_teacher_latent: torch.Tensor, eps_latent: torch.Tensor) -> None:
    if x_teacher_latent.shape != eps_latent.shape:
        raise ValueError(f"x_teacher_latent and eps_latent shapes differ: {x_teacher_latent.shape} != {eps_latent.shape}")
    if x_teacher_latent.ndim != 4:
        raise ValueError(f"expected BCHW latent tensors, got shape {tuple(x_teacher_latent.shape)}")
    if x_teacher_latent.shape[1] != 16:
        raise ValueError(f"expected Anima latent channel count 16, got {x_teacher_latent.shape[1]}")
    if not torch.isfinite(x_teacher_latent).all():
        raise ValueError("x_teacher_latent contains non-finite values")
    if not torch.isfinite(eps_latent).all():
        raise ValueError("eps_latent contains non-finite values")


def save_xpred_cache_sample(
    path: str | Path,
    x_teacher_latent: torch.Tensor,
    eps_latent: torch.Tensor,
    text_conditioning: dict[str, torch.Tensor],
    metadata: CacheMetadata,
) -> None:
    validate_latent_pair(x_teacher_latent, eps_latent)
    tensors: dict[str, torch.Tensor] = {
        "x_teacher_latent": x_teacher_latent.detach().cpu().contiguous(),
        "eps_latent": eps_latent.detach().cpu().contiguous(),
        "width": torch.tensor([metadata.width], dtype=torch.int32),
        "height": torch.tensor([metadata.height], dtype=torch.int32),
        "seed": torch.tensor([metadata.seed], dtype=torch.int64),
        "sample_index": torch.tensor([metadata.sample_index], dtype=torch.int64),
        "teacher_steps": torch.tensor([metadata.teacher_steps], dtype=torch.int32),
        "flow_shift": torch.tensor([metadata.flow_shift], dtype=torch.float32),
        "teacher_cfg": torch.tensor([metadata.teacher_cfg], dtype=torch.float32),
    }
    for key, value in text_conditioning.items():
        if not torch.is_tensor(value):
            raise TypeError(f"text_conditioning[{key!r}] must be a tensor")
        tensors[f"text_{key}"] = value.detach().cpu().contiguous()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_metadata = {
        "prompt": metadata.prompt,
        "format": "anima_rum_xpred_cache_v1",
        "teacher_lora": metadata.teacher_lora or "",
        "teacher_lora_weight": str(metadata.teacher_lora_weight),
    }
    save_file(tensors, str(path), metadata=file_metadata)


def load_xpred_cache_sample(path: str | Path, device: torch.device | str, dtype: torch.dtype) -> dict[str, Any]:
    tensors = load_file(str(path), device="cpu")
    from safetensors.torch import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        file_metadata = handle.metadata() or {}
    x_teacher_latent = tensors["x_teacher_latent"].to(device=device, dtype=dtype)
    eps_latent = tensors["eps_latent"].to(device=device, dtype=dtype)
    validate_latent_pair(x_teacher_latent, eps_latent)
    text_conditioning = {
        key.removeprefix("text_"): value.to(device=device, dtype=dtype)
        for key, value in tensors.items()
        if key.startswith("text_")
    }
    return {
        "x_teacher_latent": x_teacher_latent,
        "eps_latent": eps_latent,
        "text_conditioning": text_conditioning,
        "width": int(tensors["width"][0]),
        "height": int(tensors["height"][0]),
        "seed": int(tensors["seed"][0]),
        "sample_index": int(tensors["sample_index"][0]),
        "teacher_steps": int(tensors["teacher_steps"][0]),
        "flow_shift": float(tensors["flow_shift"][0]),
        "teacher_cfg": float(tensors["teacher_cfg"][0]),
        "teacher_lora": file_metadata.get("teacher_lora", "") or None,
        "teacher_lora_weight": float(file_metadata.get("teacher_lora_weight", "1.0")),
    }


def load_object(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("adapter spec must be 'module.submodule:factory_name'")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj = module
    for part in object_name.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{spec} resolved to a non-callable object")
    return obj


class ToyXPredNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(17, 32, kernel_size=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 16, kernel_size=1),
        )

    def forward(self, z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_map = sigma.expand(z.shape[0], 1, z.shape[2], z.shape[3]).to(dtype=z.dtype)
        return self.net(torch.cat([z, sigma_map], dim=1))


def make_toy_teacher_endpoint(eps_latent: torch.Tensor, prompt_index: int) -> torch.Tensor:
    phase = 0.05 * (prompt_index + 1)
    return torch.tanh(0.72 * eps_latent + phase)


def train_one_xpred_step(
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x_teacher_latent: torch.Tensor,
    eps_latent: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    z = reflow_latent_z(x_teacher_latent, eps_latent, sigma)
    x_pred = student(z, sigma)
    loss = F.mse_loss(x_pred.float(), x_teacher_latent.float())
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite x-pred loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for parameter in student.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("non-finite x-pred gradient")
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def sample_with_xpred_student(
    student_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eps_latent: torch.Tensor,
    sigmas: torch.Tensor,
    eps_floor: float = DEFAULT_EPS_FLOOR,
) -> torch.Tensor:
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be a 1D schedule")
    if sigmas[-1].item() != 0.0:
        raise ValueError("sigmas must end at 0")
    z = eps_latent
    for index, sigma_value in enumerate(sigmas[:-1]):
        sigma = sigma_value.reshape(1, 1, 1, 1).to(device=z.device, dtype=z.dtype).expand(z.shape[0], 1, 1, 1)
        sigma_next = sigmas[index + 1].reshape(1, 1, 1, 1).to(device=z.device, dtype=z.dtype)
        x_pred = student_forward(z, sigma)
        v = xpred_to_anima_v(z, x_pred, sigma, eps_floor)
        z = anima_euler_step(z, v, sigma, sigma_next)
        if not torch.isfinite(z).all():
            raise FloatingPointError("non-finite latent during x-pred sampling")
    return z


@torch.no_grad()
def sample_with_vpred_student(
    student_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eps_latent: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be a 1D schedule")
    if sigmas[-1].item() != 0.0:
        raise ValueError("sigmas must end at 0")
    z = eps_latent
    for index, sigma_value in enumerate(sigmas[:-1]):
        sigma = sigma_value.reshape(1, 1, 1, 1).to(device=z.device, dtype=z.dtype).expand(z.shape[0], 1, 1, 1)
        sigma_next = sigmas[index + 1].reshape(1, 1, 1, 1).to(device=z.device, dtype=z.dtype)
        v_pred = student_forward(z, sigma)
        z = anima_euler_step(z, v_pred, sigma, sigma_next)
        if not torch.isfinite(z).all():
            raise FloatingPointError("non-finite latent during v-pred sampling")
    return z


def cache_file_sort_key(path: Path) -> tuple[int, str]:
    stem_digits = "".join(ch for ch in path.stem if ch.isdigit())
    return (int(stem_digits) if stem_digits else math.inf, path.name)
