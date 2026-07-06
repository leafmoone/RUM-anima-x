from __future__ import annotations

import dataclasses
import math
import random
from pathlib import Path

import torch

from rum_xpred.anima import cache_file_sort_key, load_xpred_cache_sample, xpred_to_anima_v


DEFAULT_CACHE_BUCKETS: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (832, 1216),
    (1216, 832),
    (896, 1152),
    (1152, 896),
    (768, 1344),
    (1344, 768),
)


@dataclasses.dataclass(frozen=True)
class CacheBucket:
    name: str
    files: list[Path]


def read_prompts(path: str | Path, limit: int | None) -> list[str]:
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def chunked(items: list, batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def make_seeded_eps_batch(
    sample_indices: list[int],
    *,
    seed: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    latents = []
    for sample_index in sample_indices:
        generator = torch.Generator(device=device).manual_seed(seed + sample_index)
        latents.append(
            torch.randn(
                1,
                16,
                height // 8,
                width // 8,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        )
    return torch.cat(latents, dim=0)


def choose_cache_bucket(sample_index: int, seed: int) -> tuple[int, int]:
    rng = random.Random((int(seed) << 32) + int(sample_index))
    return DEFAULT_CACHE_BUCKETS[rng.randrange(len(DEFAULT_CACHE_BUCKETS))]


def bucket_name(width: int, height: int) -> str:
    return f"{int(width)}x{int(height)}"


def bucket_cache_path(cache_dir: Path, sample_index: int, *, width: int, height: int, bucket_enabled: bool) -> Path:
    filename = f"sample-{sample_index:06d}.safetensors"
    if not bucket_enabled:
        return cache_dir / filename
    return cache_dir / bucket_name(width, height) / filename


def collect_cache_buckets(cache_dir: str | Path) -> list[CacheBucket]:
    root = Path(cache_dir)
    buckets: list[CacheBucket] = []
    root_files = sorted(root.glob("*.safetensors"), key=cache_file_sort_key)
    if root_files:
        buckets.append(CacheBucket(name="root", files=root_files))
    for child in sorted(root.iterdir() if root.exists() else [], key=lambda path: path.name):
        if not child.is_dir():
            continue
        files = sorted(child.glob("*.safetensors"), key=cache_file_sort_key)
        if files:
            buckets.append(CacheBucket(name=child.name, files=files))
    return buckets


def merge_cache_samples(paths: list[Path], device: torch.device, dtype: torch.dtype) -> dict:
    samples = [load_xpred_cache_sample(path, device=device, dtype=dtype) for path in paths]
    shapes = {tuple(sample["x_teacher_latent"].shape[1:]) for sample in samples}
    if len(shapes) != 1:
        raise ValueError(f"mixed latent shapes in one train batch are not supported: {sorted(shapes)}")
    text_keys = set(samples[0]["text_conditioning"])
    if any(set(sample["text_conditioning"]) != text_keys for sample in samples):
        raise ValueError("all samples in a train batch must have the same text conditioning keys")
    return {
        "x_teacher_latent": torch.cat([sample["x_teacher_latent"] for sample in samples], dim=0),
        "eps_latent": torch.cat([sample["eps_latent"] for sample in samples], dim=0),
        "text_conditioning": {
            key: torch.cat([sample["text_conditioning"][key] for sample in samples], dim=0)
            for key in text_keys
        },
    }


def cache_source_name(cache_dir: str | Path) -> str:
    path = Path(cache_dir)
    name = path.parent.name if path.name.startswith("chunk-") and path.parent.name else path.name
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    return safe or "cache"


def unique_cache_source_names(cache_dirs: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    names: list[str] = []
    for cache_dir in cache_dirs:
        base = cache_source_name(cache_dir)
        count = counts.get(base, 0)
        counts[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count + 1}")
    return names


def cache_source_indices_for_paths(paths: list[Path], cache_dirs: list[str]) -> list[int]:
    roots = [Path(cache_dir).resolve() for cache_dir in cache_dirs]
    indices: list[int] = []
    for path in paths:
        resolved = path.resolve()
        for index, root in enumerate(roots):
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            indices.append(index)
            break
        else:
            raise ValueError(f"cache sample path does not belong to any cache_dir: {path}")
    return indices


def prediction_to_x(
    prediction_type: str,
    prediction: torch.Tensor,
    z: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    if prediction_type == "x":
        return prediction
    if prediction_type == "v":
        return z - sigma * prediction
    raise ValueError(f"unsupported prediction_type: {prediction_type!r}")


def reflow_loss(
    prediction_type: str,
    loss_weighting: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    z: torch.Tensor,
    sigma: torch.Tensor,
    eps_floor: float,
) -> torch.Tensor:
    if loss_weighting == "none":
        return torch.nn.functional.mse_loss(prediction.float(), target.float())
    if loss_weighting == "jlt_velocity_readout":
        if prediction_type != "x":
            raise ValueError("loss_weighting='jlt_velocity_readout' is only valid with prediction_type='x'")
        v_pred = xpred_to_anima_v(z.float(), prediction.float(), sigma.float(), eps_floor)
        v_target = xpred_to_anima_v(z.float(), target.float(), sigma.float(), eps_floor)
        return torch.nn.functional.mse_loss(v_pred, v_target)
    raise ValueError(f"unsupported loss_weighting: {loss_weighting!r}")


@torch.no_grad()
def loss_by_cache_source(
    *,
    prediction_type: str,
    loss_weighting: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    z: torch.Tensor,
    sigma: torch.Tensor,
    eps_floor: float,
    source_indices: list[int],
    source_count: int,
) -> dict[int, float]:
    losses: dict[int, float] = {}
    if not source_indices:
        return losses
    source_tensor = torch.tensor(source_indices, device=prediction.device)
    for source_index in range(source_count):
        mask = source_tensor == source_index
        if not bool(mask.any()):
            continue
        losses[source_index] = float(
            reflow_loss(
                prediction_type,
                loss_weighting,
                prediction.detach()[mask],
                target.detach()[mask],
                z.detach()[mask],
                sigma.detach()[mask],
                eps_floor,
            ).detach()
        )
    return losses


@torch.no_grad()
def x_mse_by_cache_source(
    *,
    x_pred: torch.Tensor,
    x_target: torch.Tensor,
    source_indices: list[int],
    source_count: int,
) -> dict[int, float]:
    losses: dict[int, float] = {}
    if not source_indices:
        return losses
    source_tensor = torch.tensor(source_indices, device=x_pred.device)
    for source_index in range(source_count):
        mask = source_tensor == source_index
        if not bool(mask.any()):
            continue
        losses[source_index] = float(torch.nn.functional.mse_loss(x_pred.detach()[mask].float(), x_target.detach()[mask].float()).detach())
    return losses


class CacheBatchCursor:
    def __init__(self, cache_files: list[Path], batch_size: int, *, shuffle: bool, seed: int, drop_last: bool) -> None:
        if batch_size < 1:
            raise ValueError("train_batch_size must be >= 1")
        self.cache_files = list(cache_files)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = random.Random(seed)
        self.position = 0
        if self.shuffle:
            self.rng.shuffle(self.cache_files)

    def next_count(self, count: int) -> list[Path]:
        if count < 1:
            return []
        selected = []
        while len(selected) < count:
            remaining = len(self.cache_files) - self.position
            need = count - len(selected)
            take = min(remaining, need)
            if take:
                selected.extend(self.cache_files[self.position : self.position + take])
                self.position += take
            if len(selected) == count:
                return selected
            self.position = 0
            if self.shuffle:
                self.rng.shuffle(self.cache_files)
            if self.drop_last and selected:
                selected = []
        return selected

    def next(self) -> list[Path]:
        return self.next_count(self.batch_size)


class CacheBucketBatchCursor:
    def __init__(self, buckets: list[CacheBucket], batch_size: int, *, shuffle: bool, seed: int, drop_last: bool) -> None:
        if not buckets:
            raise ValueError("at least one cache bucket is required")
        self.cursors = [
            CacheBatchCursor(bucket.files, batch_size, shuffle=shuffle, seed=seed + index, drop_last=drop_last)
            for index, bucket in enumerate(buckets)
        ]
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        self.position = 0
        self.order = list(range(len(self.cursors)))
        if self.shuffle:
            self.rng.shuffle(self.order)

    def next(self) -> list[Path]:
        if self.shuffle:
            index = self.rng.randrange(len(self.cursors))
            return self.cursors[index].next()
        index = self.order[self.position]
        self.position = (self.position + 1) % len(self.order)
        return self.cursors[index].next()


def allocate_weighted_counts(batch_size: int, weights: list[float]) -> list[int]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not weights:
        raise ValueError("at least one weight is required")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("cache mix weights must be non-negative and sum to > 0")
    total = float(sum(weights))
    raw = [batch_size * float(weight) / total for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remaining = batch_size - sum(counts)
    order = sorted(range(len(weights)), key=lambda index: raw[index] - counts[index], reverse=True)
    for index in order[:remaining]:
        counts[index] += 1
    return counts


class MultiCacheBucketBatchCursor:
    def __init__(
        self,
        cache_bucket_sets: list[list[CacheBucket]],
        batch_size: int,
        *,
        weights: list[float] | None,
        shuffle: bool,
        seed: int,
        drop_last: bool,
    ) -> None:
        if len(cache_bucket_sets) < 2:
            raise ValueError("multi cache cursor requires at least two cache dirs")
        if batch_size < 1:
            raise ValueError("train_batch_size must be >= 1")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        if weights is None:
            weights = [1.0] * len(cache_bucket_sets)
        if len(weights) != len(cache_bucket_sets):
            raise ValueError("cache_mix_weights length must match cache_dirs length")
        self.weights = [float(weight) for weight in weights]
        if any(weight < 0 for weight in self.weights) or sum(self.weights) <= 0:
            raise ValueError("cache_mix_weights must be non-negative and sum to > 0")

        bucket_names = sorted({bucket.name for buckets in cache_bucket_sets for bucket in buckets})
        self.bucket_groups: list[tuple[str, list[tuple[int, CacheBatchCursor]]]] = []
        for bucket_name in bucket_names:
            source_cursors: list[tuple[int, CacheBatchCursor]] = []
            for source_index, buckets in enumerate(cache_bucket_sets):
                bucket = next((candidate for candidate in buckets if candidate.name == bucket_name), None)
                if bucket is None:
                    continue
                source_cursors.append(
                    (
                        source_index,
                        CacheBatchCursor(
                            bucket.files,
                            batch_size,
                            shuffle=shuffle,
                            seed=seed + source_index * 1009 + len(self.bucket_groups),
                            drop_last=drop_last,
                        ),
                    )
                )
            if source_cursors:
                self.bucket_groups.append((bucket_name, source_cursors))
        if not self.bucket_groups:
            raise ValueError("no shared cache buckets found across cache dirs")
        self.position = 0
        self.order = list(range(len(self.bucket_groups)))
        if self.shuffle:
            self.rng.shuffle(self.order)

    def next(self) -> list[Path]:
        if self.shuffle:
            group_index = self.rng.randrange(len(self.bucket_groups))
        else:
            group_index = self.order[self.position]
            self.position = (self.position + 1) % len(self.order)
        _, source_cursors = self.bucket_groups[group_index]
        source_indices = [source_index for source_index, _ in source_cursors]
        weights = [self.weights[source_index] for source_index in source_indices]
        counts = allocate_weighted_counts(self.batch_size, weights)
        selected: list[Path] = []
        for count, (_, cursor) in zip(counts, source_cursors):
            selected.extend(cursor.next_count(count))
        if self.shuffle:
            self.rng.shuffle(selected)
        return selected


def prune_checkpoints(output_dir: Path, keep: int | None, *, prediction_type: str = "x") -> None:
    if keep is None or keep < 1:
        return
    checkpoints = sorted(output_dir.glob(f"{prediction_type}pred-checkpoint-step-*.safetensors"), key=cache_file_sort_key)
    for path in checkpoints[:-keep]:
        path.unlink()
