"""Optional weight audit for MoE true-on-policy debugging.

Writes per-version JSON summaries of weight tensor statistics for selected
representative parameters.  Gated behind the ``MILES_WEIGHT_AUDIT_ENABLE``
environment variable -- zero runtime cost when off.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import mpu

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    value = os.environ.get("MILES_WEIGHT_AUDIT_ENABLE", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _max_versions() -> int:
    value = os.environ.get("MILES_WEIGHT_AUDIT_MAX_VERSIONS", "3")
    try:
        return int(value)
    except ValueError:
        return 3


def _audit_dir() -> Path:
    return Path(os.environ.get("MILES_WEIGHT_AUDIT_DIR", "/tmp/miles_weight_audit"))


def _version_allowed(weight_version: int | str | None) -> bool:
    max_versions = _max_versions()
    if max_versions < 0 or weight_version is None:
        return True
    try:
        return int(weight_version) <= max_versions
    except (TypeError, ValueError):
        return True


def _family(name: str) -> str | None:
    if name in {
        "module.module.embedding.word_embeddings.weight",
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    }:
        return "embedding"
    if name in {"module.module.output_layer.weight", "lm_head.weight"}:
        return "output_layer"
    if name in {
        "module.module.decoder.final_layernorm.weight",
        "model.norm.weight",
        "model.language_model.norm.weight",
    }:
        return "final_layernorm"
    if ".mlp.router." in name or ".mlp.gate.weight" in name:
        return "moe_router"
    if ".mlp.experts." in name and (
        ".linear_fc1." in name or ".gate_proj." in name or ".up_proj." in name or ".gate_up_proj" in name
    ):
        return "moe_expert_fc1"
    if ".mlp.experts." in name and (".linear_fc2." in name or ".down_proj." in name):
        return "moe_expert_fc2"
    return None


def _selected_names(names: Sequence[str]) -> list[str]:
    by_family: dict[str, list[str]] = {}
    for name in names:
        fam = _family(name)
        if fam is not None:
            by_family.setdefault(fam, []).append(name)

    selected: list[str] = []
    for names_for_family in by_family.values():
        ordered = sorted(names_for_family)
        candidate_indices = {0, len(ordered) // 2, len(ordered) - 1}
        for index in sorted(candidate_indices):
            if 0 <= index < len(ordered):
                selected.append(ordered[index])
    return sorted(set(selected))


def _tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    with torch.no_grad():
        detached = tensor.detach()
        flat = detached.reshape(-1)
        numel = flat.numel()
        if numel == 0:
            sample = flat
        else:
            sample_size = min(4096, numel)
            midpoint = max(0, (numel - sample_size) // 2)
            sample = torch.cat(
                [
                    flat[:sample_size],
                    flat[midpoint : midpoint + sample_size],
                    flat[-sample_size:],
                ]
            )
        sample_float = sample.float()
        if sample_float.numel() == 0:
            sample_sum = 0.0
            sample_absmax = 0.0
            sample_first = None
            sample_last = None
        else:
            sample_sum = float(sample_float.sum().item())
            sample_absmax = float(sample_float.abs().max().item())
            sample_first = float(sample_float[0].item())
            sample_last = float(sample_float[-1].item())

        return {
            "shape": list(detached.shape),
            "stride": list(detached.stride()),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "numel": int(numel),
            "storage_offset": int(detached.storage_offset()),
            "sample_numel": int(sample_float.numel()),
            "sample_sum_fp32": sample_sum,
            "sample_absmax_fp32": sample_absmax,
            "sample_first_fp32": sample_first,
            "sample_last_fp32": sample_last,
        }


def write_weight_audit(
    *,
    stage: str,
    weight_version: int | str | None,
    tensors: Mapping[str, torch.Tensor] | Sequence[tuple[str, torch.Tensor]],
    chunk_index: int | None = None,
) -> None:
    if not _enabled() or not _version_allowed(weight_version):
        return

    items = list(tensors.items()) if isinstance(tensors, Mapping) else list(tensors)
    sel = _selected_names([name for name, _ in items])
    tensor_by_name = {name: tensor for name, tensor in items}
    selected = {name: _tensor_stats(tensor_by_name[name]) for name in sel if name in tensor_by_name}
    if not selected:
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    output_dir = _audit_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_suffix = "" if chunk_index is None else f"_chunk{chunk_index:04d}"
    version = "unknown" if weight_version is None else str(weight_version)
    output_path = output_dir / f"miles_{stage}_v{version}_rank{rank:03d}{chunk_suffix}.json"
    payload = {
        "stage": stage,
        "weight_version": version,
        "rank": rank,
        "selected": selected,
    }
    if dist.is_initialized():
        payload.update(
            {
                "world_size": dist.get_world_size(),
                "tp_rank": mpu.get_tensor_model_parallel_rank(),
                "pp_rank": mpu.get_pipeline_model_parallel_rank(),
                "cp_rank": mpu.get_context_parallel_rank(),
                "ep_rank": mpu.get_expert_model_parallel_rank(),
            }
        )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
