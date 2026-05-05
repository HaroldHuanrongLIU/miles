"""Optional gradient audit for MoE true-on-policy debugging.

Writes per-step JSON summaries of gradient statistics grouped by parameter
family (moe_router, moe_expert, attention, ...).  Gated behind the
``MILES_GRAD_AUDIT_ENABLE`` environment variable -- zero runtime cost when off.
"""

from __future__ import annotations

import json
import logging
import os
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def _family(name: str) -> str:
    if ".mlp.router." in name or ".router." in name:
        return "moe_router"
    if ".mlp.experts." in name or ".experts." in name:
        return "moe_expert"
    if ".self_attention." in name or ".attention." in name:
        return "attention"
    if "layernorm" in name or "layer_norm" in name:
        return "layernorm"
    if "embedding" in name:
        return "embedding"
    if "output_layer" in name:
        return "output_layer"
    return "other"


def _should_run() -> bool:
    value = os.environ.get("MILES_GRAD_AUDIT_ENABLE", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _max_steps() -> int:
    value = os.environ.get("MILES_GRAD_AUDIT_MAX_STEPS", "3")
    try:
        return int(value)
    except ValueError:
        return 3


def _output_dir(args: Namespace) -> Path:
    return Path(os.environ.get("MILES_GRAD_AUDIT_DIR") or Path(args.save).parent / "grad_audit")


def write_grad_audit(args: Namespace, rollout_id: int, step_id: int, model: Sequence[DDP]) -> None:
    if not _should_run():
        return

    max_steps = _max_steps()
    steps_per_rollout = getattr(args, "num_steps_per_rollout", None) or 1
    accumulated_step_id = rollout_id * max(1, int(steps_per_rollout)) + step_id
    if max_steps >= 0 and accumulated_step_id >= max_steps:
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    summary: dict[str, object] = {
        "rollout_id": rollout_id,
        "step_id": step_id,
        "accumulated_step_id": accumulated_step_id,
        "rank": rank,
        "tp_rank": mpu.get_tensor_model_parallel_rank(),
        "pp_rank": mpu.get_pipeline_model_parallel_rank(),
        "cp_rank": mpu.get_context_parallel_rank(),
        "ep_rank": mpu.get_expert_model_parallel_rank(),
        "families": {},
        "missing_grad_names": [],
        "zero_grad_names": [],
        "top_nonzero_grad_names": [],
    }
    families: dict[str, dict[str, float | int]] = summary["families"]  # type: ignore[assignment]
    missing_grad_names: list[str] = summary["missing_grad_names"]  # type: ignore[assignment]
    zero_grad_names: list[str] = summary["zero_grad_names"]  # type: ignore[assignment]
    top_nonzero: list[tuple[float, str]] = []

    total_params = 0
    with_grad = 0
    missing_grad = 0
    zero_grad = 0
    nonzero_grad = 0

    for chunk_idx, model_chunk in enumerate(model):
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            total_params += 1
            full_name = f"chunk{chunk_idx}.{name}"
            family_name = _family(name)
            family_stats = families.setdefault(
                family_name,
                {
                    "total": 0,
                    "with_grad": 0,
                    "missing_grad": 0,
                    "zero_grad": 0,
                    "nonzero_grad": 0,
                    "max_abs_grad": 0.0,
                },
            )
            family_stats["total"] += 1

            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is None:
                missing_grad += 1
                family_stats["missing_grad"] += 1
                if len(missing_grad_names) < 50:
                    missing_grad_names.append(full_name)
                continue

            with_grad += 1
            family_stats["with_grad"] += 1
            grad_max = float(grad.detach().abs().max().item())
            family_stats["max_abs_grad"] = max(float(family_stats["max_abs_grad"]), grad_max)
            if grad_max == 0.0:
                zero_grad += 1
                family_stats["zero_grad"] += 1
                if len(zero_grad_names) < 50:
                    zero_grad_names.append(full_name)
            else:
                nonzero_grad += 1
                family_stats["nonzero_grad"] += 1
                top_nonzero.append((grad_max, full_name))

    summary["total_params"] = total_params
    summary["with_grad"] = with_grad
    summary["missing_grad"] = missing_grad
    summary["zero_grad"] = zero_grad
    summary["nonzero_grad"] = nonzero_grad
    summary["top_nonzero_grad_names"] = [
        {"name": name, "max_abs_grad": value} for value, name in sorted(top_nonzero, reverse=True)[:20]
    ]

    out = _output_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / (
        f"grad_audit_step{accumulated_step_id:04d}_"
        f"rank{rank:03d}_tp{summary['tp_rank']}_pp{summary['pp_rank']}_"
        f"cp{summary['cp_rank']}_ep{summary['ep_rank']}.json"
    )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    logger.info(
        "[GradAudit] wrote %s total=%d with_grad=%d missing=%d nonzero=%d zero=%d families=%s",
        output_path,
        total_params,
        with_grad,
        missing_grad,
        nonzero_grad,
        zero_grad,
        summary["families"],
    )
