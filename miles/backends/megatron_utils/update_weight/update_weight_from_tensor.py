import logging
import json
import os
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from miles.backends.megatron_utils.lora_utils import LORA_ADAPTER_NAME, build_lora_sync_config, is_lora_weight_name
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .common import post_process_weights
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed.broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


def _weight_audit_enabled() -> bool:
    value = os.environ.get("MILES_WEIGHT_AUDIT_ENABLE", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _weight_audit_max_versions() -> int:
    value = os.environ.get("MILES_WEIGHT_AUDIT_MAX_VERSIONS", "3")
    try:
        return int(value)
    except ValueError:
        return 3


def _weight_audit_dir() -> Path:
    return Path(os.environ.get("MILES_WEIGHT_AUDIT_DIR", "/tmp/miles_weight_audit"))


def _weight_audit_version_allowed(weight_version: int | str | None) -> bool:
    max_versions = _weight_audit_max_versions()
    if max_versions < 0 or weight_version is None:
        return True
    try:
        return int(weight_version) <= max_versions
    except (TypeError, ValueError):
        return True


def _weight_audit_family(name: str) -> str | None:
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
        ".linear_fc1." in name
        or ".gate_proj." in name
        or ".up_proj." in name
        or ".gate_up_proj" in name
    ):
        return "moe_expert_fc1"
    if ".mlp.experts." in name and (
        ".linear_fc2." in name
        or ".down_proj." in name
    ):
        return "moe_expert_fc2"
    return None


def _weight_audit_selected_names(names: Sequence[str]) -> list[str]:
    by_family: dict[str, list[str]] = {}
    for name in names:
        family = _weight_audit_family(name)
        if family is not None:
            by_family.setdefault(family, []).append(name)

    selected: list[str] = []
    for names_for_family in by_family.values():
        ordered = sorted(names_for_family)
        candidate_indices = {0, len(ordered) // 2, len(ordered) - 1}
        for index in sorted(candidate_indices):
            if 0 <= index < len(ordered):
                selected.append(ordered[index])
    return sorted(set(selected))


def _weight_audit_tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
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


def _write_weight_audit(
    *,
    stage: str,
    weight_version: int | str | None,
    tensors: Mapping[str, torch.Tensor] | Sequence[tuple[str, torch.Tensor]],
    chunk_index: int | None = None,
) -> None:
    if not _weight_audit_enabled() or not _weight_audit_version_allowed(weight_version):
        return

    items = list(tensors.items()) if isinstance(tensors, Mapping) else list(tensors)
    selected_names = _weight_audit_selected_names([name for name, _ in items])
    tensor_by_name = {name: tensor for name, tensor in items}
    selected = {
        name: _weight_audit_tensor_stats(tensor_by_name[name])
        for name in selected_names
        if name in tensor_by_name
    }
    if not selected:
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    output_dir = _weight_audit_dir()
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


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict->GPU) -> broadcast PP/EP(GPU NCCL) -> gather TP(GPU NCCL) -> convert HF(GPU) -> send.
    Colocated: GPU->CPU serialize -> gather_object(Gloo CPU) -> Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        """
        Compute param buckets, create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.is_lora = is_lora
        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
            is_lora=self.is_lora,
        )
        if self.is_lora:
            self._lora_config = build_lora_sync_config(args)
            self._lora_loaded = False
            self._lora_base_synced = False

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None

        self._model_update_groups = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                get_parallel_state().intra_dp_cp.rank == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "miles"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]
        colocate_ipc_group_ranks = self._get_colocated_ipc_group_ranks(
            colocate_engine_nums,
            colocate_gpu_offsets,
            colocate_gpu_counts,
        )

        # Determine whether this rank is covered by any colocated engine.
        all_colocated_ranks = {
            rank for group_ranks in colocate_ipc_group_ranks for rank in group_ranks
        }
        rank_has_engine = dist.get_rank() in all_colocated_ranks

        # Create IPC Gloo gather groups matching actual engine layout.
        # Re-create on connect because MoE EP rollout groups are not necessarily
        # contiguous global ranks once CP is present.
        self._ipc_gather_group = None
        self._ipc_gather_src = None
        for group_ranks in colocate_ipc_group_ranks:
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if rank_has_engine and dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = group_ranks[0]

        # Map training ranks to colocated engine actors.
        self._ipc_engine = None
        for engine, group_ranks in zip(
            self.rollout_engines[:colocate_engine_nums],
            colocate_ipc_group_ranks,
            strict=True,
        ):
            if dist.get_rank() in group_ranks:
                self._ipc_engine = engine

    def _get_colocated_ipc_group_ranks(
        self,
        colocate_engine_nums: int,
        colocate_gpu_offsets: Sequence[int],
        colocate_gpu_counts: Sequence[int],
    ) -> list[list[int]]:
        if colocate_engine_nums == 0:
            return []

        rollout_ep_size = (
            getattr(self.args, "sglang_expert_parallel_size", None)
            or getattr(self.args, "sglang_ep_size", 1)
            or 1
        )
        if rollout_ep_size <= 1:
            return [
                list(range(offset, offset + count))
                for offset, count in zip(
                    colocate_gpu_offsets,
                    colocate_gpu_counts,
                    strict=True,
                )
            ]

        train_ep_size = mpu.get_expert_model_parallel_world_size()
        if train_ep_size != rollout_ep_size:
            raise NotImplementedError(
                "Colocated SGLang EP weight sync currently requires rollout EP "
                f"to match Megatron EP ({rollout_ep_size} != {train_ep_size})."
            )
        if any(count != rollout_ep_size for count in colocate_gpu_counts):
            raise NotImplementedError(
                "Colocated SGLang EP weight sync currently requires each rollout "
                f"engine to contain exactly one EP group of size {rollout_ep_size}; "
                f"got engine sizes {list(colocate_gpu_counts)}."
            )

        local_ep_group = tuple(
            dist.get_process_group_ranks(mpu.get_expert_model_parallel_group())
        )
        all_ep_groups: list[tuple[int, ...] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(
            all_ep_groups,
            local_ep_group,
            group=get_gloo_group(),
        )
        unique_ep_groups = sorted({group for group in all_ep_groups if group is not None})
        if len(unique_ep_groups) < colocate_engine_nums:
            raise RuntimeError(
                "Not enough Megatron EP groups to map colocated SGLang EP engines: "
                f"need {colocate_engine_nums}, got {len(unique_ep_groups)}."
            )
        return [list(group) for group in unique_ep_groups[:colocate_engine_nums]]

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    rollout_engines=self.rollout_engines,
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                )
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()
        _write_weight_audit(
            stage="megatron_local",
            weight_version=self.weight_version,
            tensors=megatron_local_weights,
        )

        # For LoRA+distributed: base weights are frozen, skip after first round.
        if not (self.is_lora and self.use_distribute and self._lora_base_synced):
            for chunk_index, hf_named_tensors in enumerate(self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights, weight_type="base"
            )):
                _write_weight_audit(
                    stage="hf_chunk_base",
                    weight_version=self.weight_version,
                    tensors=hf_named_tensors,
                    chunk_index=chunk_index,
                )
                refs, long_lived_tensors = self._send_base_params(hf_named_tensors)
                results = ray.get(refs)
                _check_weight_sync_results(results, is_lora=False)
                del long_lived_tensors

        if self.is_lora:
            lora_sync_chunk_count = 0
            for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights, weight_type="lora"
            ):
                refs, long_lived_tensors = self._send_lora_params(hf_named_tensors)
                results = ray.get(refs)
                _check_weight_sync_results(results, is_lora=True)
                del long_lived_tensors
                lora_sync_chunk_count += 1

            if lora_sync_chunk_count == 0:
                raise RuntimeError(
                    "LoRA weight sync failed: the weight iterator produced zero chunks. "
                    "No adapter weights were sent to the rollout engine. This usually means "
                    "the Megatron-Bridge or SGLang version is incompatible."
                )

            if self.use_distribute and not self._lora_base_synced:
                self._lora_base_synced = True

        dist.barrier(group=get_gloo_group())

        if rank == 0:
            # `post_process_quantization` is related to the `process_weights_after_loading`
            # in the sglang rollout side, which should always be invoked after weight
            # updating.
            post_process_weights(
                rollout_engines=self.rollout_engines,
                restore_weights_before_load=False,
                post_process_quantization=True,
            )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_base_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        refs, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors=hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                refs = (refs or []) + refs_distributed
        return refs or [], long_lived_tensors

    def _send_lora_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        if not any(is_lora_weight_name(n) for n, _ in hf_named_tensors):
            raise RuntimeError(
                "LoRA weight sync failed: chunk contains no LoRA weights "
                "(no lora_A/lora_B names found). Check weight iterator configuration."
            )
        if self.use_distribute and self._is_distributed_src_rank:
            raise NotImplementedError("LoRA weight sync is not yet supported for distributed (non-colocated) engines")
        else:
            refs, long_lived_tensors = _send_to_colocated_engine(
                hf_named_tensors=hf_named_tensors,
                ipc_engine=self._ipc_engine,
                ipc_gather_src=self._ipc_gather_src,
                ipc_gather_group=self._ipc_gather_group,
                lora_config=self._lora_config,
                lora_name=LORA_ADAPTER_NAME,
                lora_loaded=self._lora_loaded,
            )
            self._lora_loaded = True
            return refs or [], long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version=None,
    lora_config: dict | None = None,
    lora_name: str | None = None,
    lora_loaded: bool = False,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    is_lora = lora_config is not None
    long_live_tensors = []

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": flattened_tensor_bucket.get_metadata(),
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if dist.get_rank() == ipc_gather_src:
        if is_lora:
            if lora_loaded:
                ray.get(ipc_engine.unload_lora_adapter.remote(lora_name=lora_name))

            # (Yusheng) to-do-1: update lora weights from tensors should support multiple dtypes (bf16, fp8, fp16, fp32)
            # currently, we only support 1 type. If there are multiple dtypes, we need to serialize the tensors for each dtype.
            # Thus, we need to apply the same way as `ipc_engine.update_weights_from_tensor` in future
            # (Yusheng) to-do-2: need to add ci test acc here - now it will pass but fail to update lora weights

            refs.append(
                ipc_engine.load_lora_adapter_from_tensors.remote(
                    lora_name=lora_name,
                    config_dict=lora_config,
                    serialized_tensors=serialized_named_tensors[0][0],
                    load_format="flattened_bucket",
                )
            )

        else:
            num_dtypes = len(serialized_named_tensors[0])
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                    "load_format": "flattened_bucket",
                    "weight_version": str(weight_version),
                }
                refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors


def _check_weight_sync_results(results: list, *, is_lora: bool) -> None:
    """Validate return values from rollout engine weight-sync RPCs.

    Raises RuntimeError if any engine reports failure, preventing silent
    failures when SGLang versions are incompatible.
    """
    sync_type = "LoRA" if is_lora else "Base model"
    for result in results:
        if isinstance(result, Mapping):
            success = result.get("success")
            error_msg = result.get("error_message") or result.get("error") or "unknown error"
        elif hasattr(result, "success"):
            success = result.success
            error_msg = getattr(result, "error_message", "unknown error")
        else:
            continue

        if success is False:
            raise RuntimeError(
                f"{sync_type} weight sync failed on rollout engine: {error_msg}. "
                f"Check SGLang version compatibility."
            )
