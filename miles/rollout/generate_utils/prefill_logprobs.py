from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Any

from miles.backends.megatron_utils.lora_utils import LORA_ADAPTER_NAME, is_lora_enabled
from miles.utils.http_utils import post
from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.types import Sample

_DEFAULT_PREFILL_SCORING_BATCH_SIZE = 4


def _build_prefill_scoring_payload(
    args: Any,
    sample: Sample,
    sampling_params: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_len = len(sample.tokens) - sample.response_length
    if prompt_len <= 0:
        raise ValueError(
            "Cannot recompute rollout logprobs via prefill without at least one prompt token: "
            f"tokens={len(sample.tokens)}, response_length={sample.response_length}"
        )

    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            **dict(sampling_params),
            "max_new_tokens": 0,
            "temperature": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        # Request full-sequence scoring and select the response window by token
        # ids when parsing. Starting at the prompt boundary makes SGLang's first
        # returned logprob slot boundary-dependent and can surface as None.
        "logprob_start_len": 0,
    }

    if is_lora_enabled(args):
        payload["lora_path"] = LORA_ADAPTER_NAME

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    return payload


def _can_batch_prefill_score(args: Any, samples: list[Sample]) -> bool:
    if getattr(args, "sglang_router_policy", None) == "consistent_hashing":
        return False
    return not any(sample.multimodal_inputs and sample.multimodal_inputs.get("images") for sample in samples)


def _build_batch_prefill_scoring_payload(
    args: Any,
    samples: list[Sample],
    sampling_params: Mapping[str, Any],
) -> dict[str, Any]:
    payloads = [_build_prefill_scoring_payload(args, sample, sampling_params) for sample in samples]
    logprob_start_len = payloads[0]["logprob_start_len"]
    if any(payload["logprob_start_len"] != logprob_start_len for payload in payloads):
        raise ValueError("Batched SGLang prefill scoring requires a shared logprob_start_len")

    batch_payload: dict[str, Any] = {
        "input_ids": [payload["input_ids"] for payload in payloads],
        "sampling_params": payloads[0]["sampling_params"],
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
    }
    if "lora_path" in payloads[0]:
        batch_payload["lora_path"] = payloads[0]["lora_path"]
    return batch_payload


def _iter_prefill_scoring_batches(args: Any, samples: list[Sample]) -> Iterator[list[Sample]]:
    batch_size = getattr(
        args,
        "recompute_logprobs_prefill_batch_size",
        _DEFAULT_PREFILL_SCORING_BATCH_SIZE,
    )
    if batch_size <= 0:
        batch_size = 1
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


def _extract_response_logprobs(
    sample: Sample,
    meta_info: Mapping[str, Any],
    *,
    logprob_start_len: int,
) -> list[float]:
    input_token_logprobs = meta_info.get("input_token_logprobs")
    if not input_token_logprobs:
        raise ValueError("SGLang prefill scoring response did not include input_token_logprobs")

    prompt_len = len(sample.tokens) - sample.response_length
    response_tokens = sample.tokens[-sample.response_length :]
    preferred_start = max(prompt_len - logprob_start_len, 0)
    candidate_starts = [preferred_start]
    candidate_starts.extend(
        start
        for start in range(0, len(input_token_logprobs) - sample.response_length + 1)
        if start != preferred_start
    )

    matching_response_items = None
    matching_with_none = None
    for start in candidate_starts:
        end = start + sample.response_length
        if end > len(input_token_logprobs):
            continue
        response_items = input_token_logprobs[start:end]
        scored_tokens = [item[1] for item in response_items]
        if scored_tokens != response_tokens:
            continue
        if any(item[0] is None for item in response_items):
            matching_with_none = response_items
            continue
        matching_response_items = response_items
        break

    if matching_response_items is None:
        scored_tokens = [item[1] for item in input_token_logprobs]
        if matching_with_none is not None:
            none_positions = [
                idx for idx, item in enumerate(matching_with_none) if item[0] is None
            ]
            raise ValueError(
                "SGLang prefill scoring returned None inside the aligned response window: "
                f"none_positions={none_positions}, response_len={sample.response_length}"
            )
        raise ValueError(
            "SGLang prefill scoring token alignment mismatch: "
            f"expected response tail {response_tokens[:8]}... len={len(response_tokens)}, "
            f"got scored tokens {scored_tokens[:12]}... len={len(scored_tokens)}"
        )

    return [item[0] for item in matching_response_items]


def _record_prefill_scoring_metadata(sample: Sample, meta_info: Mapping[str, Any]) -> None:
    dp_rank = meta_info.get("dp_rank")
    if dp_rank is not None:
        sample.metadata["rollout_log_probs_dp_rank"] = int(dp_rank)
    sample.metadata["rollout_log_probs_source"] = "sglang_prefill_recompute"


async def recompute_rollout_logprobs_via_prefill(
    args: Any,
    sample: Sample,
    *,
    url: str,
    sampling_params: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
) -> None:
    if not getattr(args, "recompute_logprobs_via_prefill", False):
        return
    if sample.response_length == 0:
        sample.rollout_log_probs = []
        return
    if sample.status == Sample.Status.ABORTED:
        return

    payload = _build_prefill_scoring_payload(args, sample, sampling_params)
    output = await post(url, payload, headers=headers)
    meta_info = output["meta_info"]
    sample.rollout_log_probs = _extract_response_logprobs(
        sample,
        meta_info,
        logprob_start_len=payload["logprob_start_len"],
    )
    _record_prefill_scoring_metadata(sample, meta_info)


async def recompute_samples_rollout_logprobs_via_prefill(
    args: Any,
    samples: list[Sample],
    *,
    url: str,
    sampling_params: Mapping[str, Any],
) -> None:
    if not getattr(args, "recompute_logprobs_via_prefill", False):
        return

    samples_to_score = [
        sample for sample in samples if sample.response_length != 0 and sample.status != Sample.Status.ABORTED
    ]
    if not samples_to_score:
        return

    flush_url = url.rsplit("/", 1)[0] + "/flush_cache"

    if _can_batch_prefill_score(args, samples_to_score):
        samples_by_logprob_start_len: dict[int, list[Sample]] = defaultdict(list)
        for sample in samples_to_score:
            payload = _build_prefill_scoring_payload(args, sample, sampling_params)
            samples_by_logprob_start_len[payload["logprob_start_len"]].append(sample)

        for same_start_len_samples in samples_by_logprob_start_len.values():
            for batch_samples in _iter_prefill_scoring_batches(args, same_start_len_samples):
                # SGLang can serve scoring requests from radix/KV cache. Flush before
                # each scoring group so every group uses the same clean-prefill path.
                await post(flush_url, {})
                payload = _build_batch_prefill_scoring_payload(args, batch_samples, sampling_params)
                outputs = await post(url, payload)
                if not isinstance(outputs, list):
                    raise ValueError(f"SGLang batch prefill scoring returned {type(outputs).__name__}, expected list")
                if len(outputs) != len(batch_samples):
                    raise ValueError(
                        "SGLang batch prefill scoring output count mismatch: "
                        f"expected {len(batch_samples)}, got {len(outputs)}"
                    )
                for sample, output in zip(batch_samples, outputs, strict=True):
                    meta_info = output["meta_info"]
                    sample.rollout_log_probs = _extract_response_logprobs(
                        sample,
                        meta_info,
                        logprob_start_len=payload["logprob_start_len"],
                    )
                    _record_prefill_scoring_metadata(sample, meta_info)
        return

    for sample in samples_to_score:
        headers = None
        uses_consistent_hashing = getattr(args, "sglang_router_policy", None) == "consistent_hashing"
        if uses_consistent_hashing and sample.session_id:
            headers = {"X-SMG-Routing-Key": sample.session_id}

        await post(flush_url, {}, headers=headers)
        await recompute_rollout_logprobs_via_prefill(
            args,
            sample,
            url=url,
            sampling_params=sampling_params,
            headers=headers,
        )
