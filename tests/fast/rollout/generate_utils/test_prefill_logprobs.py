from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils import prefill_logprobs
from miles.utils.types import Sample


@pytest.mark.asyncio
async def test_recompute_rollout_logprobs_via_prefill_uses_response_tail(monkeypatch):
    sample = Sample(
        tokens=[10, 11, 12, 20, 21, 22],
        response_length=3,
        rollout_log_probs=[-9.0, -9.0, -9.0],
        status=Sample.Status.COMPLETED,
    )
    args = SimpleNamespace(recompute_logprobs_via_prefill=True, sglang_enable_lora=False)
    seen = {}

    async def fake_post(url, payload, headers=None):
        seen["url"] = url
        seen["payload"] = payload
        seen["headers"] = headers
        return {
            "meta_info": {
                "input_token_logprobs": [
                    (None, 10),
                    (-9.1, 11),
                    (-9.2, 12),
                    (-0.1, 20),
                    (-0.2, 21),
                    (-0.3, 22),
                ]
            }
        }

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    await prefill_logprobs.recompute_rollout_logprobs_via_prefill(
        args,
        sample,
        url="http://localhost/generate",
        sampling_params={"temperature": 1, "max_new_tokens": 128},
        headers={"X-Test": "1"},
    )

    assert sample.rollout_log_probs == [-0.1, -0.2, -0.3]
    assert sample.metadata["rollout_log_probs_source"] == "sglang_prefill_recompute"
    assert seen["url"] == "http://localhost/generate"
    assert seen["headers"] == {"X-Test": "1"}
    assert seen["payload"]["input_ids"] == sample.tokens
    assert seen["payload"]["return_logprob"] is True
    assert seen["payload"]["logprob_start_len"] == 0
    assert seen["payload"]["sampling_params"]["max_new_tokens"] == 0
    assert seen["payload"]["sampling_params"]["temperature"] == 0


@pytest.mark.asyncio
async def test_recompute_rollout_logprobs_via_prefill_checks_token_alignment(monkeypatch):
    sample = Sample(tokens=[10, 11, 20], response_length=1, status=Sample.Status.COMPLETED)
    args = SimpleNamespace(recompute_logprobs_via_prefill=True, sglang_enable_lora=False)

    async def fake_post(url, payload, headers=None):
        return {"meta_info": {"input_token_logprobs": [(None, 10), (-0.1, 11), (-0.2, 999)]}}

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    with pytest.raises(ValueError, match="token alignment mismatch"):
        await prefill_logprobs.recompute_rollout_logprobs_via_prefill(
            args,
            sample,
            url="http://localhost/generate",
            sampling_params={},
        )


@pytest.mark.asyncio
async def test_recompute_samples_flushes_each_batch_and_batches_prefill_score(monkeypatch):
    samples = [
        Sample(tokens=[10, 11, 20], response_length=1, status=Sample.Status.COMPLETED),
        Sample(tokens=[10, 11, 21], response_length=1, status=Sample.Status.COMPLETED),
    ]
    args = SimpleNamespace(
        recompute_logprobs_via_prefill=True,
        sglang_enable_lora=False,
        sglang_router_policy="round_robin",
    )
    calls = []

    async def fake_post(url, payload, action="post", headers=None):
        calls.append((url, payload, action, headers))
        if url.endswith("/flush_cache"):
            return {}
        return [
            {
                "meta_info": {
                    "input_token_logprobs": [
                        (None, tokens[0]),
                        (-0.5, tokens[1]),
                        (-float(tokens[-1]), tokens[-1]),
                    ]
                }
            }
            for tokens in payload["input_ids"]
        ]

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    await prefill_logprobs.recompute_samples_rollout_logprobs_via_prefill(
        args,
        samples,
        url="http://localhost/generate",
        sampling_params={"max_new_tokens": 32},
    )

    assert [sample.rollout_log_probs for sample in samples] == [[-20.0], [-21.0]]
    assert [call[0] for call in calls] == [
        "http://localhost/flush_cache",
        "http://localhost/generate",
    ]
    assert [call[2] for call in calls] == ["post", "post"]
    assert calls[1][1]["input_ids"] == [[10, 11, 20], [10, 11, 21]]
    assert calls[1][1]["logprob_start_len"] == 0


@pytest.mark.asyncio
async def test_recompute_samples_batches_full_sequence_scoring(monkeypatch):
    samples = [
        Sample(tokens=[10, 11, 20], response_length=1, status=Sample.Status.COMPLETED),
        Sample(tokens=[10, 11, 12, 21], response_length=1, status=Sample.Status.COMPLETED),
        Sample(tokens=[10, 11, 22], response_length=1, status=Sample.Status.COMPLETED),
    ]
    args = SimpleNamespace(
        recompute_logprobs_via_prefill=True,
        sglang_enable_lora=False,
        sglang_router_policy="round_robin",
    )
    calls = []

    async def fake_post(url, payload, action="post", headers=None):
        calls.append((url, payload, action, headers))
        if url.endswith("/flush_cache"):
            return {}
        return [
            {
                "meta_info": {
                    "input_token_logprobs": [
                        (None, tokens[0]),
                        *[(-0.5, token) for token in tokens[1:-1]],
                        (-float(tokens[-1]), tokens[-1]),
                    ]
                }
            }
            for tokens in payload["input_ids"]
        ]

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    await prefill_logprobs.recompute_samples_rollout_logprobs_via_prefill(
        args,
        samples,
        url="http://localhost/generate",
        sampling_params={"max_new_tokens": 32},
    )

    assert [sample.rollout_log_probs for sample in samples] == [
        [-20.0],
        [-21.0],
        [-22.0],
    ]
    assert [call[0] for call in calls] == [
        "http://localhost/flush_cache",
        "http://localhost/generate",
    ]
    assert calls[1][1]["logprob_start_len"] == 0
    assert calls[1][1]["input_ids"] == [[10, 11, 20], [10, 11, 12, 21], [10, 11, 22]]


@pytest.mark.asyncio
async def test_recompute_samples_caps_prefill_scoring_batch_size(monkeypatch):
    samples = [
        Sample(tokens=[10, 11, 20 + i], response_length=1, status=Sample.Status.COMPLETED)
        for i in range(5)
    ]
    args = SimpleNamespace(
        recompute_logprobs_via_prefill=True,
        recompute_logprobs_prefill_batch_size=2,
        sglang_enable_lora=False,
        sglang_router_policy="round_robin",
    )
    calls = []

    async def fake_post(url, payload, action="post", headers=None):
        calls.append((url, payload, action, headers))
        if url.endswith("/flush_cache"):
            return {}
        return [
            {
                "meta_info": {
                    "input_token_logprobs": [
                        (None, tokens[0]),
                        (-0.5, tokens[1]),
                        (-float(tokens[-1]), tokens[-1]),
                    ]
                }
            }
            for tokens in payload["input_ids"]
        ]

    monkeypatch.setattr(prefill_logprobs, "post", fake_post)

    await prefill_logprobs.recompute_samples_rollout_logprobs_via_prefill(
        args,
        samples,
        url="http://localhost/generate",
        sampling_params={},
    )

    generate_payloads = [call[1] for call in calls if call[0].endswith("/generate")]
    assert [len(payload["input_ids"]) for payload in generate_payloads] == [2, 2, 1]
    assert [sample.rollout_log_probs for sample in samples] == [
        [-(20 + i)] for i in range(5)
    ]


def test_extract_response_logprobs_selects_aligned_window_before_padding():
    sample = Sample(tokens=[10, 11, 20, 21, 22], response_length=3)
    meta_info = {
        "input_token_logprobs": [
            (-9.0, 11),
            (-9.1, 20),
            (-0.1, 20),
            (-0.2, 21),
            (-0.3, 22),
            (None, 0),
        ]
    }

    assert prefill_logprobs._extract_response_logprobs(sample, meta_info, logprob_start_len=0) == [
        -0.1,
        -0.2,
        -0.3,
    ]


def test_extract_response_logprobs_prefers_response_offset_for_repeated_tokens():
    sample = Sample(tokens=[7, 8, 8, 8], response_length=2)
    meta_info = {
        "input_token_logprobs": [
            (-9.0, 8),
            (-9.1, 8),
            (-0.1, 8),
            (-0.2, 8),
            (None, 0),
        ]
    }

    assert prefill_logprobs._extract_response_logprobs(sample, meta_info, logprob_start_len=0) == [
        -0.1,
        -0.2,
    ]
