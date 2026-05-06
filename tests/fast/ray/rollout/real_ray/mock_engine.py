"""MockSGLangEngine: a Ray actor that mimics SGLangEngine's public API surface
without touching CUDA / sglang / actual model weights.

Used by tests that need real Ray actor lifecycle but do not need a real
engine. The contract test in ``test_mock_engine_contract.py`` enforces
that this mock exposes every method called by ``miles/ray/rollout/``."""

from __future__ import annotations

import logging
import threading
from typing import Any

import ray

logger = logging.getLogger(__name__)


@ray.remote
class MockSGLangEngine:
    """In-memory stand-in for ``SGLangEngine``.

    Records every call into ``self.calls`` so tests can assert sequence and
    arguments. Fault injection is set via ``set_fault(method, exception)``."""

    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int = 0,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int = 1,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine

        self.initialized = False
        self.calls: list[tuple[str, tuple, dict]] = []
        self._faults: dict[str, BaseException] = {}
        self._port_seq = 20000
        self._lock = threading.Lock()

    # --------------------------- fault control ---------------------------

    def set_fault(self, method: str, exception: BaseException | None):
        """Schedule the given method to raise on its next call. Pass None to clear."""
        if exception is None:
            self._faults.pop(method, None)
        else:
            self._faults[method] = exception

    def get_calls(self) -> list[tuple[str, tuple, dict]]:
        return list(self.calls)

    def get_init_kwargs(self) -> dict | None:
        for name, _args, kwargs in self.calls:
            if name == "init":
                return dict(kwargs)
        return None

    # --------------------------- engine API ------------------------------

    def init(self, **kwargs):
        # Real SGLangEngine.init returns None; mirror that so tests asserting on
        # the return value catch real-path bugs.
        self._record("init", (), kwargs)
        self._maybe_fault("init")
        self.initialized = True
        return None

    def shutdown(self):
        self._record("shutdown", (), {})
        self._maybe_fault("shutdown")
        self.initialized = False
        return True

    def health_generate(self, timeout: float = 5.0):
        self._record("health_generate", (), {"timeout": timeout})
        self._maybe_fault("health_generate")
        return True

    def release_memory_occupation(self, tags: list[str] | None = None):
        self._record("release_memory_occupation", (), {"tags": tags})
        self._maybe_fault("release_memory_occupation")
        return True

    def resume_memory_occupation(self, tags: list[str] | None = None):
        self._record("resume_memory_occupation", (), {"tags": tags})
        self._maybe_fault("resume_memory_occupation")
        return True

    def update_weights_from_disk(self, model_path: str, load_format: str | None = None):
        self._record("update_weights_from_disk", (model_path,), {"load_format": load_format})
        self._maybe_fault("update_weights_from_disk")
        return True

    def update_weights_from_tensor(self, *args, **kwargs):
        self._record("update_weights_from_tensor", args, kwargs)
        return True

    def check_weights(self, action: str):
        # Real SGLangEngine.check_weights returns the http server response
        # (an arbitrary dict produced by the model's check_weights handler).
        # Tests should not assume any particular shape; we return a sentinel
        # dict that's neither truthy in a meaningful way nor matches what the
        # real server returns, encouraging tests to assert on .called rather
        # than the return value.
        self._record("check_weights", (action,), {})
        return {"_mock": True, "action": action}

    def get_server_info(self):
        self._record("get_server_info", (), {})
        return {"rank": self.rank, "worker_type": self.worker_type}

    def get_weight_version(self):
        self._record("get_weight_version", (), {})
        return "mock-v0"

    def get_parallelism_info(self, rank: int):
        self._record("get_parallelism_info", (rank,), {})
        return {"rank": rank}

    def flush_cache(self):
        self._record("flush_cache", (), {})
        return True

    def pause_generation(self):
        self._record("pause_generation", (), {})

    def continue_generation(self):
        self._record("continue_generation", (), {})

    def update_weight_version(self, weight_version: str):
        self._record("update_weight_version", (weight_version,), {})

    def post_process_weights(self, *args, **kwargs):
        self._record("post_process_weights", args, kwargs)

    def init_weights_update_group(self, *args, **kwargs):
        self._record("init_weights_update_group", args, kwargs)

    def destroy_weights_update_group(self, *args, **kwargs):
        self._record("destroy_weights_update_group", args, kwargs)

    def update_weights_from_distributed(self, *args, **kwargs):
        self._record("update_weights_from_distributed", args, kwargs)

    def get_remote_instance_transfer_engine_info(self, rank: int):
        self._record("get_remote_instance_transfer_engine_info", (rank,), {})
        return {"rank": rank}

    def load_lora_adapter_from_tensors(self, *args, **kwargs):
        self._record("load_lora_adapter_from_tensors", args, kwargs)

    def unload_lora_adapter(self, lora_name: str):
        self._record("unload_lora_adapter", (lora_name,), {})

    def start_profile(self, *args, **kwargs):
        self._record("start_profile", args, kwargs)

    def stop_profile(self):
        self._record("stop_profile", (), {})

    def simulate_crash(self):
        """Match real SGLangEngine.simulate_crash semantics.

        Real implementation (sglang_engine.py:575) calls ``self.shutdown()``;
        the actor itself stays alive (only the http server process dies).
        Earlier mock used ``os._exit(1)`` which kills the actor — that
        diverged from production and would mask any test that depended on
        actor-still-alive behavior."""
        self._record("simulate_crash", (), {})
        self.shutdown()

    # --------------------------- port allocation ------------------------

    def _get_current_node_ip_and_free_port(self, start_port: int = 15000, consecutive: int = 1):
        """Mock the port-finding helper used by addr_allocator."""
        self._record("_get_current_node_ip_and_free_port", (), {"start_port": start_port, "consecutive": consecutive})
        with self._lock:
            port = max(self._port_seq, start_port)
            self._port_seq = port + consecutive
            return ("127.0.0.1", port)

    # --------------------------- impl ------------------------------------

    def _record(self, name: str, args: tuple, kwargs: dict) -> None:
        self.calls.append((name, args, kwargs))

    def _maybe_fault(self, method: str) -> None:
        exc = self._faults.pop(method, None)
        if exc is not None:
            raise exc
