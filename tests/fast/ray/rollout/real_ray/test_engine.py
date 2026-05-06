"""Engine-level real Ray actor smoke tests.

Drives MockSGLangEngine as a real ``@ray.remote`` actor (not a MagicMock):
real ``ray.init()`` minicluster, real ``.remote()`` / ``ray.get()`` round-trips,
real ``ray.kill()`` teardown. Catches the class of bug that "all mocks pass
but a real Ray run hangs / drops calls / mis-serializes args"."""

from __future__ import annotations

import pytest
import ray

from tests.fast.ray.rollout.conftest import make_args
from tests.fast.ray.rollout.lifecycle.mock_engine import MockSGLangEngine


class TestMockEngineRealRayLifecycle:
    """End-to-end: a real Ray actor can be created, driven through every
    method the rollout code calls, and torn down — without any monkeypatching
    of ``ray.remote`` / ``ray.get`` / ``ray.kill``."""

    def test_actor_construction_and_method_round_trip(self, ray_cluster):
        args = make_args(rollout_num_gpus_per_engine=1)
        actor = MockSGLangEngine.options(num_cpus=0.1, num_gpus=0).remote(
            args, rank=0, worker_type="regular", base_gpu_id=0,
            sglang_overrides={}, num_gpus_per_engine=1,
        )
        try:
            # Drive the engine through every method rollout code touches.
            # Asserting on the mock's return values would just pin the mock
            # (mock contract is enforced by test_mock_engine_contract.py); the
            # value here is that .remote() / ray.get() actually round-trip.
            ray.get(actor.init.remote(host="127.0.0.1", port=20000))
            ray.get(actor.health_generate.remote(timeout=1.0))
            ray.get(actor.release_memory_occupation.remote(tags=["WEIGHTS"]))
            ray.get(actor.resume_memory_occupation.remote(tags=["WEIGHTS"]))
            ray.get(actor.update_weights_from_disk.remote(model_path="/fake"))
            ray.get(actor.check_weights.remote(action="pre_update"))

            # Call ordering survives the Ray boundary: arg serialization didn't
            # silently drop or reorder messages. This IS a real-Ray claim.
            calls = ray.get(actor.get_calls.remote())
            method_names = [name for name, _, _ in calls]
            assert method_names == [
                "init",
                "health_generate",
                "release_memory_occupation",
                "resume_memory_occupation",
                "update_weights_from_disk",
                "check_weights",
            ]
        finally:
            try:
                ray.get(actor.shutdown.remote())
            finally:
                ray.kill(actor)

    def test_fault_injection_round_trips_through_ray(self, ray_cluster):
        """``set_fault`` schedules an exception on the next call to a method.
        Verify the exception actually surfaces back to the caller via ``ray.get``."""
        args = make_args(rollout_num_gpus_per_engine=1)
        actor = MockSGLangEngine.options(num_cpus=0.1, num_gpus=0).remote(
            args, rank=0, worker_type="regular", base_gpu_id=0,
            sglang_overrides={}, num_gpus_per_engine=1,
        )
        try:
            ray.get(actor.set_fault.remote("health_generate", RuntimeError("boom")))
            with pytest.raises(ray.exceptions.RayTaskError, match="boom"):
                ray.get(actor.health_generate.remote(timeout=1.0))
            # Fault is one-shot — second call must succeed.
            assert ray.get(actor.health_generate.remote(timeout=1.0)) is True
        finally:
            ray.kill(actor)

