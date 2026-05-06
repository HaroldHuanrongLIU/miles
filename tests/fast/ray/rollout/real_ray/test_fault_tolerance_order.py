"""``actor_group.update_weights`` ordering under fault tolerance.

Production code (miles/ray/actor_group.py:122-132):

    async def update_weights(self):
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return
        if self.args.use_fault_tolerance:
            await self.rollout_manager.recover_updatable_engines.remote()  # (1)
        info = await self.rollout_manager.get_updatable_engines_and_lock.remote()  # (2)
        await self._broadcast("update_weights", info=info)  # (3)

If (1) raises, (2) and (3) MUST NOT run — otherwise actors would receive a
stale or inconsistent EnginesAndLock for engines that were not yet recovered.

Tests drive the REAL ``RayTrainGroup.update_weights`` against a fake ``self``
so a regression in actor_group.py is caught here, not in a hand-copied mirror.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.ray.actor_group import RayTrainGroup


def _make_remote_call(callable_or_value, *, is_coro=False):
    """Build a ``foo.remote()`` shaped mock.

    Production calls ``self.rollout_manager.recover_updatable_engines.remote()``
    and awaits the result. ``rollout_manager`` is a ray actor handle, so each
    method has a ``.remote`` attribute that returns an awaitable ObjectRef.
    """
    obj = MagicMock()
    if is_coro:
        obj.remote = AsyncMock(side_effect=callable_or_value)
    else:
        obj.remote = AsyncMock(return_value=callable_or_value)
    return obj


def _make_fake_self(*, use_fault_tolerance=True, debug_train_only=False, debug_rollout_only=False):
    """Build a fake ``self`` that supplies the attributes ``RayTrainGroup.update_weights``
    actually touches: ``args``, ``rollout_manager``, and ``_broadcast``."""
    self_ = SimpleNamespace()
    self_.args = SimpleNamespace(
        debug_train_only=debug_train_only,
        debug_rollout_only=debug_rollout_only,
        use_fault_tolerance=use_fault_tolerance,
    )
    self_.rollout_manager = SimpleNamespace(
        recover_updatable_engines=_make_remote_call(None),
        get_updatable_engines_and_lock=_make_remote_call(SimpleNamespace(name="EnginesAndLock")),
    )
    self_._broadcast = AsyncMock(return_value=None)
    return self_


class TestFaultToleranceOrder:
    @pytest.mark.asyncio
    async def test_recover_runs_before_get_info(self):
        order = []
        s = _make_fake_self()

        async def _recover():
            order.append("recover")
        async def _get_info():
            order.append("get_info")
            return SimpleNamespace()

        s.rollout_manager.recover_updatable_engines.remote = AsyncMock(side_effect=_recover)
        s.rollout_manager.get_updatable_engines_and_lock.remote = AsyncMock(side_effect=_get_info)

        await RayTrainGroup.update_weights(s)

        assert order == ["recover", "get_info"]
        s._broadcast.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recover_failure_aborts_get_info_and_broadcast(self):
        s = _make_fake_self()

        async def _recover_fail():
            raise RuntimeError("recover failed")

        s.rollout_manager.recover_updatable_engines.remote = AsyncMock(side_effect=_recover_fail)

        with pytest.raises(RuntimeError, match="recover failed"):
            await RayTrainGroup.update_weights(s)

        s.rollout_manager.get_updatable_engines_and_lock.remote.assert_not_awaited()
        s._broadcast.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_fault_tolerance_skips_recover(self):
        s = _make_fake_self(use_fault_tolerance=False)
        await RayTrainGroup.update_weights(s)
        s.rollout_manager.recover_updatable_engines.remote.assert_not_awaited()
        s.rollout_manager.get_updatable_engines_and_lock.remote.assert_awaited_once()
        s._broadcast.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_debug_modes_short_circuit_completely(self):
        s = _make_fake_self(debug_train_only=True)
        await RayTrainGroup.update_weights(s)
        s.rollout_manager.recover_updatable_engines.remote.assert_not_awaited()
        s.rollout_manager.get_updatable_engines_and_lock.remote.assert_not_awaited()
        s._broadcast.assert_not_awaited()


# ----------------------------- chaos / property tests -----------------------------


# (step_index, step_name): step_index 0 == recover, 1 == get_info, 2 == broadcast.
# Used by the parametrized chaos test below to pick a failure point.
_STEPS: list[tuple[int, str]] = [
    (0, "recover"),
    (1, "get_info"),
    (2, "broadcast"),
]


class TestFaultToleranceChaos:
    """Property-style: for ANY single failing step in update_weights, every
    later step MUST NOT execute. Catches regressions where a new try/except
    quietly swallows an upstream failure and lets stale state leak into
    ``_broadcast`` (the actor side of update_weights)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fail_step_index, fail_step_name", _STEPS)
    async def test_failure_at_any_step_aborts_subsequent_steps(
        self, fail_step_index: int, fail_step_name: str,
    ):
        s = _make_fake_self()
        executed: list[str] = []

        async def _recover():
            executed.append("recover")
            if fail_step_index == 0:
                raise RuntimeError(f"chaos@{fail_step_name}")

        async def _get_info():
            executed.append("get_info")
            if fail_step_index == 1:
                raise RuntimeError(f"chaos@{fail_step_name}")
            return SimpleNamespace(name="EnginesAndLock")

        async def _broadcast(*args, **kwargs):
            executed.append("broadcast")
            if fail_step_index == 2:
                raise RuntimeError(f"chaos@{fail_step_name}")

        s.rollout_manager.recover_updatable_engines.remote = AsyncMock(side_effect=_recover)
        s.rollout_manager.get_updatable_engines_and_lock.remote = AsyncMock(side_effect=_get_info)
        s._broadcast = AsyncMock(side_effect=_broadcast)

        with pytest.raises(RuntimeError, match=f"chaos@{fail_step_name}"):
            await RayTrainGroup.update_weights(s)

        # Invariant: every step strictly after the failing one must NOT have run.
        expected_executed = [name for i, name in _STEPS if i <= fail_step_index]
        assert executed == expected_executed, (
            f"failure at {fail_step_name} did not abort downstream: executed={executed}"
        )

    @pytest.mark.asyncio
    async def test_recover_succeeds_then_get_info_failure_skips_broadcast(self):
        """Mid-pipeline failure: recover ran clean, but get_info raised.
        broadcast must not run with stale (or missing) info — otherwise actors
        receive an EnginesAndLock for the pre-recover state."""
        s = _make_fake_self()
        executed: list[str] = []

        async def _recover():
            executed.append("recover")

        async def _get_info():
            executed.append("get_info")
            raise RuntimeError("get_info exploded mid-pipeline")

        async def _broadcast(*args, **kwargs):
            executed.append("broadcast")

        s.rollout_manager.recover_updatable_engines.remote = AsyncMock(side_effect=_recover)
        s.rollout_manager.get_updatable_engines_and_lock.remote = AsyncMock(side_effect=_get_info)
        s._broadcast = AsyncMock(side_effect=_broadcast)

        with pytest.raises(RuntimeError, match="get_info exploded"):
            await RayTrainGroup.update_weights(s)

        assert executed == ["recover", "get_info"]
        s._broadcast.assert_not_awaited()
