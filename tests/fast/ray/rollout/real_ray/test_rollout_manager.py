"""``RolloutManager`` cell dispatch + ``EnginesAndLock`` flow driven through
real Ray actors.

We do not stand up a full ``@ray.remote`` ``RolloutManager`` actor (its
``__init__`` calls ``start_rollout_servers`` which is heavy). Instead we
unwrap the class via ``__ray_actor_class__`` and call its methods on a
minimal fake ``self`` that holds:

- ``self.servers``: real dict of real RolloutServer with real ServerGroups
  whose engines are real ``MockSGLangEngine`` Ray actors
- ``self.rollout_engine_lock``: a real ``asyncio.Lock``
- ``self._health_monitors``: empty list (recover_updatable_engines bails
  before touching them in our test scenarios)

This file isolates the methods that genuinely need real actor data flow
(start_cell, stop_cell, get_updatable_engines_and_lock with real engines).
Pure routing/flag-flip helpers without Ray content live in
``test_grab_bag.py``."""

from __future__ import annotations

import asyncio

import pytest
import ray

from miles.ray.rollout.addr_allocator import PortCursors
from miles.ray.rollout.rollout_manager import RolloutManager
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_engine import ServerEngine
from miles.ray.rollout.server_group import ServerGroup
from tests.fast.ray.rollout.conftest import make_args


_RolloutManagerCls = RolloutManager.__ray_actor_class__


def _build_server(*, pg_tuple: tuple, model_name: str, update_weights: bool,
                  num_engines: int = 2, num_groups: int = 1) -> RolloutServer:
    args = make_args(num_gpus_per_node=8)
    groups = []
    for _ in range(num_groups):
        engines = [ServerEngine() for _ in range(num_engines)]
        groups.append(ServerGroup(
            args=args, pg=pg_tuple, all_engines=engines,
            num_gpus_per_engine=1, has_new_engines=False,
            update_weights=update_weights,
        ))
    return RolloutServer(server_groups=groups, model_name=model_name,
                         update_weights=update_weights)


def _build_fake_self(servers: dict[str, RolloutServer]) -> _RolloutManagerCls:
    """Build a RolloutManager instance bypassing ``__init__`` (which would call
    ``start_rollout_servers``). The methods we test only touch the few attrs
    we set here. Using ``__new__`` keeps ``self._get_updatable_server`` etc.
    as real bound methods rather than needing manual binding."""
    fake = _RolloutManagerCls.__new__(_RolloutManagerCls)
    fake.servers = servers
    fake.rollout_engine_lock = asyncio.Lock()
    fake._health_monitors = []
    fake.rollout_id = 0
    return fake


def _start_all(servers: dict[str, RolloutServer]) -> None:
    for srv in servers.values():
        for g in srv.server_groups:
            handles, indices = g.start_engines(PortCursors.empty())
            ray.get(handles)
            g.mark_alive(indices)


def _kill_all(servers: dict[str, RolloutServer]) -> None:
    for srv in servers.values():
        for g in srv.server_groups:
            for e in g.all_engines:
                if e.is_allocated:
                    try:
                        ray.kill(e.actor_handle)
                    except Exception:
                        pass


# ----------------------------- start_cell / stop_cell -----------------------------


@pytest.mark.asyncio
class TestStartStopCell:
    async def test_start_cell_creates_actors_for_dead_engines(
        self, patched_sglang_engine, placement_group_factory,
    ):
        """Mark all engines stopped, then start_cell(0) must drive a real
        recover() that creates a real actor and runs init on it."""
        pg = placement_group_factory(2)
        srv = _build_server(pg_tuple=pg, model_name="actor", update_weights=True,
                            num_engines=2)
        # Mark engines stopped so recover() will start them.
        for e in srv.server_groups[0].all_engines:
            e.mark_stopped()
        servers = {"actor": srv}
        fake = _build_fake_self(servers)

        try:
            await _RolloutManagerCls.start_cell(fake, 0)
            # cell 0 → engine 0 (nodes_per_engine=1, so cell-per-engine)
            assert srv.server_groups[0].all_engines[0].is_allocated
            calls = ray.get(srv.server_groups[0].all_engines[0].actor_handle.get_calls.remote())
            assert "init" in [c[0] for c in calls]
        finally:
            _kill_all(servers)

    async def test_stop_cell_kills_target_engine(
        self, patched_sglang_engine, placement_group_factory,
    ):
        """stop_cell(0) must kill exactly the actor for that cell, not others."""
        pg = placement_group_factory(2)
        srv = _build_server(pg_tuple=pg, model_name="actor", update_weights=True,
                            num_engines=2)
        servers = {"actor": srv}
        _start_all(servers)
        fake = _build_fake_self(servers)

        try:
            actors = [e.actor_handle for e in srv.server_groups[0].all_engines]
            await _RolloutManagerCls.stop_cell(fake, 0)

            assert not srv.server_groups[0].all_engines[0].is_allocated
            assert srv.server_groups[0].all_engines[1].is_allocated, "cell 1 untouched"

            # Real Ray claim: the killed actor's followup .remote() raises.
            with pytest.raises((ray.exceptions.RayActorError, ray.exceptions.RayTaskError)):
                ray.get(actors[0].health_generate.remote(timeout=1.0), timeout=10.0)
        finally:
            _kill_all(servers)


# ----------------------------- get_updatable_engines_and_lock -----------------------------


@pytest.mark.asyncio
class TestGetUpdatableEnginesAndLock:
    async def test_returns_populated_when_updatable_exists(
        self, patched_sglang_engine, placement_group_factory,
    ):
        """The updatable server's engines / counts / offsets / has_new_engines
        flow into EnginesAndLock. Engines are real Ray actor handles."""
        pg_a = placement_group_factory(2)
        pg_b = placement_group_factory(2)
        ref = _build_server(pg_tuple=pg_a, model_name="ref", update_weights=False,
                            num_engines=2)
        actor = _build_server(pg_tuple=pg_b, model_name="actor", update_weights=True,
                              num_engines=2)
        actor.server_groups[0].has_new_engines = True
        servers = {"actor": actor, "ref": ref}
        _start_all(servers)
        fake = _build_fake_self(servers)

        try:
            eal = await _RolloutManagerCls.get_updatable_engines_and_lock(fake)
            assert eal.has_new_engines is True
            assert len(eal.rollout_engines) == 2
            assert eal.engine_gpu_counts == [1, 1]
            # Sanity: the handles are real ActorHandles, not MagicMock.
            assert all(isinstance(h, ray.actor.ActorHandle) for h in eal.rollout_engines)
        finally:
            _kill_all(servers)
