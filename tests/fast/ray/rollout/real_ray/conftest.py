"""Shared fixtures for lifecycle/ tests that drive real Ray actors with
``MockSGLangEngine`` (no GPU, no real sglang).

Two fixtures:
- ``ray_cluster`` (module-scoped): real ``ray.init()`` minicluster. On CI the
  RAY_ADDRESS env var is set; ``ray.init`` then connects to the existing
  cluster and rejects ``num_cpus``, so we branch on that.
- ``placement_group_factory`` (function-scoped): hands out CPU-only placement
  groups that ``ServerGroup.start_engines`` can consume. Each test cleans up
  the PG it created in the fixture's teardown.
"""

from __future__ import annotations

import os

import pytest
import ray

# Per-engine resource footprint when running MockSGLangEngine. Production
# ServerGroup.start_engines hard-codes ``num_gpus=0.2, num_cpus=0.2`` on
# the actor's .options(...) call, so each PG bundle must satisfy that.
_PER_ENGINE_NUM_CPUS = 0.2
_PER_ENGINE_NUM_GPUS = 0.2


@pytest.fixture(scope="module")
def ray_cluster():
    """Boot (or attach to) a Ray cluster.

    On a clean machine: ``ray.init(num_cpus=4, ...)``. On CI where
    RAY_ADDRESS points at an existing cluster: connect without ``num_cpus``
    (Ray rejects it when joining an existing cluster)."""
    if not ray.is_initialized():
        kwargs: dict = dict(
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
        )
        if not os.environ.get("RAY_ADDRESS"):
            kwargs["num_cpus"] = 4
        ray.init(**kwargs)
    yield
    # Don't shut down — other test modules may share this session-wide state.


@pytest.fixture
def placement_group_factory(ray_cluster):
    """Hand out ad-hoc CPU-only placement groups.

    Yields a callable ``make(num_engines: int) -> tuple``. The returned tuple
    matches what ``ServerGroup.pg`` expects:
    ``(placement_group, reordered_bundle_indices, reordered_gpu_ids)``.

    All PGs created during the test are torn down on fixture teardown."""
    created: list = []

    def _make(num_engines: int) -> tuple:
        bundles = [
            {"CPU": _PER_ENGINE_NUM_CPUS, "GPU": _PER_ENGINE_NUM_GPUS}
            for _ in range(num_engines)
        ]
        pg = ray.util.placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())
        created.append(pg)
        # ServerGroup expects (pg, bundle_indices, gpu_ids).
        return (pg, list(range(num_engines)), list(range(num_engines)))

    yield _make

    for pg in created:
        try:
            ray.util.remove_placement_group(pg)
        except Exception:
            pass


@pytest.fixture
def mock_engine_class(ray_cluster):
    """Hand back the unwrapped MockSGLangEngine class so tests can pass it as
    the ``SGLangEngine`` substitute via ``monkeypatch.setattr``.

    Production code does ``ray.remote(SGLangEngine)`` — substituting the
    already-wrapped MockSGLangEngine would double-wrap and fail; so callers
    monkeypatch ``SGLangEngine`` (the unwrapped class) inside
    ``miles.ray.rollout.server_group``."""
    from tests.fast.ray.rollout.lifecycle.mock_engine import MockSGLangEngine
    return MockSGLangEngine.__ray_actor_class__


@pytest.fixture
def patched_sglang_engine(monkeypatch, mock_engine_class):
    """Replace ``miles.ray.rollout.server_group.SGLangEngine`` with the
    unwrapped MockSGLangEngine class for the duration of one test.

    Also patches ``allocate_rollout_engine_addr_and_ports_normal`` to return
    a fully-formed addr_and_ports dict without needing the engines to actually
    speak the ``_get_current_node_ip_and_free_port`` protocol — that path is
    covered separately by addr_allocator tests."""
    import miles.ray.rollout.server_group as mod

    monkeypatch.setattr(mod, "SGLangEngine", mock_engine_class)

    from miles.ray.rollout.addr_allocator import PortCursors

    def _fake_alloc(*args, **kwargs):
        engines = kwargs["rollout_engines"]
        addr_and_ports = {}
        for rank, _ in engines:
            addr_and_ports[rank] = dict(
                host="127.0.0.1",
                port=30000 + rank,
                nccl_port=31000 + rank,
                engine_info_bootstrap_port=32000 + rank,
                dist_init_addr=f"127.0.0.1:{33000 + rank}",
            )
        return addr_and_ports, PortCursors(_values={0: 34000})

    monkeypatch.setattr(mod, "allocate_rollout_engine_addr_and_ports_normal", _fake_alloc)


@pytest.fixture
def patched_sglang_engine_real_allocator(monkeypatch, mock_engine_class):
    """Like ``patched_sglang_engine`` but does NOT stub the addr allocator —
    use this to exercise the real ``allocate_rollout_engine_addr_and_ports_normal``
    against MockSGLangEngine actors. The mock implements
    ``_get_current_node_ip_and_free_port`` with a deterministic
    ``max(seq, start_port)`` counter, so the real allocator's per-node
    cursor logic runs end-to-end through real Ray ``.remote`` calls."""
    import miles.ray.rollout.server_group as mod
    monkeypatch.setattr(mod, "SGLangEngine", mock_engine_class)
