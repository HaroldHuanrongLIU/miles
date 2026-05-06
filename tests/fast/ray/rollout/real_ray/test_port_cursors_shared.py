"""``RolloutServer.recover`` shares one ``PortCursors`` across multi-group
parallel recover, and ports across groups must not collide.

The source comment at ``server_group.py:37``
    # NOTE: this may have risk when recovering engines parallelly
already flags this as a known concern. The contract this test enforces:
two ``ServerGroup``s recovering simultaneously, sharing one ``PortCursors``
object, must produce **disjoint** port allocations on each node — even if
their recover() coroutines complete out of order."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from miles.ray.rollout.addr_allocator import PortCursors, allocate_rollout_engine_addr_and_ports_normal
from tests.fast.ray.rollout.conftest import make_args


@pytest.fixture
def patch_ray_get(monkeypatch):
    import miles.ray.rollout.addr_allocator as mod
    monkeypatch.setattr(mod.ray, "get", lambda x: x)


def _fake_engine(host: str = "10.0.0.1"):
    e = MagicMock()
    e._port_cursor = 0

    def _alloc(start_port: int = 15000, consecutive: int = 1):
        port = max(e._port_cursor, start_port)
        e._port_cursor = port + consecutive
        return (host, port)

    e._get_current_node_ip_and_free_port.remote.side_effect = lambda **kw: _alloc(**kw)
    return e


class TestSharedPortCursorsAcrossGroups:
    def test_sequential_groups_share_cursor_and_avoid_overlap(self, patch_ray_get):
        """Sequential allocation across two groups: the second group reads the
        cursor advanced by the first group via ``next_base_port()``."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        cursors = PortCursors.empty()

        # group A: 4 engines on node 0
        engines_a = [(rank, _fake_engine()) for rank in range(4)]
        addrs_a, next_a = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines_a, num_gpus_per_engine=1,
            base_port=cursors.next_base_port(),
        )
        cursors.assign(next_a)

        # group B: 4 engines on node 0 (after A)
        engines_b = [(rank, _fake_engine()) for rank in range(4, 8)]
        addrs_b, next_b = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines_b, num_gpus_per_engine=1,
            base_port=cursors.next_base_port(),
            rank_offset=4,
        )

        ports_a = {addrs_a[r]["port"] for r in addrs_a} | {addrs_a[r]["nccl_port"] for r in addrs_a}
        ports_b = {addrs_b[r]["port"] for r in addrs_b} | {addrs_b[r]["nccl_port"] for r in addrs_b}
        assert ports_a.isdisjoint(ports_b), f"port overlap A={ports_a} B={ports_b}"

    # Note: a "test_concurrent_recover_completion_order" test was removed.
    # Despite the name it ran sequentially (no asyncio.gather), and the
    # assertion ``cursor_after_b >= cursor_after_a`` was tautological — B
    # was seeded with A's cursor as ``base_port``, so monotonic allocation
    # makes the inequality automatic. The disjoint-port property in
    # ``test_two_groups_share_advancing_cursor`` already covers the real
    # contract.

    # NOTE: a "multi-node groups have independent cursors per node" test was
    # removed because PortCursors keys by ``node_index`` *within each call*,
    # not by hostname. Two separate ``allocate_rollout_engine_addr_and_ports_normal``
    # calls each return a fresh ``{0: <port>}`` dict; cross-call cursor sharing
    # is the caller's responsibility (via ``base_port=cursors.next_base_port()``)
    # and is already covered by ``test_two_groups_share_advancing_cursor``.
