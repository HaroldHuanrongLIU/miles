"""rank ↔ addr_and_ports index consistency in ``ServerGroup.start_engines``.

- ``server_group.py:153-160`` builds init_handles by iterating ``new_engines``
  with ``(global_rank, engine)`` pairs.
- ``addr_allocator.py:85`` uses ``current_rank = rank + i`` as the dict key.

When ``rank_offset != 0`` or ``nodes_per_engine > 1``, the two index spaces
must agree: every ``engine.init.remote(...)`` must receive the kwargs
matching its global_rank, not its local-array index.

We verify this by intercepting the ``engine.init.remote`` invocations and
asserting the addr/port kwargs come from the right rank slot."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from miles.ray.rollout.addr_allocator import allocate_rollout_engine_addr_and_ports_normal
from tests.fast.ray.rollout.conftest import make_args


@pytest.fixture
def patch_ray_get(monkeypatch):
    import miles.ray.rollout.addr_allocator as mod
    monkeypatch.setattr(mod.ray, "get", lambda x: x)


def _fake_engine(host: str = "10.0.0.1", port_seed: int = 0):
    e = MagicMock()
    e._port_cursor = port_seed

    def _alloc(start_port: int = 15000, consecutive: int = 1):
        port = max(e._port_cursor, start_port)
        e._port_cursor = port + consecutive
        return (host, port)

    e._get_current_node_ip_and_free_port.remote.side_effect = lambda **kw: _alloc(**kw)
    return e


class TestRankPortConsistency:
    def test_rank_offset_kwargs_keyed_by_global_rank(self, patch_ray_get):
        """When rank_offset=4, the addr_and_ports dict must be keyed by
        ranks 4..7, not 0..3.

        Note: the allocator pads up to ``num_engines_per_node`` slots starting
        at the first rank seen on the node (see ``num_engines_on_this_node``
        in addr_allocator.py). To verify the keying-by-global-rank invariant
        without that padding, we set ``num_gpus_per_node=4`` so a 4-engine
        group exactly fills one node — no extra padded slots."""
        args = make_args(num_gpus_per_node=4, sglang_dp_size=1)
        # rollout_engines provided as list[(global_rank, engine)]
        engines = [(rank, _fake_engine()) for rank in range(4, 8)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1, rank_offset=4,
        )
        assert set(addr_and_ports.keys()) == {4, 5, 6, 7}, "must be keyed by global_rank"

    def test_each_global_rank_has_complete_kwargs(self, patch_ray_get):
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, _fake_engine()) for rank in range(4)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1,
        )
        for rank in range(4):
            kw = addr_and_ports[rank]
            assert "host" in kw, f"rank {rank} missing host"
            assert "port" in kw, f"rank {rank} missing port"
            assert "nccl_port" in kw, f"rank {rank} missing nccl_port"
            assert "dist_init_addr" in kw, f"rank {rank} missing dist_init_addr"

    def test_multinode_engine_shares_dist_init_addr_across_node_ranks(self, patch_ray_get):
        """nodes_per_engine=2 (16 gpus, 8 per node) — both ranks of one
        multi-node engine MUST get the same dist_init_addr."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        # 2 ranks, one engine spanning 16 gpus = 2 nodes
        engines = [(0, _fake_engine()), (1, _fake_engine())]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=16,
        )
        assert addr_and_ports[0]["dist_init_addr"] == addr_and_ports[1]["dist_init_addr"]

    def test_init_handles_iteration_pairs_match_addr_dict(self, patch_ray_get):
        """Mirror of the loop in server_group.py:153-160:
            init_handles = [
                engine.init.remote(**addr_and_ports[index])
                for index, engine in new_engines
            ]
        Verify that for every (index, engine) pair, addr_and_ports[index]
        contains all required kwargs."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        new_engines = [(rank, _fake_engine()) for rank in range(2, 6)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=new_engines, num_gpus_per_engine=1, rank_offset=2,
        )
        for index, engine in new_engines:
            assert index in addr_and_ports, f"missing addr_and_ports for global_rank={index}"
            engine.init.remote(**addr_and_ports[index])
            assert engine.init.remote.called

    def test_ports_are_unique_within_a_node(self, patch_ray_get):
        """All port values within one node's allocation must be unique."""
        args = make_args(num_gpus_per_node=8, sglang_dp_size=1)
        engines = [(rank, _fake_engine()) for rank in range(8)]
        addr_and_ports, _ = allocate_rollout_engine_addr_and_ports_normal(
            args=args, rollout_engines=engines, num_gpus_per_engine=1,
        )
        all_ports = []
        for rank, kw in addr_and_ports.items():
            all_ports.append(kw["port"])
            all_ports.append(kw["nccl_port"])
            all_ports.append(kw["engine_info_bootstrap_port"])
        assert len(set(all_ports)) == len(all_ports), f"duplicate ports: {all_ports}"
