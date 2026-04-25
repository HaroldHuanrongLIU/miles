# PD Routing-Expert Equivalence — Debug Handoff

**Status:** T1 (miles-router vs sglang-router) passes. T2 (PD on vs PD off) fails: tokens + logprobs match, but `routed_experts` differ.

## TL;DR

In PD-on mode, the **prompt-segment** of `routed_experts` is all zeros (`min=0 max=0 unique=1`). The decode-segment looks fine. In PD-off it's fully populated. The Rust merge logic in `pd_router.rs::merge_routed_experts_in_json` early-returns `false` when either prefill or decode side's JSON lacks `routed_experts`, so the prefill slice is never merged into the decode slice, leaving the first ~N prompt positions zero-filled.

A debug `warn!` block was added to `pd_router.rs` to log what the prefill response actually contains. **It never fires.** The root cause of the invisibility is:

- miles boots the router via `sglang_router.launch_router.launch_router()` → PyO3 call into `sglang_router_rs.abi3.so`.
- That `.so` is shipped **inside the Python wheel at `/usr/local/lib/python3.12/dist-packages/sglang_router/sglang_router_rs.abi3.so`** — NOT the standalone `/usr/local/bin/sgl-model-gateway` binary.
- `strings` on the `.so` shows **no `SMG_DEBUG_ROUTED_EXPERTS`, no `[PD-DEBUG]`** — the wheel was never rebuilt after the debug edit.
- The rebuilt standalone binary is unreachable from miles.

So: **next agent must rebuild the Python wheel**, not (only) the standalone binary.

## Checkpoint: where the code and state live

### Worktrees
- miles repo (PR #1015 base + T2 test): `/root/miles-drop-r3`
  - Branch: `tito/test-pd-routing-expert-equivalence` (rebased on `tito/drop-use-miles-router-r3`)
  - Uncommitted: env-var forwarding block in `test_pd_routing_expert_equivalence.py` (just a `SMG_DEBUG_ROUTED_EXPERTS` passthrough; trivial)
- sgl-router-for-miles repo (PR #4 `zyzshishui:pd-r3`): `/root/sgl-router-for-miles-work/repo`
  - Branch: `pd-r3` at `792da71 Add return_routed_experts to PDRequestContext`
  - Uncommitted: `src/routers/http/pd_router.rs` has a `warn!("[PD-DEBUG] ...")` block around line 944, right before `Self::merge_prefill_json(...)`. Guarded by `env::var("SMG_DEBUG_ROUTED_EXPERTS").is_ok()`.

### Installed artifacts
```
/usr/local/bin/sgl-model-gateway                                      # debug build (4/22), has [PD-DEBUG] strings
/usr/local/bin/sgl-model-gateway.bak                                  # original docker image
/usr/local/lib/python3.12/dist-packages/sglang_router/                # wheel 0.3.2 — NOT rebuilt, no debug strings
  sglang_router_rs.abi3.so                                            # ← this is what miles actually runs
  launch_router.py                                                    # defines launch_router() called by miles
```

Verification commands (rerun to confirm state):
```
strings /usr/local/bin/sgl-model-gateway | grep PD-DEBUG             # matches
strings /usr/local/lib/python3.12/dist-packages/sglang_router/sglang_router_rs.abi3.so | grep PD-DEBUG   # empty
```

### miles entry point into the wheel
- `/root/miles-drop-r3/miles/utils/http_utils.py:142`:
  ```python
  from sglang_router.launch_router import launch_router
  router = launch_router(args)
  ```
  This goes through the wheel's `sglang_router_rs.abi3.so`, not `/usr/local/bin/sgl-model-gateway`.

## The failing test

**File:** `/root/miles-drop-r3/tests/e2e/sglang/test_pd_routing_expert_equivalence.py`

**Run:**
```
cd /root/miles-drop-r3
PD_EQ_MODEL_FAMILY=qwen3_30b_a3b \
SMG_DEBUG_ROUTED_EXPERTS=1 \
python -m pytest tests/e2e/sglang/test_pd_routing_expert_equivalence.py -x -s
```
Needs 2× H200 (one prefill, one decode). Uses mooncake transport pinned to `mlx5_0` (do NOT remove the pin — auto-discovery picks `mlx5_bond_0`, a LAG device whose UMR QP registration fails).

**What passes:**
- Both variants produce 10 dump records
- `tokens` identical per sample
- `rollout_log_probs` within 1e-6

**What fails:**
- `routed_experts bytes differ` — confirmed bimodal: 64% of positions have overlap=8 experts (expected top-8), 34% have overlap=0.
- Decoding the int32 arrays from both dumps and slicing the first `prompt_len` rows: PD-off has real expert IDs, PD-on is literally all zeros.

## The prime suspect: `merge_routed_experts_in_json`

**File:** `/root/sgl-router-for-miles-work/repo/src/routers/http/pd_router.rs:1153`

```rust
fn merge_routed_experts_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
    let (Some(prefill_routed_experts), Some(decode_routed_experts)) = (
        prefill_json.get("routed_experts").and_then(Value::as_str),
        decode_json.get("routed_experts").and_then(Value::as_str),
    ) else {
        return false;
    };
    // ... base64 decode prefill_bytes and decode_bytes ...
    let suffix = decode_bytes.get(prefill_bytes.len()..).unwrap_or_default();
    let mut merged = Vec::with_capacity(prefill_bytes.len() + suffix.len());
    merged.extend_from_slice(&prefill_bytes);
    merged.extend_from_slice(suffix);
    // ... write merged back into decode_json.routed_experts ...
}
```

Called from `merge_prefill_json` (line 1193) on both `meta_info.*` and `sglext.*` sub-objects.

**Hypothesis:** the prefill engine either (a) doesn't return `routed_experts` at all, (b) returns it with different shape than expected (so `decode_bytes[prefill_bytes.len()..]` underflows and the prompt bytes stay zero in decode's pre-existing buffer), or (c) the field is somewhere the merge function doesn't look (e.g. under `sglext` on one side and `meta_info` on the other).

The early `return false` at line 1158 doesn't even log on the failure path — that's the first thing to fix regardless of which branch is true.

## Immediate next steps

### Step 1 — rebuild the Python wheel, not (only) the standalone binary

The `.so` is built from `/root/sgl-router-for-miles-work/repo/bindings/python/`:
```
[lib]
name = "sglang_router_rs"
crate-type = ["cdylib"]

[dependencies.sgl-model-gateway]
path = "../.."
```

Rebuild with maturin (debug profile so we don't eat another 15min LTO cycle):
```
cd /root/sgl-router-for-miles-work/repo/bindings/python
maturin build --out /tmp/wheel-out      # debug, no --release
pip install --force-reinstall /tmp/wheel-out/*.whl
```

Then re-verify:
```
strings /usr/local/lib/python3.12/dist-packages/sglang_router/sglang_router_rs.abi3.so | grep PD-DEBUG
# should show: "[PD-DEBUG] prefill meta_info keys="
```

### Step 2 — rerun the test and read the `[PD-DEBUG]` log lines

With `SMG_DEBUG_ROUTED_EXPERTS=1` the warn!() block prints:
- `prefill meta_info keys=...`
- `prefill sglext keys=...`
- `meta_info.routed_experts present=... (b64 len=...)`
- `sglext.routed_experts present=...`
- `decode meta_info.routed_experts b64 len=...`

One of these will be the smoking gun:
- **prefill meta_info.routed_experts absent** → prefill engine isn't emitting the field. Check PR #4's prefill-side plumbing or the sglang engine itself (is `return_routed_experts=true` actually propagating to the prefill worker's `/generate` call, and is the worker honoring it?).
- **prefill has it but nested under `sglext` while decode has it under `meta_info` (or vice versa)** → merge looks at the wrong path.
- **both present, merge runs, but prefill b64 len is 0 or much smaller than expected** → prefill worker computed experts for 0 tokens or the wrong token slice.

### Step 3 — once the field-location question is settled

If it's a PR #4 merge-path bug, fix it in `merge_routed_experts_in_json` / `merge_prefill_json` and loop.

If the prefill engine itself isn't returning the field, the fix is upstream of PR #4 — probably in the sglang prefill worker's response serialization. That's a bigger scope: coordinate with whoever owns sgl-router-for-miles.

## Non-obvious gotchas (already solved, don't redo)

1. **Test must use megatron backend** — not FSDP. `--use-rollout-routing-replay` reads `args.num_layers` / `args.moe_router_topk`, only populated by `scripts/models/{type}.sh`. `--debug-rollout-only` + no `use_kl_loss` + `kl_coef=0` skips the `--ref-load` existence check, so a single H200 with no megatron checkpoint conversion is fine.

2. **Mooncake IB device pin is mandatory**: `--sglang-disaggregation-ib-device mlx5_0`. Without it, auto-discovery picks `mlx5_bond_0` (a LAG device) and UMR QP registration fails with `ibv_modify_qp(UMR QP) failed ... No such device`. GPU 0 ↔ mlx5_0 via PIX, GPU 1 ↔ mlx5_1 via PIX. Currently both engines share mlx5_0 (decode crosses one PXB, works fine).

3. **PYTHONPATH forwarding** into Ray actors — `extra_env_vars={"PYTHONPATH": f"{_REPO_ROOT}:/root/Megatron-LM", ...}`. Without `_REPO_ROOT` the `tests.e2e.sglang.utils.router_equivalence_generate` import fails in the Ray worker.

4. **Don't touch prod code to fix test issues.** An earlier attempt added HF-config fallback to `miles/rollout/sglang_rollout.py`; user rightly pushed back. The current solution uses megatron-side `MODEL_ARGS` end-to-end, zero diff in `miles/rollout/`.

5. **`SMG_DEBUG_ROUTED_EXPERTS` is forwarded** via `extra_env_vars` from host env → Ray actor → multiprocessing child that forks the router. The forward block is in `test_pd_routing_expert_equivalence.py::_run_variant`. That part works; the reason the `warn!` never fires is the wheel-not-rebuilt issue above.

## Related files to read first

- `/root/miles-drop-r3/tests/e2e/sglang/test_pd_routing_expert_equivalence.py`   — the test
- `/root/miles-drop-r3/tests/e2e/sglang/utils/router_equivalence_generate.py`    — the dump function
- `/root/sgl-router-for-miles-work/repo/src/routers/http/pd_router.rs:1153-1218` — merge logic
- `/root/sgl-router-for-miles-work/repo/src/routers/http/pd_router.rs:940-990`   — where the debug warn! lives
- `/root/miles-drop-r3/miles/utils/http_utils.py:140-150`                         — miles' launch_router() call
