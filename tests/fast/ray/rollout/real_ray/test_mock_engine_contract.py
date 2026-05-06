"""Contract test: prevents MockSGLangEngine from drifting away from the
real SGLangEngine API used by miles/ray/rollout/.

Without this test, every other lifecycle test ends up testing the mock
itself rather than the real code path."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from miles.backends.sglang_utils.sglang_engine import SGLangEngine
from tests.fast.ray.rollout.lifecycle.mock_engine import MockSGLangEngine


ROLLOUT_DIR = Path(__file__).resolve().parents[5] / "miles" / "ray" / "rollout"


def _grep_engine_method_calls(directory: Path) -> set[str]:
    """Find every ``<engine|actor_handle>.<method>.remote(...)`` call in the
    rollout dir. The set returned is method names that the rollout code
    expects to exist on every SGLangEngine actor."""
    pattern = re.compile(r"(?:engine|actor_handle|rollout_engine)\.([a-zA-Z_][a-zA-Z0-9_]*)\.remote\(")
    methods: set[str] = set()
    for py in directory.rglob("*.py"):
        for m in pattern.finditer(py.read_text()):
            methods.add(m.group(1))
    return methods


def _public_methods(cls) -> set[str]:
    """Public methods (and known semi-private helpers) of ``cls``."""
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__  # unwrap @ray.remote
    keep_underscored = {"_get_current_node_ip_and_free_port"}
    return {
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("__") and (not name.startswith("_") or name in keep_underscored)
    }


@pytest.fixture(scope="module")
def used_methods() -> set[str]:
    used = _grep_engine_method_calls(ROLLOUT_DIR)
    assert used, f"Expected to find engine.<method>.remote(...) calls under {ROLLOUT_DIR}"
    return used


def test_mock_implements_every_method_used_in_rollout_dir(used_methods: set[str]) -> None:
    real_methods = _public_methods(SGLangEngine)
    mock_methods = _public_methods(MockSGLangEngine)

    must_have = used_methods & real_methods
    missing_on_mock = must_have - mock_methods
    assert not missing_on_mock, (
        f"MockSGLangEngine is missing real-API methods that are called in "
        f"miles/ray/rollout/: {sorted(missing_on_mock)}. "
        f"Add stub implementations to mock_engine.py before adding the dependent test."
    )


def test_mock_does_not_invent_methods_outside_real_api(used_methods: set[str]) -> None:
    """Inverse: mock should not pretend to implement methods that don't exist
    on the real engine. Only enforced for methods actually exercised by the
    rollout dir, since mock is allowed to expose helpers like ``set_fault``."""
    real_methods = _public_methods(SGLangEngine)
    mock_methods = _public_methods(MockSGLangEngine)

    invented = (mock_methods & used_methods) - real_methods
    assert not invented, (
        f"MockSGLangEngine declares methods that are called by rollout code but "
        f"do not exist on the real SGLangEngine: {sorted(invented)}. "
        f"This will produce false positives — the mock test passes but real code "
        f"would AttributeError."
    )


def test_signature_compat_for_init() -> None:
    """``init`` is the most important signature to keep aligned because the
    rollout code passes addr/port kwargs that come from addr_allocator."""
    real_cls = SGLangEngine
    mock_cls = MockSGLangEngine.__ray_actor_class__

    real_sig = inspect.signature(real_cls.init)
    mock_sig = inspect.signature(mock_cls.init)
    real_params = set(real_sig.parameters) - {"self"}
    mock_params = set(mock_sig.parameters) - {"self"}

    # Mock accepts **kwargs catch-all; real signature lists explicit params.
    # We only require that mock does not drop anything the real one declares.
    if "kwargs" not in mock_params:
        missing = real_params - mock_params
        assert not missing, f"MockSGLangEngine.init drops real params: {sorted(missing)}"
