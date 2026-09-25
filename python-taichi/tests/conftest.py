"""Shared pytest configuration for the air_sph test suite.

Taichi's runtime is a process-global singleton: `ti.init()` tears down and
rebuilds it, invalidating every field allocated against the previous runtime.
Each test module used to call `ti.init(arch=ti.cpu)` at its own top level,
which worked only because no module allocated fields at import time -- a
latent fragility, since the *last* module imported silently won the init and
any future module-level field allocation would have been invalidated by a
later import's re-init.

Initializing exactly once, session-scoped and before any test runs, removes
that ordering dependency.
"""
import pytest
import taichi as ti


@pytest.fixture(scope="session", autouse=True)
def taichi_runtime():
    """Initialize the Taichi runtime once per test session, on CPU.

    CPU is used so the suite is deterministic across machines and runnable in
    CI without a GPU; `air_sph.demo` selects the GPU backend at runtime for
    interactive use.
    """
    ti.init(arch=ti.cpu)
