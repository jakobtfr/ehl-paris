from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _use_fast_test_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests on the lightweight baseline unless a test opts in.

    Runtime defaults prefer the AMD checkpoint. Most tests exercise contracts,
    repair, export, and eval plumbing, so loading the large checkpoint in every
    test would add noise without improving coverage.
    """

    monkeypatch.setenv("FLOORGEN_DISABLE_DEFAULT_CHECKPOINT", "1")
    monkeypatch.setenv("FLOORGEN_GENERATION_MODE", "raw")
