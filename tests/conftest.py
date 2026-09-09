"""Shared pytest fixtures and helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
STUDY_DATA = (
    Path(__file__).parent.parent
    / "research"
    / "kv-cache-reasoning-divergence-study"
    / "data"
)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def study_data_dir() -> Path:
    """Real data from the completed study — used as ground-truth regression
    fixtures, not synthetic data. Tests using this should skip if it's absent
    (e.g. a checkout that excludes research/) rather than fail."""
    if not STUDY_DATA.exists():
        pytest.skip(f"study data not present at {STUDY_DATA}")
    return STUDY_DATA


@pytest.fixture
def aime_tiny(fixtures_dir) -> list[dict]:
    with (fixtures_dir / "aime_tiny.json").open() as f:
        return json.load(f)


@pytest.fixture
def math500_tiny(fixtures_dir) -> list[dict]:
    with (fixtures_dir / "math500_tiny.json").open() as f:
        return json.load(f)


@pytest.fixture
def baseline_trace_text(fixtures_dir) -> str:
    return (fixtures_dir / "trace_baseline_sample.txt").read_text(encoding="utf-8")


@pytest.fixture
def quant_trace_text(fixtures_dir) -> str:
    return (fixtures_dir / "trace_quant_sample.txt").read_text(encoding="utf-8")
