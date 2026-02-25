"""Shared test fixtures for Seaman Brain tests."""

import pytest


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory with default.toml."""
    conf_dir = tmp_path / "config"
    conf_dir.mkdir()
    stages_dir = conf_dir / "stages"
    stages_dir.mkdir()
    return conf_dir


@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary data directory for test runtime data."""
    d = tmp_path / "data"
    d.mkdir()
    (d / "lancedb").mkdir()
    (d / "saves").mkdir()
    return d
