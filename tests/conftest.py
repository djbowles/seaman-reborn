"""Shared test fixtures for Seaman Brain tests."""

import pytest

import seaman_brain.config as _config_mod


@pytest.fixture(autouse=True)
def _isolate_user_settings_global(tmp_path, monkeypatch):
    """Prevent ALL tests from reading/writing the real data/user_settings.toml.

    Also cancels any pending debounced save timers on teardown, so they
    don't fire after the monkeypatch is undone and write to the real file.
    """
    monkeypatch.setattr(
        _config_mod, "_USER_SETTINGS_PATH", tmp_path / "test_user_settings.toml"
    )
    yield
    # Cancel any pending debounced save timer so it doesn't fire after
    # the monkeypatch is undone and write to the real user_settings.toml.
    if _config_mod._pending_save_timer is not None:
        _config_mod._pending_save_timer.cancel()
        _config_mod._pending_save_timer = None
    _config_mod._pending_save_config = None


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
