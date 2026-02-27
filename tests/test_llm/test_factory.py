"""Tests for the LLM provider factory."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from seaman_brain.config import LLMConfig
from seaman_brain.llm.base import LLMProvider
from seaman_brain.llm.factory import create_provider

# --- Happy path tests ---


def test_create_ollama_provider():
    """Factory creates OllamaProvider for 'ollama' config."""
    config = LLMConfig(provider="ollama", model="qwen3-coder:30b")
    provider = create_provider(config)
    assert isinstance(provider, LLMProvider)
    assert provider.model == "qwen3-coder:30b"


def test_create_openai_provider():
    """Factory creates OpenAIProvider for 'openai' config."""
    config = LLMConfig(provider="openai", model="gpt-4o")
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = create_provider(config)
    assert isinstance(provider, LLMProvider)
    assert provider.model == "gpt-4o"


def test_create_anthropic_provider():
    """Factory creates AnthropicProvider for 'anthropic' config."""
    config = LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        provider = create_provider(config)
    assert isinstance(provider, LLMProvider)
    assert provider.model == "claude-sonnet-4-20250514"


def test_create_provider_default_config():
    """Factory uses default config (ollama) when None is passed."""
    provider = create_provider(None)
    assert isinstance(provider, LLMProvider)
    assert provider.model == "qwen3-coder:30b"


# --- Edge case tests ---


def test_provider_name_case_insensitive():
    """Factory handles uppercase/mixed-case provider names."""
    config = LLMConfig(provider="Ollama", model="test-model")
    provider = create_provider(config)
    assert isinstance(provider, LLMProvider)


def test_config_values_passed_through():
    """Factory passes config values to the provider instance."""
    config = LLMConfig(
        provider="ollama",
        model="custom-model",
        temperature=0.5,
        max_tokens=1024,
        base_url="http://custom:11434",
    )
    provider = create_provider(config)
    assert provider.model == "custom-model"
    assert provider.temperature == 0.5
    assert provider.base_url == "http://custom:11434"


# --- Error handling tests ---


def test_unknown_provider_raises_value_error():
    """Factory raises ValueError for unknown provider names."""
    config = LLMConfig(provider="unknown_provider")
    with pytest.raises(ValueError, match="Unknown LLM provider: 'unknown_provider'"):
        create_provider(config)


def test_unknown_provider_lists_supported():
    """Error message lists supported providers."""
    config = LLMConfig(provider="gemini")
    with pytest.raises(ValueError, match="ollama, openai, anthropic"):
        create_provider(config)


def _block_import(module_key: str):
    """Create a context manager that blocks importing a specific provider module.

    Removes the cached module from sys.modules and patches builtins.__import__
    to raise ImportError for the target module.
    """
    original_import = __import__

    def _blocked(name, *args, **kwargs):
        if name == module_key:
            raise ImportError(f"No module named '{module_key}'")
        return original_import(name, *args, **kwargs)

    return _blocked, module_key


def test_ollama_import_error():
    """Factory raises ImportError when ollama package is missing."""
    config = LLMConfig(provider="ollama")
    blocked_fn, mod = _block_import("seaman_brain.llm.ollama_provider")
    cached = sys.modules.pop("seaman_brain.llm.ollama_provider", None)
    try:
        with patch("builtins.__import__", side_effect=blocked_fn):
            with pytest.raises(ImportError, match="ollama"):
                create_provider(config)
    finally:
        if cached is not None:
            sys.modules["seaman_brain.llm.ollama_provider"] = cached


def test_openai_import_error():
    """Factory raises ImportError when openai package is missing."""
    config = LLMConfig(provider="openai")
    blocked_fn, mod = _block_import("seaman_brain.llm.openai_provider")
    cached = sys.modules.pop("seaman_brain.llm.openai_provider", None)
    try:
        with patch("builtins.__import__", side_effect=blocked_fn):
            with pytest.raises(ImportError, match="openai"):
                create_provider(config)
    finally:
        if cached is not None:
            sys.modules["seaman_brain.llm.openai_provider"] = cached


def test_anthropic_import_error():
    """Factory raises ImportError when anthropic package is missing."""
    config = LLMConfig(provider="anthropic")
    blocked_fn, mod = _block_import("seaman_brain.llm.anthropic_provider")
    cached = sys.modules.pop("seaman_brain.llm.anthropic_provider", None)
    try:
        with patch("builtins.__import__", side_effect=blocked_fn):
            with pytest.raises(ImportError, match="anthropic"):
                create_provider(config)
    finally:
        if cached is not None:
            sys.modules["seaman_brain.llm.anthropic_provider"] = cached
