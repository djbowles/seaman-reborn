"""LLM provider factory.

Creates the appropriate LLMProvider implementation based on config settings.
Cloud providers (openai, anthropic) require optional dependencies.
"""

from __future__ import annotations

import logging

from seaman_brain.config import LLMConfig
from seaman_brain.llm.base import LLMProvider

logger = logging.getLogger(__name__)


def create_provider(
    config: LLMConfig | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """Create an LLM provider instance from configuration.

    Selects the provider implementation based on config.provider string
    and returns a configured instance.

    Args:
        config: LLM configuration. Uses defaults if None.
        api_key: Optional API key override for cloud providers.

    Returns:
        A configured LLMProvider instance.

    Raises:
        ValueError: If the provider name is unknown.
        ImportError: If the required provider package is not installed.
    """
    cfg = config or LLMConfig()
    provider_name = cfg.provider.lower()

    if provider_name == "ollama":
        try:
            from seaman_brain.llm.ollama_provider import OllamaProvider
        except ImportError as exc:
            raise ImportError(
                "Ollama provider requires the 'ollama' package. "
                "Install it with: pip install ollama"
            ) from exc
        return OllamaProvider(config=cfg)

    if provider_name == "openai":
        try:
            from seaman_brain.llm.openai_provider import OpenAIProvider
        except ImportError as exc:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. "
                "Install it with: pip install -e \".[cloud]\""
            ) from exc
        return OpenAIProvider(config=cfg, api_key=api_key)

    if provider_name == "anthropic":
        try:
            from seaman_brain.llm.anthropic_provider import AnthropicProvider
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install it with: pip install -e \".[cloud]\""
            ) from exc
        return AnthropicProvider(config=cfg, api_key=api_key)

    raise ValueError(
        f"Unknown LLM provider: '{cfg.provider}'. "
        f"Supported providers: ollama, openai, anthropic"
    )


def create_cloud_provider(config: LLMConfig) -> LLMProvider | None:
    """Try to create a cloud LLM provider from the routing config fields.

    Uses config.cloud_provider and config.cloud_model to create an Anthropic
    or OpenAI provider. Returns None (with a warning) if the provider cannot
    be created (missing API key, missing package, etc.).

    Args:
        config: LLM configuration with cloud_provider and cloud_model set.

    Returns:
        A configured cloud LLMProvider, or None if creation failed.
    """
    cloud_name = config.cloud_provider.lower().strip()
    if not cloud_name:
        return None

    # Build a config for the cloud provider
    cloud_cfg = LLMConfig(
        provider=cloud_name,
        model=config.cloud_model or _default_cloud_model(cloud_name),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    api_key = config.cloud_api_key or None  # empty string -> None (use env var)

    try:
        provider = create_provider(cloud_cfg, api_key=api_key)
        logger.info(
            "Cloud LLM provider created: %s (model=%s)",
            cloud_name, cloud_cfg.model,
        )
        return provider
    except (ImportError, ValueError) as exc:
        logger.warning("Failed to create cloud provider '%s': %s", cloud_name, exc)
        return None


def _default_cloud_model(provider: str) -> str:
    """Return a sensible default model for a cloud provider.

    Args:
        provider: The cloud provider name.

    Returns:
        A default model string.
    """
    defaults = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
    }
    return defaults.get(provider, "")
