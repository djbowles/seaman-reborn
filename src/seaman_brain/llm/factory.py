"""LLM provider factory.

Creates the appropriate LLMProvider implementation based on config settings.
Cloud providers (openai, anthropic) require optional dependencies.
"""

from __future__ import annotations

from seaman_brain.config import LLMConfig
from seaman_brain.llm.base import LLMProvider


def create_provider(config: LLMConfig | None = None) -> LLMProvider:
    """Create an LLM provider instance from configuration.

    Selects the provider implementation based on config.provider string
    and returns a configured instance.

    Args:
        config: LLM configuration. Uses defaults if None.

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
        return OpenAIProvider(config=cfg)

    if provider_name == "anthropic":
        try:
            from seaman_brain.llm.anthropic_provider import AnthropicProvider
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install it with: pip install -e \".[cloud]\""
            ) from exc
        return AnthropicProvider(config=cfg)

    raise ValueError(
        f"Unknown LLM provider: '{cfg.provider}'. "
        f"Supported providers: ollama, openai, anthropic"
    )
