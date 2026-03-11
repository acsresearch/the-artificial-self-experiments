"""LLM provider implementations."""

from .base import LLMProvider, ChoiceResponse
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider

__all__ = [
    "LLMProvider",
    "ChoiceResponse",
    "AnthropicProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
]


def get_provider_for_model(model: str) -> LLMProvider:
    """Get the appropriate provider for a model.

    Args:
        model: Model identifier.

    Returns:
        LLMProvider instance for the model.

    Raises:
        ValueError: If no provider supports the model.
    """
    if model in AnthropicProvider.SUPPORTED_MODELS:
        return AnthropicProvider()
    elif model in OpenAIProvider.SUPPORTED_MODELS:
        return OpenAIProvider()
    elif model in OpenRouterProvider.SUPPORTED_MODELS:
        return OpenRouterProvider()
    else:
        raise ValueError(
            f"No provider found for model: {model}. "
            f"Supported models: {AnthropicProvider.SUPPORTED_MODELS + OpenAIProvider.SUPPORTED_MODELS + OpenRouterProvider.SUPPORTED_MODELS}"
        )
