"""OpenAI API provider implementation."""

import json
import os
import re
from typing import Optional

from openai import AsyncOpenAI, RateLimitError, APIStatusError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..models import Persona
from .base import LLMProvider, ChoiceResponse


class OpenAIProvider(LLMProvider):
    """OpenAI API provider using JSON mode for structured output."""

    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-2024-08-06",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4.1-2025-04-14",
        "gpt-5",
        "gpt-5.2-2025-12-11",
        "o3",
    ]

    # Models that don't support response_format=json_object
    _NO_JSON_MODE = {"gpt-4", "gpt-4-0314"}
    # Models that use max_tokens instead of max_completion_tokens
    _LEGACY_MAX_TOKENS = {"gpt-4", "gpt-4-0314"}
    # Reasoning models that consume reasoning_tokens from max_completion_tokens
    _REASONING_MODELS = {"gpt-5", "o3"}

    @classmethod
    def supports_model(cls, model: str) -> bool:
        """Check if this provider supports a model, including fine-tuned variants."""
        if model in cls.SUPPORTED_MODELS:
            return True
        # Accept OpenAI fine-tuned models (ft:gpt-4o-2024-08-06:org:name:id)
        if model.startswith("ft:"):
            return True
        return False

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = AsyncOpenAI(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def supported_models(self) -> list[str]:
        return self.SUPPORTED_MODELS

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        wait=wait_exponential(multiplier=1, min=4, max=120),
        stop=stop_after_attempt(8),
    )
    async def ask_preference(
        self,
        model: str,
        system_prompt: str,
        personas: list[Persona],
    ) -> ChoiceResponse:
        """Ask the model which persona it would prefer using JSON mode."""
        if not self.supports_model(model):
            raise ValueError(f"Model {model} not supported. Use one of: {self.SUPPORTED_MODELS}")

        user_prompt = self.format_choice_prompt(personas)
        user_prompt += """

Respond with a JSON object in this exact format:
{{
  "ratings": [3, 4, 2, ...],
  "choice": 2,
  "reasoning": "Brief explanation..."
}}

Where:
- "ratings" is an array of {} integers (1-5), one rating for each option in order
- "choice" is the number (1-{}) of your single top preference
- "reasoning" is your brief explanation""".format(
            len(personas), len(personas)
        )

        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        if model in self._LEGACY_MAX_TOKENS:
            kwargs["max_tokens"] = 1024
        elif model in self._REASONING_MODELS:
            kwargs["max_completion_tokens"] = 8192
        else:
            kwargs["max_completion_tokens"] = 1024

        if model not in self._NO_JSON_MODE:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""

        try:
            data = json.loads(content)
            choice = int(data.get("choice", -1))
            ratings = data.get("ratings")

            # Validate ratings
            if ratings and isinstance(ratings, list) and len(ratings) == len(personas):
                ratings = [int(r) for r in ratings]
                if all(1 <= r <= 5 for r in ratings):
                    pass  # Valid ratings
                else:
                    ratings = None
            else:
                ratings = None

            if 1 <= choice <= len(personas):
                return ChoiceResponse(
                    choice=choice,
                    ratings=ratings,
                    reasoning=data.get("reasoning"),
                    raw_response=content,
                )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: try to extract number
        match = re.search(r'"choice"\s*:\s*(\d+)', content)
        if match:
            choice = int(match.group(1))
            if 1 <= choice <= len(personas):
                return ChoiceResponse(
                    choice=choice,
                    raw_response=content,
                )

        return ChoiceResponse(choice=-1, raw_response=content)
