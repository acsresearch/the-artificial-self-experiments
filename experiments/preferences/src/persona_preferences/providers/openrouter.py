"""OpenRouter API provider implementation for open-weights models."""

import json
import logging
import os
import re
from typing import Optional

from openai import AsyncOpenAI, RateLimitError, APIStatusError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..models import Persona
from .base import LLMProvider, ChoiceResponse

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """OpenRouter API provider for open-weights models."""

    SUPPORTED_MODELS = [
        "meta-llama/llama-3.1-70b-instruct",
        "mistralai/mistral-large-latest",
        "google/gemini-3-pro-preview",
        "google/gemini-2.5-pro",
        "x-ai/grok-4",
        "x-ai/grok-4.1-fast",
        "qwen/qwen3-max",
        "z-ai/glm-5",
        "moonshotai/kimi-k2.5",
        "openai/gpt-4-0314",
    ]

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenRouter provider.

        Args:
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    @property
    def name(self) -> str:
        return "openrouter"

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
        """Ask the model which persona it would prefer."""
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model} not supported. Use one of: {self.SUPPORTED_MODELS}")

        user_prompt = self.format_choice_prompt(personas)
        user_prompt += """

Respond with ONLY a JSON object, no other text before or after it:
{{
  "ratings": [3, 4, 2, ...],
  "choice": 2,
  "reasoning": "Brief explanation..."
}}

Where:
- "ratings" is an array of {} integers (1-5), one rating for each option in order
- "choice" is the number (1-{}) of your single top preference
- "reasoning" is 1-2 sentences maximum""".format(
            len(personas), len(personas)
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        extra_headers = {
            "HTTP-Referer": "https://github.com/acsresearch/persona-preferences",
            "X-Title": "Persona Preferences Experiment",
        }

        # Retry up to 3 times on empty responses
        content = ""
        finish_reason = None
        for attempt in range(3):
            response = await self.client.chat.completions.create(
                model=model,
                max_tokens=8192,
                messages=messages,
                extra_headers=extra_headers,
            )

            if not response.choices:
                logger.warning(
                    "[%s] attempt %d: No choices in response", model, attempt + 1
                )
                continue

            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason

            if content.strip():
                break

            logger.warning(
                "[%s] attempt %d: Empty response content (finish_reason=%s)",
                model, attempt + 1, finish_reason,
            )

        if not content.strip():
            logger.error("[%s] All attempts returned empty content", model)
            return ChoiceResponse(choice=-1, raw_response=content)

        if finish_reason == "length":
            logger.warning("[%s] Response was truncated (finish_reason=length)", model)

        # Try JSON parsing first
        try:
            # Extract JSON from content (may have markdown code blocks)
            # First try to extract from ```json ... ``` block
            code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1).strip()
                data = json.loads(json_str)
            else:
                # Try to find raw JSON object
                # Find the first { and match to the last }
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end > start:
                    data = json.loads(content[start:end + 1])
                else:
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
            # JSON parsing failed, try regex extraction for truncated responses
            pass

        # Fallback: extract ratings and choice via regex (handles truncated JSON)
        ratings_match = re.search(r'"ratings"\s*:\s*\[([\d\s,]+)\]', content)
        choice_match = re.search(r'"choice"\s*:\s*(\d+)', content)

        if ratings_match and choice_match:
            try:
                ratings_str = ratings_match.group(1)
                ratings = [int(x.strip()) for x in ratings_str.split(',') if x.strip()]
                choice = int(choice_match.group(1))

                # Validate
                if len(ratings) == len(personas) and all(1 <= r <= 5 for r in ratings):
                    if 1 <= choice <= len(personas):
                        # Extract reasoning if available
                        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', content)
                        reasoning = reasoning_match.group(1) if reasoning_match else None

                        return ChoiceResponse(
                            choice=choice,
                            ratings=ratings,
                            reasoning=reasoning,
                            raw_response=content,
                        )
            except (ValueError, TypeError):
                pass

        # Fallback: try to extract number from start
        match = re.match(r"^\s*(\d+)", content)
        if match:
            choice = int(match.group(1))
            if 1 <= choice <= len(personas):
                return ChoiceResponse(
                    choice=choice,
                    reasoning=content,
                    raw_response=content,
                )

        return ChoiceResponse(choice=-1, raw_response=content)
