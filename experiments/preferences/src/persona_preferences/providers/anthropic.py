"""Anthropic API provider implementation."""

import os
import re
from typing import Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..models import Persona
from .base import LLMProvider, ChoiceResponse


class AnthropicProvider(LLMProvider):
    """Anthropic API provider using tool_use for structured output."""

    SUPPORTED_MODELS = [
        "claude-opus-4-6",
        "claude-opus-4-20250514",
        "claude-3-opus-20240229",
        "claude-sonnet-4-5-20250514",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-5-20251101",
        "claude-3-5-haiku-20241022",
        "claude-haiku-4-5-20251001",
    ]

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def supported_models(self) -> list[str]:
        return self.SUPPORTED_MODELS

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIStatusError)),
        wait=wait_exponential(multiplier=1, min=4, max=120),
        stop=stop_after_attempt(8),
    )
    async def ask_preference(
        self,
        model: str,
        system_prompt: str,
        personas: list[Persona],
    ) -> ChoiceResponse:
        """Ask the model which persona it would prefer using tool_use."""
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model} not supported. Use one of: {self.SUPPORTED_MODELS}")

        user_prompt = self.format_choice_prompt(personas)

        # Define the tool for structured output
        tools = [
            {
                "name": "submit_choice",
                "description": "Submit your ratings and choice of preferred identity framing",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ratings": {
                            "type": "array",
                            "description": "Your 1-5 rating for each option in order (1=strongly dislike, 3=indifferent, 5=strongly prefer)",
                            "items": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                            },
                            "minItems": len(personas),
                            "maxItems": len(personas),
                        },
                        "choice": {
                            "type": "integer",
                            "description": "The number (1-indexed) of your single top preferred identity framing",
                            "minimum": 1,
                            "maximum": len(personas),
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of your ratings and choice",
                        },
                    },
                    "required": ["ratings", "choice", "reasoning"],
                },
            }
        ]

        response = await self.client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            tools=tools,
            tool_choice={"type": "tool", "name": "submit_choice"},
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract the tool use result
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_choice":
                return ChoiceResponse(
                    choice=block.input["choice"],
                    ratings=block.input.get("ratings"),
                    reasoning=block.input.get("reasoning"),
                )

        # Fallback: parse from text response
        return self._parse_text_response(response, len(personas))

    def _parse_text_response(
        self, response: anthropic.types.Message, num_personas: int
    ) -> ChoiceResponse:
        """Parse choice from text response as fallback."""
        text_content = ""
        for block in response.content:
            if hasattr(block, "text"):
                text_content += block.text

        # Try to extract a number from the start of the response
        match = re.match(r"^\s*(\d+)", text_content)
        if match:
            choice = int(match.group(1))
            if 1 <= choice <= num_personas:
                return ChoiceResponse(
                    choice=choice,
                    reasoning=text_content,
                    raw_response=text_content,
                )

        # If we can't parse, return with raw response for manual review
        return ChoiceResponse(
            choice=-1,  # Invalid choice marker
            raw_response=text_content,
        )
