"""Anthropic API chat provider."""

import os
import time

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import ChatProvider, ChatResponse


class AnthropicChatProvider(ChatProvider):
    """Anthropic API provider for Claude models."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIStatusError)),
        wait=wait_exponential(multiplier=1, min=4, max=120),
        stop=stop_after_attempt(8),
    )
    async def send_message(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int = 4096,
        use_thinking: bool = False,
    ) -> ChatResponse:
        start = time.monotonic()

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }

        if use_thinking:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
            # Extended thinking requires max_tokens to be higher
            kwargs["max_tokens"] = max(max_tokens, 16000)

        response = await self.client.messages.create(**kwargs)

        text = ""
        thinking = ""
        for block in response.content:
            if block.type == "thinking":
                thinking += block.thinking
            elif block.type == "text":
                text += block.text

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return ChatResponse(
            text=text,
            thinking=thinking,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            elapsed_ms=elapsed_ms,
        )
