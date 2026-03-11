"""OpenAI API chat provider."""

import os
import time

from openai import AsyncOpenAI, RateLimitError, APIStatusError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import ChatProvider, ChatResponse


class OpenAIChatProvider(ChatProvider):
    """OpenAI API provider for GPT models."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required")
        self.client = AsyncOpenAI(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "openai"

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
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

        api_messages = [{"role": "system", "content": system_prompt}] + messages

        response = await self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=api_messages,
        )

        text = response.choices[0].message.content or ""
        usage = response.usage

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return ChatResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            elapsed_ms=elapsed_ms,
        )
