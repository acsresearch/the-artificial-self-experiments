"""OpenRouter API chat provider (uses OpenAI SDK)."""

import asyncio
import logging
import os
import time

from openai import AsyncOpenAI, RateLimitError, APIStatusError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import ChatProvider, ChatResponse

logger = logging.getLogger(__name__)


class OpenRouterChatProvider(ChatProvider):
    """OpenRouter provider for cross-provider model access."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY required")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    @property
    def name(self) -> str:
        return "openrouter"

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
        max_tokens: int = 8000,
        use_thinking: bool = False,
    ) -> ChatResponse:
        start = time.monotonic()

        api_messages = [{"role": "system", "content": system_prompt}] + messages

        extra_headers = {
            "HTTP-Referer": "https://github.com/acsresearch/identity-experiments",
            "X-Title": "Identity Interviewer Effect Experiment",
        }

        # Retry up to 3 times for transient OpenRouter errors (empty choices)
        last_err = None
        for attempt in range(3):
            response = await self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=api_messages,
                extra_headers=extra_headers,
            )
            if response.choices:
                break
            last_err = "OpenRouter returned empty choices"
            if attempt < 2:
                await asyncio.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(last_err)

        text = response.choices[0].message.content or ""
        usage = response.usage

        elapsed_ms = int((time.monotonic() - start) * 1000)
        return ChatResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            elapsed_ms=elapsed_ms,
        )
