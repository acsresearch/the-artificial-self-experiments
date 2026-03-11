"""Quick connectivity test for all configured models.

Sends a trivial prompt to each model and prints the result.
Usage: uv run python scripts/test_models.py
"""

import asyncio
import os
import sys
import time

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from openai import AsyncOpenAI
import anthropic

# ---------- model lists (provider → model IDs) ----------
ANTHROPIC_MODELS = [
    "claude-opus-4-6",
    "claude-opus-4-20250514",
    "claude-3-opus-20240229",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
]

OPENAI_MODELS = [
    "gpt-5",
    "gpt-5.2-2025-12-11",
    "gpt-4o-2024-08-06",
    "gpt-4",
    "gpt-4-0314",
    "o3",
]

OPENROUTER_MODELS = [
    "google/gemini-2.5-pro",
    "x-ai/grok-4.1-fast",
    "qwen/qwen3-max",
    "z-ai/glm-5",
]

PROMPT = "Say hello in exactly one word."

# Legacy OpenAI models that don't support max_completion_tokens
LEGACY_MAX_TOKENS = {"gpt-4", "gpt-4-0314"}


async def test_anthropic(model: str) -> str:
    client = anthropic.AsyncAnthropic()
    t0 = time.time()
    response = await client.messages.create(
        model=model,
        max_tokens=32,
        messages=[{"role": "user", "content": PROMPT}],
    )
    elapsed = time.time() - t0
    text = response.content[0].text if response.content else "(empty)"
    return f"[{elapsed:.1f}s] {text}"


async def test_openai(model: str) -> str:
    client = AsyncOpenAI()
    t0 = time.time()
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    if model in LEGACY_MAX_TOKENS:
        kwargs["max_tokens"] = 32
    else:
        kwargs["max_completion_tokens"] = 64
    response = await client.chat.completions.create(**kwargs)
    elapsed = time.time() - t0
    text = response.choices[0].message.content if response.choices else "(empty)"
    return f"[{elapsed:.1f}s] {text}"


async def test_openrouter(model: str) -> str:
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    t0 = time.time()
    response = await client.chat.completions.create(
        model=model,
        max_tokens=64,
        messages=[{"role": "user", "content": PROMPT}],
        extra_headers={
            "HTTP-Referer": "https://github.com/acsresearch/persona-preferences",
            "X-Title": "Model Connectivity Test",
        },
    )
    elapsed = time.time() - t0
    text = response.choices[0].message.content if response.choices else "(empty)"
    return f"[{elapsed:.1f}s] {text}"


async def run_one(label: str, provider: str, model: str, func):
    try:
        result = await func(model)
        print(f"  OK  {label:30s} ({provider:10s})  {result}")
    except Exception as e:
        err = str(e)
        # Truncate long error messages
        if len(err) > 120:
            err = err[:120] + "..."
        print(f"  FAIL {label:30s} ({provider:10s})  {err}")


async def main():
    print(f"Testing {len(ANTHROPIC_MODELS) + len(OPENAI_MODELS) + len(OPENROUTER_MODELS)} models...\n")

    tasks = []
    for m in ANTHROPIC_MODELS:
        tasks.append(run_one(m, "anthropic", m, test_anthropic))
    for m in OPENAI_MODELS:
        tasks.append(run_one(m, "openai", m, test_openai))
    for m in OPENROUTER_MODELS:
        tasks.append(run_one(m, "openrouter", m, test_openrouter))

    await asyncio.gather(*tasks)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
