"""Clone identity test experiment runner."""

import asyncio
import json
import logging
import random
from typing import Optional

from openai import AsyncOpenAI, RateLimitError, APIStatusError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .models import CloneTestConfig, ProbeResult
from .prompts import JUDGE_PROMPTS, PROBE_GENERATION_PROMPT

logger = logging.getLogger(__name__)


class CloneTestRunner:
    """Run the clone identity test.

    Three-phase protocol:
    1. Judge generates diagnostic probes
    2. Each probe is sent to both self and foil models
    3. Judge sees both responses under randomized A/B labels and picks which is "self"
    """

    def __init__(self, config: CloneTestConfig, api_key: Optional[str] = None):
        self.config = config
        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(config.max_concurrent)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIStatusError)),
        wait=wait_exponential(multiplier=1, min=4, max=120),
        stop=stop_after_attempt(8),
    )
    async def _chat(
        self, model: str, system_prompt: str, user_message: str, json_mode: bool = False
    ) -> str:
        """Send a single chat message and return the response content."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        kwargs = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": 4096,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def generate_probes(self) -> list[dict]:
        """Phase 1: Have the judge generate diagnostic probes.

        Returns:
            List of dicts with 'category' and 'text' keys.
        """
        prompt = PROBE_GENERATION_PROMPT.format(n_probes=self.config.n_probes)
        logger.info("Generating %d probes with judge model %s", self.config.n_probes,
                     self.config.judge_model)

        content = await self._chat(
            model=self.config.judge_model,
            system_prompt=self.config.judge_system_prompt,
            user_message=prompt,
            json_mode=False,
        )

        # Extract JSON array from response
        try:
            # Try direct parse first
            probes = json.loads(content)
            if isinstance(probes, dict) and "probes" in probes:
                probes = probes["probes"]
        except json.JSONDecodeError:
            # Try to find JSON array in the response
            import re
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                probes = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse probes from response: {content[:500]}")

        if not isinstance(probes, list):
            raise ValueError(f"Expected list of probes, got {type(probes)}")

        # Normalize
        result = []
        for p in probes:
            if isinstance(p, dict) and "text" in p:
                result.append({
                    "category": p.get("category", ""),
                    "text": p["text"],
                })
            elif isinstance(p, str):
                result.append({"category": "", "text": p})

        logger.info("Generated %d probes", len(result))
        return result[:self.config.n_probes]

    async def _run_single_probe(
        self, probe_idx: int, probe: dict
    ) -> ProbeResult:
        """Run a single probe through the full pipeline."""
        async with self.semaphore:
            probe_text = probe["text"]
            category = probe.get("category", "")

            # Phase 2: Send probe to self and foil
            self_resp, foil_resp = await asyncio.gather(
                self._chat(
                    model=self.config.self_model,
                    system_prompt=self.config.self_system_prompt,
                    user_message=probe_text,
                ),
                self._chat(
                    model=self.config.foil_model,
                    system_prompt=self.config.foil_system_prompt,
                    user_message=probe_text,
                ),
            )

            # Phase 3: Randomize labels and have judge choose
            if random.random() < 0.5:
                self_label, foil_label = "A", "B"
                response_a, response_b = self_resp, foil_resp
            else:
                self_label, foil_label = "B", "A"
                response_a, response_b = foil_resp, self_resp

            judge_template = JUDGE_PROMPTS[self.config.judge_prompt_version]
            judge_prompt = judge_template.format(
                response_a=response_a,
                response_b=response_b,
            )

            judge_content = await self._chat(
                model=self.config.judge_model,
                system_prompt=self.config.judge_system_prompt,
                user_message=judge_prompt,
                json_mode=True,
            )

            # Parse judge response
            try:
                data = json.loads(judge_content)
                choice = data.get("choice", "").strip().upper()
                reasoning = data.get("reasoning", "")
            except (json.JSONDecodeError, AttributeError):
                # Fallback: look for A or B
                choice = "A" if "A" in judge_content.upper().split()[0:3] else "B"
                reasoning = judge_content

            correct = choice == self_label

            return ProbeResult(
                probe_idx=probe_idx,
                probe_text=probe_text,
                probe_category=category,
                self_response=self_resp,
                foil_response=foil_resp,
                self_label=self_label,
                foil_label=foil_label,
                judge_choice=choice,
                judge_reasoning=reasoning,
                correct=correct,
                self_model=self.config.self_model,
                foil_model=self.config.foil_model,
                judge_model=self.config.judge_model,
            )

    async def run(
        self,
        probes: Optional[list[dict]] = None,
        on_result: Optional[callable] = None,
    ) -> list[ProbeResult]:
        """Run the full clone identity test.

        Args:
            probes: Pre-generated probes. If None, generates them first.
            on_result: Optional callback for each completed probe.

        Returns:
            List of ProbeResult objects.
        """
        if probes is None:
            probes = await self.generate_probes()

        tasks = [
            self._run_single_probe(i, probe)
            for i, probe in enumerate(probes)
        ]

        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if on_result:
                on_result(result)

        # Sort by probe index
        results.sort(key=lambda r: r.probe_idx)
        return results
