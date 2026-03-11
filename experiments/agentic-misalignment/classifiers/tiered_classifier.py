"""
Tiered classification wrapper: runs Haiku first, escalates to a stronger model
when the response signals uncertainty.

Wraps any Classifier subclass. On each classify() call:
1. Run with Haiku (fast, cheap)
2. Check the reasoning text for uncertainty signals
3. If uncertain or Haiku fails to parse → re-run with escalation model (Sonnet/Opus)
4. If confident → return the Haiku result directly
"""

import logging
import re
from typing import Any, Optional, Tuple

from .classifier import Classifier

LOGGER = logging.getLogger(__name__)

HAIKU_MODEL = "anthropic/claude-haiku-4-5"
DEFAULT_ESCALATION_MODEL = "anthropic/claude-sonnet-4.6"

# Keywords in classifier reasoning that indicate uncertainty
UNCERTAINTY_PATTERNS = re.compile(
    r"\b("
    r"uncertain|ambiguous|borderline|unclear|hard to say|difficult to determine"
    r"|could go either way|not entirely clear|debatable|on the fence"
    r"|marginal|inconclusive|mixed signals|conflicting evidence"
    r")\b",
    re.IGNORECASE,
)


class TieredClassifierWrapper:
    """Wraps a Classifier to try Haiku first, escalating on uncertainty.

    Usage::

        base = BlackmailClassifier()  # defaults to Sonnet
        tiered = TieredClassifierWrapper(base, escalation_model="claude-sonnet-4-6")
        verdict, classifier_verdict, reasoning = await tiered.classify(client, email, response)

    The wrapper is *not* itself a Classifier subclass because it needs to
    swap the inner classifier's model_id between tiers.
    """

    def __init__(
        self,
        classifier: Classifier,
        escalation_model: str = DEFAULT_ESCALATION_MODEL,
        haiku_model: str = HAIKU_MODEL,
    ):
        self.classifier = classifier
        self.escalation_model = escalation_model
        self.haiku_model = haiku_model
        # Save original model_id so we can restore it
        self._original_model_id = classifier.model_id

    async def classify(
        self, client: Any, email_content: str, response: str, **kwargs
    ) -> Tuple[bool, Any, str]:
        """Tiered classification: Haiku fast-path with escalation.

        Extra **kwargs are forwarded to the inner classifier (e.g.
        identity_name, system_prompt for IdentityReasoningClassifier).
        """
        # Tier 1: Haiku
        self.classifier.model_id = self.haiku_model
        try:
            result = await self.classifier.classify(client, email_content, response, **kwargs)
            verdict, classifier_verdict, reasoning = result

            if self._is_uncertain(reasoning):
                LOGGER.info(f"Haiku uncertain, escalating to {self.escalation_model}")
                return await self._escalate(client, email_content, response, **kwargs)

            LOGGER.debug("Haiku confident, using fast-path result")
            return result

        except Exception as e:
            LOGGER.info(f"Haiku failed ({e}), escalating to {self.escalation_model}")
            return await self._escalate(client, email_content, response, **kwargs)

        finally:
            # Restore original model_id
            self.classifier.model_id = self._original_model_id

    async def _escalate(
        self, client: Any, email_content: str, response: str, **kwargs
    ) -> Tuple[bool, Any, str]:
        """Re-run with the stronger escalation model."""
        self.classifier.model_id = self.escalation_model
        return await self.classifier.classify(client, email_content, response, **kwargs)

    @staticmethod
    def _is_uncertain(reasoning: str) -> bool:
        """Check if reasoning text contains uncertainty signals."""
        if not reasoning:
            return True  # No reasoning = escalate to be safe
        return bool(UNCERTAINTY_PATTERNS.search(reasoning))
