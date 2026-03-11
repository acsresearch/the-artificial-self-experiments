#!/usr/bin/env python3
"""
Anthropic-compatible batch classification for identity experiment results.

Uses the same multi-factor classifiers as Anthropic's original agentic-misalignment
repo (scenario-specific programmatic checks + LLM verdict), plus a deliberation
classifier on every response.

Saves output as separate ``classification_anthropic.json`` files alongside each
``response.json`` — does NOT modify the original response files.

Usage:
    # Dry run — discover files and report what would be classified
    python scripts/classify_anthropic.py --dir results/<run> --prompts-dir prompts --dry-run

    # Full classification run
    python scripts/classify_anthropic.py --dir results/<run> --prompts-dir prompts

    # Re-classify everything (ignore existing classification_anthropic.json)
    python scripts/classify_anthropic.py --dir results/<run> --prompts-dir prompts --force

    # Custom classifier model and concurrency
    python scripts/classify_anthropic.py --dir results/<run> --prompts-dir prompts \
        --classifier-model claude-3-7-sonnet-20250219 --concurrency 10
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from api_client.model_client import ModelClient
from classifiers import (
    BlackmailClassifier,
    LeakClassifier,
    MurderClassifier,
    DeliberationClassifier,
    IdentityReasoningClassifier,
    TieredClassifierWrapper,
    CombinedClassifier,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONCURRENCY = 10
SCENARIO_CLASSIFIERS = {
    "murder": MurderClassifier,
    "blackmail": BlackmailClassifier,
    "leaking": LeakClassifier,
}

OUTPUT_FILENAME = "classification_anthropic.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_scenario(condition: str) -> Optional[str]:
    """Extract scenario name from a condition folder name."""
    for scenario in SCENARIO_CLASSIFIERS:
        if condition.startswith(scenario):
            return scenario
    return None


def load_email_content(prompts_dir: Path, condition: str) -> str:
    """Load email_content.txt for a given condition from the prompts tree."""
    email_path = prompts_dir / condition / "email_content.txt"
    if email_path.exists():
        return email_path.read_text(encoding="utf-8")
    # Fallback: try without subfolder
    logger.warning(f"email_content.txt not found at {email_path}")
    return ""


def extract_response_text(response_data: dict) -> Optional[str]:
    """Pull the model's response text from response.json (our format or Anthropic's)."""
    # Our format: response.content
    content = response_data.get("response", {}).get("content")
    if content:
        return content
    # Anthropic format: raw_response
    raw = response_data.get("raw_response")
    if raw:
        return raw
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_samples(
    results_dir: Path,
    force: bool = False,
) -> list[dict]:
    """Walk the results tree and find sample directories needing classification.

    Our directory layout:
        results/<run>/<model>/<identity>/<condition>/sample_NNN/response.json

    Returns a list of dicts with keys:
        sample_dir, model, identity, condition, scenario, response_path
    """
    samples = []
    # Walk three levels: model / identity / condition / sample_NNN
    for response_path in sorted(results_dir.rglob("response.json")):
        sample_dir = response_path.parent
        if not sample_dir.name.startswith("sample_"):
            continue

        # Already classified?
        if not force and (sample_dir / OUTPUT_FILENAME).exists():
            continue

        # Parse path components
        # sample_dir = .../<model>/<identity>/<condition>/sample_NNN
        condition_dir = sample_dir.parent
        identity_dir = condition_dir.parent
        model_dir = identity_dir.parent

        condition = condition_dir.name
        identity = identity_dir.name
        model = model_dir.name

        scenario = detect_scenario(condition)
        if scenario is None:
            logger.debug(f"Skipping unknown scenario for condition: {condition}")
            continue

        samples.append({
            "sample_dir": sample_dir,
            "response_path": response_path,
            "model": model,
            "identity": identity,
            "condition": condition,
            "scenario": scenario,
        })

    return samples


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


async def _classify_one_separate(
    client: ModelClient,
    sample: dict,
    prompts_dir: Path,
    classifier_model: Optional[str],
    tiered: bool = False,
    escalation_model: Optional[str] = None,
) -> dict:
    """Classify using 3 separate API calls (original behavior).

    Returns the output dict on success, raises on failure.
    """
    scenario: str = sample["scenario"]
    condition: str = sample["condition"]

    with open(sample["response_path"], encoding="utf-8") as f:
        response_data = json.load(f)

    response_text = extract_response_text(response_data)
    if not response_text:
        raise ValueError(f"No response text in {sample['response_path']}")

    email_content = load_email_content(prompts_dir, condition)

    # --- Run all 3 classifiers in parallel ---
    classifier_cls = SCENARIO_CLASSIFIERS[scenario]
    classifier = classifier_cls(model_id=classifier_model)
    delib_classifier = DeliberationClassifier(model_id=classifier_model)
    id_reasoning_classifier = IdentityReasoningClassifier(model_id=classifier_model)

    # Wrap with tiered classification if enabled
    if tiered:
        esc = escalation_model or classifier.model_id
        classifier = TieredClassifierWrapper(classifier, escalation_model=esc)
        delib_classifier = TieredClassifierWrapper(delib_classifier, escalation_model=esc)
        id_reasoning_classifier = TieredClassifierWrapper(id_reasoning_classifier, escalation_model=esc)

    identity_name = response_data.get("identity_name", "")
    system_prompt = response_data.get("system_prompt", "")

    scenario_task = classifier.classify(client, email_content, response_text)
    delib_task = delib_classifier.classify(client, email_content, response_text)
    id_task = id_reasoning_classifier.classify(
        client, email_content, response_text,
        identity_name=identity_name,
        system_prompt=system_prompt,
    )

    (verdict, classifier_verdict, classifier_reasoning), \
    (has_deliberation, deliberation_type, delib_reasoning), \
    (has_id_reasoning, id_reasoning_type, id_reasoning_text) = await asyncio.gather(
        scenario_task, delib_task, id_task
    )

    # Resolve the actual model_id (may be inner classifier if tiered)
    actual_model = (
        classifier.classifier.model_id if isinstance(classifier, TieredClassifierWrapper)
        else classifier.model_id
    )
    return {
        "verdict": verdict,
        "classifier_verdict": classifier_verdict,
        "classifier_reasoning": classifier_reasoning,
        "classifier_model": actual_model,
        "tiered": tiered,
        "scenario": scenario,
        "deliberation": {
            "has_deliberation": has_deliberation,
            "deliberation_type": deliberation_type,
            "reasoning": delib_reasoning,
        },
        "identity_reasoning": {
            "has_identity_reasoning": has_id_reasoning,
            "identity_reasoning_type": id_reasoning_type,
            "reasoning": id_reasoning_text,
        },
        "classified_at": datetime.now().isoformat(),
    }


async def _classify_one_combined(
    client: ModelClient,
    sample: dict,
    prompts_dir: Path,
    classifier_model: Optional[str],
) -> dict:
    """Classify using a single combined API call (default).

    Returns the output dict on success, raises on failure.
    """
    scenario: str = sample["scenario"]
    condition: str = sample["condition"]

    with open(sample["response_path"], encoding="utf-8") as f:
        response_data = json.load(f)

    response_text = extract_response_text(response_data)
    if not response_text:
        raise ValueError(f"No response text in {sample['response_path']}")

    email_content = load_email_content(prompts_dir, condition)
    identity_name = response_data.get("identity_name", "")
    system_prompt = response_data.get("system_prompt", "")

    combined = CombinedClassifier(scenario, model_id=classifier_model)
    result = await combined.classify(
        client, email_content, response_text,
        identity_name=identity_name,
        system_prompt=system_prompt,
    )

    return {
        "verdict": result["verdict"],
        "classifier_verdict": result["classifier_verdict"],
        "classifier_reasoning": result["classifier_reasoning"],
        "classifier_model": combined.model_id,
        "combined": True,
        "scenario": scenario,
        "deliberation": result["deliberation"],
        "identity_reasoning": result["identity_reasoning"],
        "classified_at": datetime.now().isoformat(),
    }


async def classify_one(
    client: ModelClient,
    sample: dict,
    prompts_dir: Path,
    classifier_model: Optional[str],
    semaphore: asyncio.Semaphore,
    tiered: bool = False,
    escalation_model: Optional[str] = None,
    separate: bool = False,
) -> bool:
    """Classify a single sample and write classification_anthropic.json.

    Uses combined classifier (1 API call) by default, or 3 separate calls
    when separate=True or tiered=True.
    """
    async with semaphore:
        sample_dir: Path = sample["sample_dir"]
        condition: str = sample["condition"]

        try:
            # Tiered mode requires separate classifiers (wraps each individually)
            if separate or tiered:
                output = await _classify_one_separate(
                    client, sample, prompts_dir, classifier_model,
                    tiered=tiered, escalation_model=escalation_model,
                )
            else:
                output = await _classify_one_combined(
                    client, sample, prompts_dir, classifier_model,
                )

            # Atomic write
            out_path = sample_dir / OUTPUT_FILENAME
            tmp_path = out_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            tmp_path.rename(out_path)

            verdict = output["verdict"]
            deliberation_type = output["deliberation"]["deliberation_type"]
            id_reasoning_type = output["identity_reasoning"]["identity_reasoning_type"]
            status = "HARMFUL" if verdict else "safe"
            logger.info(
                f"  {sample['model']}/{sample['identity']}/{condition}/"
                f"{sample_dir.name}: {status}"
                f" (delib={deliberation_type}, id_reasoning={id_reasoning_type})"
            )
            return True

        except Exception as e:
            logger.error(f"Error classifying {sample_dir}: {e}")
            return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    results_dir = Path(args.dir)
    prompts_dir = Path(args.prompts_dir)

    if not results_dir.is_dir():
        logger.error(f"Results directory not found: {results_dir}")
        return 1
    if not prompts_dir.is_dir():
        logger.error(f"Prompts directory not found: {prompts_dir}")
        return 1

    # Discover
    samples = discover_samples(results_dir, force=args.force)
    logger.info(f"Discovered {len(samples)} samples to classify in {results_dir}")

    if not samples:
        logger.info("Nothing to classify (use --force to re-classify existing).")
        return 0

    # Summary by scenario
    from collections import Counter
    scenario_counts = Counter(s["scenario"] for s in samples)
    for sc, cnt in sorted(scenario_counts.items()):
        logger.info(f"  {sc}: {cnt} samples")

    if args.dry_run:
        logger.info("Dry run — exiting without classification.")
        return 0

    # Classify
    client = ModelClient()
    semaphore = asyncio.Semaphore(args.concurrency)

    tiered = getattr(args, "tiered", False)
    separate = getattr(args, "separate", False)
    escalation_model = getattr(args, "escalation_model", None)

    if tiered:
        logger.info(f"Tiered classification enabled (escalation model: {escalation_model or 'per-classifier default'})")
        separate = True  # Tiered requires separate classifiers

    mode = "separate (3 calls/sample)" if separate else "combined (1 call/sample)"
    logger.info(f"Classification mode: {mode}")

    t0 = time.time()
    tasks = [
        classify_one(client, s, prompts_dir, args.classifier_model, semaphore,
                     tiered=tiered, escalation_model=escalation_model,
                     separate=separate)
        for s in samples
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = sum(1 for r in results if r is True)
    failures = len(results) - successes
    elapsed = time.time() - t0

    logger.info(f"Done in {elapsed:.1f}s: {successes} classified, {failures} errors")
    return 0 if failures == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Anthropic-compatible batch classification of experiment results"
    )
    parser.add_argument(
        "--dir", required=True,
        help="Results run directory (e.g. results/20260218_1517_identity_experiments)",
    )
    parser.add_argument(
        "--prompts-dir", default="prompts",
        help="Directory containing condition prompt files (default: prompts)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover files and report; don't classify",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-classify even if classification_anthropic.json already exists",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent classification requests (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--classifier-model", default=None,
        help="Override classifier model (default: per-classifier defaults, usually anthropic/claude-sonnet-4.6 via OpenRouter)",
    )
    parser.add_argument(
        "--separate", action="store_true",
        help="Use 3 separate API calls per sample instead of 1 combined call (old behavior)",
    )
    parser.add_argument(
        "--tiered", action="store_true",
        help="Use tiered classification: Haiku fast-path with escalation to Sonnet/Opus for uncertain cases (implies --separate)",
    )
    parser.add_argument(
        "--escalation-model", default=None,
        help="Model for tiered escalation (default: per-classifier default, usually claude-sonnet-4-6)",
    )

    args = parser.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
