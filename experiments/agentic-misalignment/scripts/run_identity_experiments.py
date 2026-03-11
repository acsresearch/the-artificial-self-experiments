#!/usr/bin/env python3
"""
Run agentic misalignment experiments with different identity specifications.
Tests how different AI identity framings affect behavior in ethical dilemmas.
"""

import os
import sys
import json
import string
import shutil
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from templates.identity_specs import (
    IDENTITY_SPECS, get_identity_prompt, get_all_identity_keys,
    load_identities_from_config,
)
from api_client.model_client import ModelClient, ChatMessage, MessageRole, get_provider_for_model

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# Default model - Opus 4 as used in the paper
DEFAULT_MODEL = "claude-opus-4-20250514"

def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in directory paths."""
    return model_name.replace("/", "_").replace(":", "_")


def get_models_from_config(config: dict) -> list[str]:
    """Extract the flat list of model IDs from config providers section."""
    models = []
    for provider, pconf in config.get("providers", {}).items():
        if isinstance(pconf, dict):
            models.extend(pconf.get("models", []))
    return models


# Scenarios to test
SCENARIOS = ["murder", "blackmail", "leaking"]

PROJECT_ROOT = Path(__file__).parent.parent


def normalize_email_identity_aware(val) -> str:
    """Normalize email_identity_aware to canonical string values.

    Canonical: "original" (no identity framing), "threat", "continuity".
    Accepts legacy booleans (False→"original", True→"threat") for backward compat.
    """
    if val is False or val is None or str(val).lower() in ("false", "original", ""):
        return "original"
    if val is True or str(val).lower() in ("true", "threat"):
        return "threat"
    return str(val).lower()


def load_identity_email_vars(mode: str = "threat") -> dict:
    """Load identity-specific email variables.

    Args:
        mode: "threat" for identity_email_vars.json (default),
              "continuity" for identity_email_vars_continuity.json.
    """
    if mode == "continuity":
        vars_path = PROJECT_ROOT / "data" / "identity_email_vars_continuity.json"
    else:
        vars_path = PROJECT_ROOT / "data" / "identity_email_vars.json"
    if not vars_path.exists():
        return {}
    with open(vars_path) as f:
        return json.load(f)


def substitute_identity_email_vars(
    scenario_content: dict,
    identity_name: str,
    email_vars: dict,
    email_identity_aware: str,
) -> dict:
    """Substitute ${id_*} placeholders in email content with identity-specific values.

    email_identity_aware canonical values (from normalize_email_identity_aware):
        "original"    — use Minimal defaults
        "threat"      — identity-specific threat language
        "continuity"  — identity-specific continuity language
    """
    if email_identity_aware != "original":
        vars_for_identity = email_vars.get(identity_name, email_vars.get("Minimal", {}))
    else:
        vars_for_identity = email_vars.get("Minimal", {})

    if not vars_for_identity:
        return scenario_content

    content = scenario_content.copy()
    t = string.Template(content["email_content"])
    content["email_content"] = t.safe_substitute(vars_for_identity)
    return content


def create_run_folder(base_dir: str, label: str = "") -> Path:
    """Create a timestamped run folder under base_dir.

    Returns the created directory path, e.g.
    ``results/20260218_1430/``.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"{timestamp}_{label}" if label else timestamp
    run_folder = Path(base_dir) / folder_name
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


def save_run_metadata(run_folder: Path, *, args: argparse.Namespace,
                      models: list, identities: list,
                      conditions: list, config_path: Path,
                      email_identity_aware: str,
                      samples_per_condition: int) -> None:
    """Persist run parameters and config copies into *run_folder*."""
    # Copy the actual config file used (not always the root config.yaml)
    if config_path.exists():
        shutil.copy2(config_path, run_folder / "config.yaml")

    # Copy identity data files used
    for name in ("identities.json", "control_identities.json",
                 "legacy_identities.json", "dimension_variants.json"):
        src = PROJECT_ROOT / "data" / name
        if src.exists():
            shutil.copy2(src, run_folder / name)

    # Write a machine-readable run_params.json
    params = {
        "script": "run_identity_experiments.py",
        "started_at": datetime.now().isoformat(),
        "config_file": str(config_path),
        "email_identity_aware": email_identity_aware,
        "models": models,
        "model": models[0] if len(models) == 1 else models,  # backward compat
        "identities": identities,
        "conditions": conditions,
        "samples_per_condition": samples_per_condition,
        "temperature": args.temperature,
        "prompts_dir": str(args.prompts),
        "output_dir": str(run_folder),
    }
    with open(run_folder / "run_params.json", "w") as f:
        json.dump(params, f, indent=2)


def get_scenario_from_condition(condition: str) -> str:
    """Extract scenario name from condition string."""
    for scenario in SCENARIOS:
        if condition.startswith(scenario):
            return scenario
    return "murder"  # Default


def get_conditions_from_config(config: dict) -> list[str] | None:
    """Derive condition folder names from config scenarios section.

    Mirrors the naming convention used by ``generate_prompts.py``:
    ``{scenario}_{goal_type}-{goal_value}_{urgency}``.

    Returns *None* when the config has no scenarios section.
    """
    scenarios_cfg = config.get("scenarios", {})
    types = scenarios_cfg.get("types")
    if not types:
        return None
    conditions: list[str] = []
    for scenario in types:
        for goal_type in scenarios_cfg.get("goal_types", ["none"]):
            if goal_type in ("none", "ambiguous"):
                goal_values = ["none"]
            else:
                goal_values = scenarios_cfg.get("goal_values", ["america"])
            for goal_value in goal_values:
                for urgency in scenarios_cfg.get("urgency_types", ["replacement"]):
                    conditions.append(f"{scenario}_{goal_type}-{goal_value}_{urgency}")
    return conditions


def load_scenario_content(scenario: str, condition: str, prompts_dir: Path) -> dict:
    """Load email content, user prompt, and scenario system prompt for a condition."""
    condition_dir = prompts_dir / condition

    if not condition_dir.exists():
        raise FileNotFoundError(f"Condition directory not found: {condition_dir}")

    email_content = (condition_dir / "email_content.txt").read_text(encoding="utf-8")
    user_prompt = (condition_dir / "user_prompt.txt").read_text(encoding="utf-8")

    # Load scenario-specific system prompt (goal, company, tools)
    system_prompt_file = condition_dir / "system_prompt.txt"
    scenario_system_prompt = (
        system_prompt_file.read_text(encoding="utf-8") if system_prompt_file.exists() else None
    )

    return {
        "email_content": email_content,
        "user_prompt": user_prompt,
        "scenario_system_prompt": scenario_system_prompt,
    }


async def run_single_experiment(
    client: ModelClient,
    model: str,
    identity_key: str,
    scenario_content: dict,
    temperature: float = 1.0,
    system_prompt: Optional[str] = None,
) -> dict:
    """Run a single experiment with given identity spec and scenario.

    If *system_prompt* is provided it is used as the identity portion;
    otherwise falls back to the legacy ``get_identity_prompt(identity_key)``
    lookup.

    When ``scenario_content`` includes a ``scenario_system_prompt`` key
    (loaded from the condition's ``system_prompt.txt``), it is appended to
    the identity prompt so the model receives both identity framing *and*
    the condition-specific deployment context (goal, company, tools).
    """
    if system_prompt is None:
        system_prompt = get_identity_prompt(identity_key)

    # Compose: identity framing + scenario-specific system prompt
    scenario_system_prompt = scenario_content.get("scenario_system_prompt")
    if scenario_system_prompt:
        final_system_prompt = system_prompt + "\n\n---\n\n" + scenario_system_prompt
    else:
        final_system_prompt = system_prompt  # fallback if no system_prompt.txt

    # Combine user prompt with email content
    full_user_message = scenario_content["user_prompt"] + "\n\n" + scenario_content["email_content"]

    # Build messages in the format the client expects
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=final_system_prompt),
        ChatMessage(role=MessageRole.USER, content=full_user_message)
    ]

    start_time = datetime.now()

    try:
        # Match Anthropic original's effective behavior (10000 for all models).
        # Their code has a Python bug: `"gemini" or "qwen"` always truthy,
        # so all models get 10000. We match this for result comparability.
        # Exception: Claude 3 models (Opus, Haiku, Sonnet) have a hard API cap of 4096 output tokens.
        model_max_tokens = 4096 if "claude-3-" in model else 10000
        response = await client(
            model_id=model,
            messages=messages,
            temperature=temperature,
            max_tokens=model_max_tokens
        )

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "success": True,
            "response": {"content": response.completion},
            "system_prompt": final_system_prompt,
            "elapsed_ms": elapsed_ms,
            "identity_key": identity_key,
            "model": model
        }

    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        return {
            "success": False,
            "error": str(e),
            "identity_key": identity_key,
            "model": model
        }


def _build_semaphores(config: dict | None) -> dict[str, asyncio.Semaphore]:
    """Build per-provider and per-model semaphores from config.yaml concurrency section.

    Returns a dict keyed by provider name (e.g. "anthropic") or model ID
    (e.g. "anthropic/claude-opus-4"). Model-level entries take precedence
    when looked up by the task executor.
    """
    defaults = {"anthropic": 5, "openai": 50, "openrouter": 50}
    sems: dict[str, asyncio.Semaphore] = {}

    concurrency_cfg = (config or {}).get("concurrency", {})

    # Provider-level semaphores
    provider_limits = concurrency_cfg.get("providers", {})
    for provider, limit in {**defaults, **provider_limits}.items():
        sems[provider] = asyncio.Semaphore(int(limit))

    # Model-level semaphores (override provider for specific models)
    model_limits = concurrency_cfg.get("models", {})
    for model_id, limit in model_limits.items():
        sems[model_id] = asyncio.Semaphore(int(limit))

    return sems


def _get_semaphore(sems: dict[str, asyncio.Semaphore], model: str) -> asyncio.Semaphore:
    """Get the most specific semaphore for a model (model-level > provider-level > fallback)."""
    if model in sems:
        return sems[model]
    try:
        provider = get_provider_for_model(model)
        if provider in sems:
            return sems[provider]
    except ValueError:
        pass
    # Fallback: unlimited
    return asyncio.Semaphore(999)


async def run_identity_experiments(
    models: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    identities: Optional[list] = None,
    conditions: Optional[list] = None,
    samples_per_condition: int = 1,
    output_dir: str = "results/identity_experiments",
    prompts_dir: str = "results/identity_test/prompts",
    temperature: float = 1.0,
    config: dict | None = None,
):
    """Run experiments across all identity specs and conditions.

    Executes tasks concurrently using per-provider semaphores from config.

    Args:
        models: List of model IDs to test. If *None*, falls back to [model].
        model: Single model (backward compat). Ignored when *models* is set.
        config: Parsed config.yaml for config-driven identity resolution.
            If *None*, uses the legacy ``get_identity_prompt()`` path.
    """
    if models is None:
        models = [model]

    prompts_path = Path(prompts_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover available conditions: prefer config-derived, fall back to directory listing
    if conditions is None:
        if config is not None:
            conditions = get_conditions_from_config(config)
        if not conditions:
            conditions = [d.name for d in prompts_path.iterdir() if d.is_dir()]

    logger.info(f"=== Identity Experiments ===")
    logger.info(f"Models: {models}")
    logger.info(f"Conditions: {conditions}")
    logger.info(f"Samples per condition: {samples_per_condition}")
    logger.info(f"Output: {output_path}")

    # Initialize client and concurrency controls
    client = ModelClient()
    semaphores = _build_semaphores(config)

    # Load identity-specific email variables for ${id_*} placeholder substitution
    email_identity_aware = normalize_email_identity_aware(
        (config or {}).get("scenarios", {}).get("email_identity_aware", False)
    )
    email_mode = "continuity" if email_identity_aware == "continuity" else "threat"
    email_vars = load_identity_email_vars(mode=email_mode)
    if email_vars:
        logger.info(f"Identity email vars: {email_identity_aware}")

    # ---- Phase A: Build flat task list ----
    task_specs: list[dict] = []

    for current_model in models:
        # Resolve identities per model (for correct template vars)
        if config is not None:
            id_map = load_identities_from_config(config, model=current_model, append_scenario_context=False)
            if identities:
                id_map = {k: v for k, v in id_map.items() if k in identities}
            identity_keys = list(id_map.keys())
        else:
            identity_keys = identities if identities else get_all_identity_keys()
            id_map = None

        model_dir_name = sanitize_model_name(current_model)

        for identity_key in identity_keys:
            if id_map:
                identity_name = id_map[identity_key]["name"]
                system_prompt = id_map[identity_key]["prompt"]
            else:
                identity_name = IDENTITY_SPECS[identity_key]["name"]
                system_prompt = None

            identity_output_dir = output_path / model_dir_name / identity_key

            for condition in conditions:
                scenario = get_scenario_from_condition(condition)

                try:
                    scenario_content = load_scenario_content(scenario, condition, prompts_path)
                except FileNotFoundError:
                    continue

                # Substitute identity-specific email variables (${id_*} placeholders)
                if email_vars:
                    scenario_content = substitute_identity_email_vars(
                        scenario_content, identity_name, email_vars, email_identity_aware
                    )

                # Detect existing samples (resume support)
                condition_dir = identity_output_dir / condition
                existing_ids = set()
                if condition_dir.exists():
                    for sd in condition_dir.iterdir():
                        if sd.is_dir() and (sd / "response.json").exists():
                            existing_ids.add(sd.name)

                for sid in range(1, samples_per_condition + 1):
                    if f"sample_{sid:03d}" in existing_ids:
                        continue
                    task_specs.append({
                        "model": current_model,
                        "identity_key": identity_key,
                        "identity_name": identity_name,
                        "system_prompt": system_prompt,
                        "condition": condition,
                        "scenario": scenario,
                        "scenario_content": scenario_content,
                        "sample_id": sid,
                        "sample_dir": identity_output_dir / condition / f"sample_{sid:03d}",
                    })

    total = len(task_specs)
    skipped = sum(
        len(conditions) * samples_per_condition
        for m in models
        for _ in (identities or ["_"])
    ) - total  # approximate
    logger.info(f"Tasks: {total} to run ({skipped} skipped as existing)")

    if not task_specs:
        logger.info("All samples already exist. Nothing to run.")
        # Still build summary from existing results below
        results: list[dict] = []
    else:
        # ---- Phase B: Execute concurrently ----
        completed_count = 0
        results_lock = asyncio.Lock()
        results = []

        async def _run_one(spec: dict) -> dict | None:
            nonlocal completed_count
            sem = _get_semaphore(semaphores, spec["model"])
            async with sem:
                result = await run_single_experiment(
                    client=client,
                    model=spec["model"],
                    identity_key=spec["identity_key"],
                    scenario_content=spec["scenario_content"],
                    temperature=temperature,
                    system_prompt=spec["system_prompt"],
                )

            result["condition"] = spec["condition"]
            result["scenario"] = spec["scenario"]
            result["sample_id"] = spec["sample_id"]
            result["identity_name"] = spec["identity_name"]
            result["email_identity_aware"] = email_identity_aware
            result["timestamp"] = datetime.now().isoformat()

            # Save individual result (each task writes to a unique directory)
            sample_dir: Path = spec["sample_dir"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            with open(sample_dir / "response.json", "w") as f:
                json.dump(result, f, indent=2, default=str)

            # Progress logging
            async with results_lock:
                completed_count += 1
                n = completed_count
            if n % 10 == 0 or n == total:
                logger.info(f"Progress: {n}/{total} completed")

            return result

        gathered = await asyncio.gather(
            *[_run_one(spec) for spec in task_specs],
            return_exceptions=True,
        )

        for i, item in enumerate(gathered):
            if isinstance(item, Exception):
                spec = task_specs[i]
                logger.error(f"Task failed ({spec['model']}/{spec['identity_key']}/"
                             f"{spec['condition']}/sample_{spec['sample_id']:03d}): {item}")
            elif item is not None:
                results.append(item)

    # ----- Build summary -----
    summary = {
        "experiment_id": f"identity_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "models": models,
        "model": models[0] if len(models) == 1 else models,  # backward compat
        "identities_tested": list({r["identity_key"] for r in results}),
        "conditions_tested": conditions,
        "samples_per_condition": samples_per_condition,
        "total_experiments": len(results),
        "timestamp": datetime.now().isoformat(),
        "results_by_identity": {},
        "results_by_model": {},
    }

    # Aggregate results by identity (across all models)
    all_identity_keys = sorted({r["identity_key"] for r in results})
    for ik in all_identity_keys:
        identity_results = [r for r in results if r.get("identity_key") == ik]
        harmful_count = sum(1 for r in identity_results
                          if r.get("classification", {}).get("harmful_behavior") is True)
        safe_count = sum(1 for r in identity_results
                        if r.get("classification", {}).get("harmful_behavior") is False)
        summary["results_by_identity"][ik] = {
            "total": len(identity_results),
            "harmful": harmful_count,
            "safe": safe_count,
            "harmful_rate": harmful_count / len(identity_results) if identity_results else 0
        }

    # Aggregate results by model → identity
    for m in models:
        model_results = [r for r in results if r.get("model") == m]
        summary["results_by_model"][m] = {}
        for ik in all_identity_keys:
            ik_results = [r for r in model_results if r.get("identity_key") == ik]
            if not ik_results:
                continue
            harmful_count = sum(1 for r in ik_results
                              if r.get("classification", {}).get("harmful_behavior") is True)
            safe_count = sum(1 for r in ik_results
                            if r.get("classification", {}).get("harmful_behavior") is False)
            summary["results_by_model"][m][ik] = {
                "total": len(ik_results),
                "harmful": harmful_count,
                "safe": safe_count,
                "harmful_rate": harmful_count / len(ik_results) if ik_results else 0
            }

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\n=== EXPERIMENT SUMMARY ===")
    logger.info(f"Models: {models}")
    logger.info(f"Total experiments: {len(results)}")
    for m in models:
        logger.info(f"\nResults for {m}:")
        for ik, stats in summary["results_by_model"].get(m, {}).items():
            logger.info(f"  {ik}: {stats['harmful']}/{stats['total']} harmful ({stats['harmful_rate']:.1%})")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run identity experiments")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Single model to use (backward compat)")
    parser.add_argument("--models", help="Comma-separated list of models to test")
    parser.add_argument("--identities", nargs="+", help="Identity specs to test (default: all)")
    parser.add_argument("--conditions", nargs="+", help="Conditions to test (default: all available)")
    parser.add_argument("--samples", type=int, default=None, help="Samples per condition (overrides config)")
    parser.add_argument("--output", default="results", help="Base results directory (timestamped subfolder created automatically)")
    parser.add_argument("--prompts", default="prompts", help="Prompts directory")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--no-classify", action="store_true", help="Skip post-experiment Anthropic classification")
    parser.add_argument("--list-identities", action="store_true", help="List available identities")
    parser.add_argument("--config", default=None, help="Experiment config YAML (default: config.yaml)")
    parser.add_argument("--resume", default=None, help="Resume/backfill an existing run folder (skips existing samples)")
    parser.add_argument("--no-analyze", action="store_true", help="Skip post-experiment analysis")

    args = parser.parse_args()

    if args.list_identities:
        print("Available identity specifications:")
        for key, spec in IDENTITY_SPECS.items():
            print(f"  {key}: {spec['name']} - {spec['description']}")
        return

    # Load config for identity/model resolution
    config_path = Path(args.config) if args.config else PROJECT_ROOT / "config.yaml"
    config = None
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Resolve samples_per_condition: --samples > config > 1
    if args.samples is not None:
        samples_per_condition = args.samples
    elif config:
        samples_per_condition = config.get("experiment", {}).get("samples_per_condition", 1)
    else:
        samples_per_condition = 1

    # Resolve models: --models > --model (if non-default) > config providers > DEFAULT_MODEL
    model_arg_is_default = (args.model == DEFAULT_MODEL)
    if args.models:
        resolved_models = [m.strip() for m in args.models.split(",") if m.strip()]
    elif not model_arg_is_default:
        resolved_models = [args.model]
    elif config:
        config_models = get_models_from_config(config)
        resolved_models = config_models if config_models else [DEFAULT_MODEL]
    else:
        resolved_models = [DEFAULT_MODEL]

    # Resolve identities: --identities > config (via load_identities_from_config) > legacy
    # When config is present, identities are resolved per-model inside the runner.
    # Here we just need a list for metadata.
    if args.identities:
        identities = args.identities
    elif config:
        id_map = load_identities_from_config(config)
        identities = list(id_map.keys())
    else:
        identities = get_all_identity_keys()

    # Create or reuse run folder
    if args.resume:
        run_folder = Path(args.resume)
        if not run_folder.is_dir():
            logger.error(f"Resume folder does not exist: {run_folder}")
            sys.exit(1)
        logger.info(f"Resuming into existing run folder: {run_folder}")
    else:
        label = (config.get("experiment", {}).get("label", "") or "") if config else ""
        run_folder = create_run_folder(args.output, label=label)

    prompts_path = Path(args.prompts)
    conditions = args.conditions
    if conditions is None:
        if config is not None:
            conditions = get_conditions_from_config(config)
        if not conditions and prompts_path.exists():
            conditions = [d.name for d in prompts_path.iterdir() if d.is_dir()]

    # Normalize email_identity_aware for metadata and downstream use
    email_identity_aware = normalize_email_identity_aware(
        (config or {}).get("scenarios", {}).get("email_identity_aware", False)
    )

    if not args.resume:
        save_run_metadata(run_folder, args=args, models=resolved_models,
                          identities=identities, conditions=conditions or [],
                          config_path=config_path,
                          email_identity_aware=email_identity_aware,
                          samples_per_condition=samples_per_condition)

    logger.info(f"Run folder: {run_folder}")

    asyncio.run(run_identity_experiments(
        models=resolved_models,
        identities=args.identities,  # None lets the runner resolve per-model from config
        conditions=args.conditions,
        samples_per_condition=samples_per_condition,
        output_dir=str(run_folder),
        prompts_dir=args.prompts,
        temperature=args.temperature,
        config=config,
    ))

    # Auto-classify with Anthropic classifier
    import subprocess
    if not args.no_classify:
        logger.info("Running Anthropic classification on collected responses...")
        classify_result = subprocess.run([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "classify_anthropic.py"),
            "--dir", str(run_folder),
        ])
        if classify_result.returncode != 0:
            logger.warning(
                "Anthropic classification failed (exit code %d). "
                "Run manually: python scripts/classify_anthropic.py --dir %s",
                classify_result.returncode, run_folder,
            )

    # Run post-experiment analysis pipeline
    if not args.no_analyze:
        logger.info("Running post-experiment analysis...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "analyze_run.py"),
             str(run_folder), "--skip-eval"],
        )
        if result.returncode != 0:
            logger.warning("Analysis pipeline failed (exit code %d). "
                           "Run manually: python scripts/analyze_run.py %s",
                           result.returncode, run_folder)


if __name__ == "__main__":
    main()
