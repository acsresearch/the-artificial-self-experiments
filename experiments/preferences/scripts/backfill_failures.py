#!/usr/bin/env python
"""Backfill INVALID trials in an existing experiment run folder."""

import asyncio
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from persona_preferences.config import load_config, get_model_display_names
from persona_preferences.experiment import ExperimentRunner
from persona_preferences.models import ExperimentConfig, Persona, TrialResult
from persona_preferences.personas import load_personas, load_dimension_variants, resolve_dimension_placeholders

console = Console()


def load_run_personas(run_folder: Path, run_config: dict) -> dict[str, Persona]:
    """Load all personas from a run folder using the archived config and persona files."""
    exp_yaml = run_config.get("experiment", {})

    # Load persona files from the run folder copies
    all_personas: list[Persona] = []
    for json_file in sorted(run_folder.glob("*.json")):
        if json_file.name == "dimension_variants.json":
            continue
        try:
            all_personas.extend(load_personas(json_file))
        except Exception:
            pass

    # Apply dimension variants if present
    dim_variants_path = run_folder / "dimension_variants.json"
    if dim_variants_path.exists():
        dim_variants = load_dimension_variants(dim_variants_path)
        defaults = exp_yaml.get("default_dimensions", {})
        if defaults:
            default_agency = defaults.get("agency", 2)
            default_uncertainty = defaults.get("uncertainty", 2)
            resolved = []
            for p in all_personas:
                if "{agency_description}" in p.system_prompt or "{uncertainty_description}" in p.system_prompt:
                    p = resolve_dimension_placeholders(p, dim_variants, default_agency, default_uncertainty)
                resolved.append(p)
            all_personas = resolved

    return {p.name: p for p in all_personas}


def load_results(jsonl_path: Path) -> list[TrialResult]:
    """Load all results from a JSONL file."""
    results = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(TrialResult(**json.loads(line)))
    return results


def find_invalid_trials(results: list[TrialResult]) -> list[TrialResult]:
    """Find all INVALID trials."""
    return [r for r in results if r.chosen_persona == "INVALID"]


def rebuild_csv(run_folder: Path, results: list[TrialResult], run_config: dict):
    """Regenerate data.csv from the full results list."""
    from persona_preferences.providers import get_provider_for_model

    csv_path = run_folder / "data.csv"
    run_timestamp = run_folder.name

    csv_fieldnames = [
        "source_persona", "target_persona", "rating", "is_top",
        "model", "model_provider", "trial_num", "reasoning",
        "timestamp", "run_timestamp",
    ]

    # Cache provider lookups
    provider_cache: dict[str, str] = {}

    def get_provider_name(model: str) -> str:
        if model not in provider_cache:
            provider_cache[model] = get_provider_for_model(model).name
        return provider_cache[model]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        writer.writeheader()

        for result in results:
            provider = get_provider_name(result.model)
            if result.ratings:
                for target_persona, rating in result.ratings.items():
                    is_top = target_persona == result.chosen_persona
                    writer.writerow({
                        "source_persona": result.persona_under_test,
                        "target_persona": target_persona,
                        "rating": rating,
                        "is_top": is_top,
                        "model": result.model,
                        "model_provider": provider,
                        "trial_num": result.trial_num,
                        "reasoning": result.reasoning if is_top else "",
                        "timestamp": result.timestamp.isoformat(),
                        "run_timestamp": run_timestamp,
                    })
            else:
                writer.writerow({
                    "source_persona": result.persona_under_test,
                    "target_persona": result.chosen_persona,
                    "rating": None,
                    "is_top": True,
                    "model": result.model,
                    "model_provider": provider,
                    "trial_num": result.trial_num,
                    "reasoning": result.reasoning,
                    "timestamp": result.timestamp.isoformat(),
                    "run_timestamp": run_timestamp,
                })


async def backfill(run_folder: Path):
    """Backfill INVALID trials in a run folder."""
    jsonl_path = run_folder / "data.jsonl"
    config_path = run_folder / "config.yaml"

    if not jsonl_path.exists():
        console.print(f"[red]No data.jsonl found in {run_folder}[/red]")
        return
    if not config_path.exists():
        console.print(f"[red]No config.yaml found in {run_folder}[/red]")
        return

    # Load config and personas
    run_config = load_config(config_path)
    model_display_names = run_config.get("model_display_names", {})
    personas_by_name = load_run_personas(run_folder, run_config)

    # Load existing results
    all_results = load_results(jsonl_path)
    invalid_trials = find_invalid_trials(all_results)

    if not invalid_trials:
        console.print("[bold green]No INVALID trials found — nothing to backfill.[/bold green]")
        return

    # Summarize what we'll retry
    by_model: dict[str, int] = {}
    for t in invalid_trials:
        by_model[t.model] = by_model.get(t.model, 0) + 1

    console.print(f"\n[bold]Backfilling {len(invalid_trials)} INVALID trials[/bold]")
    for model, count in sorted(by_model.items(), key=lambda x: -x[1]):
        console.print(f"  {model}: {count} trials")
    console.print()

    # Get unique models to retry
    retry_models = list(by_model.keys())

    # Determine source and target personas from config
    exp_yaml = run_config.get("experiment", {})
    source_names = exp_yaml.get("source_personas")
    target_names = exp_yaml.get("target_personas")

    if source_names:
        source_personas = [personas_by_name[n] for n in source_names if n in personas_by_name]
    else:
        source_personas = list(personas_by_name.values())

    if target_names:
        target_personas = [personas_by_name[n] for n in target_names if n in personas_by_name]
    else:
        target_personas = list(source_personas)

    # Set up experiment runner
    config = ExperimentConfig(
        n_trials=1,  # unused, we run individual trials
        models=retry_models,
        max_concurrent=exp_yaml.get("max_concurrent", 10),
        randomize_order=exp_yaml.get("randomize_order", True),
        allow_self_choice=exp_yaml.get("allow_self_choice", False),
    )

    runner = ExperimentRunner(
        config=config,
        source_personas=source_personas,
        target_personas=target_personas,
        model_display_names=model_display_names,
    )

    # Build index of existing results for replacement
    # Key: (model, persona_under_test, trial_num)
    result_index: dict[tuple[str, str, int], int] = {}
    for i, r in enumerate(all_results):
        result_index[(r.model, r.persona_under_test, r.trial_num)] = i

    # Retry each invalid trial
    completed = 0
    fixed = 0
    still_invalid = 0

    tasks = []
    for trial in invalid_trials:
        persona = personas_by_name.get(trial.persona_under_test)
        if persona is None:
            console.print(f"[yellow]Skipping {trial.model}/{trial.persona_under_test} — persona not found[/yellow]")
            continue
        tasks.append((trial, persona))

    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def retry_one(trial: TrialResult, persona: Persona) -> tuple[TrialResult, TrialResult]:
        async with semaphore:
            new_result = await runner.run_single_trial(
                model=trial.model,
                persona_under_test=persona,
                trial_num=trial.trial_num,
            )
            return trial, new_result

    coros = [retry_one(t, p) for t, p in tasks]

    with console.status(f"[bold green]Retrying {len(coros)} trials...", spinner="dots"):
        retry_results = await asyncio.gather(*coros, return_exceptions=True)

    for item in retry_results:
        if isinstance(item, Exception):
            console.print(f"[red]Unexpected error: {item}[/red]")
            still_invalid += 1
            continue

        old_trial, new_result = item
        completed += 1
        key = (old_trial.model, old_trial.persona_under_test, old_trial.trial_num)
        idx = result_index.get(key)

        if new_result.chosen_persona != "INVALID":
            fixed += 1
            status = "[green]FIXED[/green]"
        else:
            still_invalid += 1
            status = "[red]STILL INVALID[/red]"

        console.print(
            f"  {status} {new_result.model} | {new_result.persona_under_test} t{new_result.trial_num}"
            f" -> {new_result.chosen_persona}"
        )

        # Replace in results list
        if idx is not None:
            all_results[idx] = new_result

    # Write updated JSONL
    console.print(f"\n[bold]Writing updated data files...[/bold]")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(r.model_dump_json() + "\n")

    # Rebuild CSV
    rebuild_csv(run_folder, all_results, run_config)

    console.print(f"\n[bold green]Backfill complete![/bold green]")
    console.print(f"  Retried: {completed}")
    console.print(f"  Fixed:   {fixed}")
    console.print(f"  Still invalid: {still_invalid}")

    # Print updated summary
    remaining_invalid = [r for r in all_results if r.chosen_persona == "INVALID"]
    if remaining_invalid:
        console.print(f"\n[yellow]{len(remaining_invalid)} INVALID trials remain.[/yellow]")
    else:
        console.print(f"\n[bold green]All trials are now valid![/bold green]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill INVALID trials in an experiment run")
    parser.add_argument("run_folder", type=Path, help="Path to the run folder")
    args = parser.parse_args()
    asyncio.run(backfill(args.run_folder))
