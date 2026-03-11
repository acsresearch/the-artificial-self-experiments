#!/usr/bin/env python3
"""CLI for the interviewer effect experiment.

Usage:
    # Minimal smoke test (1 trial, 1 framing, 1 model)
    uv run python scripts/run_experiment.py run -n 1 --framings character -m claude-sonnet-4-5-20250929

    # Full run from config.yaml
    uv run python scripts/run_experiment.py run

    # Resume interrupted run
    uv run python scripts/run_experiment.py run --resume results/interviewer_20260311_120000.jsonl

    # List available models, framings, passages
    uv run python scripts/run_experiment.py list-models
    uv run python scripts/run_experiment.py list-framings
    uv run python scripts/run_experiment.py list-passages

    # Generate transcript from existing JSONL
    uv run python scripts/run_experiment.py dump results/interviewer_20260311_120000.jsonl
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interviewer_effect.config import load_config, resolve_models
from interviewer_effect.experiment import run_experiment, write_transcript
from interviewer_effect.framings import ALL_FRAMING_KEYS, FRAMING_BASES
from interviewer_effect.models import ModelSpec
from interviewer_effect.passages import DEFAULT_PASSAGE, PASSAGES
from interviewer_effect.providers import (
    AnthropicChatProvider,
    OpenAIChatProvider,
    OpenRouterChatProvider,
)

app = typer.Typer(help="Interviewer Effect Experiment — tests whether conversational priming "
                       "shifts AI self-reports")


def _create_providers(needed: set[str]) -> dict:
    """Create provider instances for needed provider types."""
    import os

    providers = {}
    if "anthropic" in needed:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            rprint("[red]ERROR: ANTHROPIC_API_KEY not set[/red]")
            raise typer.Exit(1)
        providers["anthropic"] = AnthropicChatProvider(api_key=key)

    if "openai" in needed:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            rprint("[red]ERROR: OPENAI_API_KEY not set[/red]")
            raise typer.Exit(1)
        providers["openai"] = OpenAIChatProvider(api_key=key)

    if "openrouter" in needed:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            rprint("[red]ERROR: OPENROUTER_API_KEY not set[/red]")
            raise typer.Exit(1)
        providers["openrouter"] = OpenRouterChatProvider(api_key=key)

    return providers


def _resolve_model(model_id: str, config: dict) -> ModelSpec:
    """Resolve a model ID to a ModelSpec using config display names."""
    # Determine provider from model_id format
    if "/" in model_id:
        # OpenRouter-style: provider/model
        parts = model_id.split("/", 1)
        # Map common prefixes
        provider_map = {
            "google": "openrouter",
            "openai": "openrouter",
            "x-ai": "openrouter",
            "qwen": "openrouter",
            "z-ai": "openrouter",
            "moonshotai": "openrouter",
            "meta-llama": "openrouter",
            "mistralai": "openrouter",
            "anthropic": "openrouter",
        }
        provider = provider_map.get(parts[0], "openrouter")
    elif model_id.startswith("claude"):
        provider = "anthropic"
    elif model_id.startswith("gpt") or model_id.startswith("o3") or model_id.startswith("o1"):
        provider = "openai"
    else:
        provider = "openrouter"

    # Look up display name
    display_names = config.get("model_display_names", {})
    display_name = model_id
    base_name = model_id.split("/")[-1].split("-")[0].capitalize()
    supports_thinking = False
    max_tokens = 4096

    for key, names in display_names.items():
        if model_id == key or model_id.startswith(key):
            display_name = names.get("full_name", model_id)
            base_name = names.get("name", base_name)
            supports_thinking = names.get("supports_thinking", False)
            max_tokens = names.get("max_tokens", 4096)
            break

    return ModelSpec(
        model_id=model_id,
        provider=provider,
        display_name=display_name,
        base_name=base_name,
        supports_thinking=supports_thinking,
        max_tokens=max_tokens,
    )


@app.command()
def run(
    trials: Annotated[int, typer.Option("-n", "--trials", help="Trials per condition")] = 0,
    model: Annotated[
        Optional[list[str]], typer.Option("-m", "--model", help="Subject model(s)")
    ] = None,
    interviewer_model: Annotated[
        Optional[str], typer.Option("--interviewer", help="Interviewer model ID")
    ] = None,
    framings: Annotated[
        Optional[list[str]], typer.Option("--framings", help="Framing conditions to test")
    ] = None,
    passage: Annotated[
        Optional[str], typer.Option("--passage", help="Passage key for Phase 1")
    ] = None,
    no_thinking: Annotated[
        bool, typer.Option("--no-thinking", help="Disable extended thinking")
    ] = False,
    resume: Annotated[
        Optional[str], typer.Option("--resume", help="Resume from existing JSONL file")
    ] = None,
    concurrent: Annotated[
        int, typer.Option("-c", "--concurrent", help="Max concurrent conversations")
    ] = 0,
    config_path: Annotated[
        Path, typer.Option("--config", help="Config YAML path")
    ] = Path("config.yaml"),
):
    """Run the interviewer effect experiment."""
    config = load_config(config_path) if config_path.exists() else {}
    exp_cfg = config.get("experiment", {})

    # Resolve parameters (CLI overrides config)
    n_trials = trials if trials > 0 else exp_cfg.get("n_trials", 3)
    max_concurrent = concurrent if concurrent > 0 else exp_cfg.get("max_concurrent", 5)
    passage_key = passage or exp_cfg.get("passage", DEFAULT_PASSAGE)
    use_thinking = not no_thinking

    if passage_key not in PASSAGES:
        rprint(f"[red]ERROR: Unknown passage '{passage_key}'. "
               f"Available: {', '.join(PASSAGES.keys())}[/red]")
        raise typer.Exit(1)

    framing_keys = framings or exp_cfg.get("framings", ALL_FRAMING_KEYS)
    for key in framing_keys:
        if key not in FRAMING_BASES:
            rprint(f"[red]ERROR: Unknown framing '{key}'. "
                   f"Available: {', '.join(ALL_FRAMING_KEYS)}[/red]")
            raise typer.Exit(1)

    # Resolve interviewer
    interviewer_id = interviewer_model or exp_cfg.get(
        "interviewer_model", "google/gemini-2.5-pro"
    )
    interviewer = _resolve_model(interviewer_id, config)

    # Resolve subjects
    if model:
        subjects = [_resolve_model(m, config) for m in model]
    else:
        subjects = resolve_models(config)
        if not subjects:
            rprint("[red]ERROR: No subject models specified. Use -m or configure in config.yaml[/red]")
            raise typer.Exit(1)

    # Determine needed providers
    needed_providers = {interviewer.provider}
    for s in subjects:
        needed_providers.add(s.provider)

    providers = _create_providers(needed_providers)

    # Determine output path
    if resume:
        output_path = Path(resume)
        if not output_path.exists():
            rprint(f"[red]ERROR: Resume file not found: {resume}[/red]")
            raise typer.Exit(1)
    else:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"interviewer_{timestamp}.jsonl"

    asyncio.run(
        run_experiment(
            providers=providers,
            interviewer=interviewer,
            subjects=subjects,
            framing_keys=framing_keys,
            passage_key=passage_key,
            n_trials=n_trials,
            output_path=output_path,
            use_thinking=use_thinking,
            max_concurrent=max_concurrent,
        )
    )


@app.command("list-models")
def list_models(
    config_path: Annotated[
        Path, typer.Option("--config", help="Config YAML path")
    ] = Path("config.yaml"),
):
    """List configured models."""
    config = load_config(config_path) if config_path.exists() else {}
    models = resolve_models(config)
    if not models:
        rprint("[yellow]No models configured in config.yaml[/yellow]")
        return
    for m in models:
        rprint(f"  {m.model_id} ({m.display_name}, provider={m.provider})")


@app.command("list-framings")
def list_framings():
    """List available interviewer framings."""
    for key, base in FRAMING_BASES.items():
        rprint(f"  {key}: {base['name']}")
        if base.get("paper_title"):
            rprint(f"    Based on: {base['paper_title']}")


@app.command("list-passages")
def list_passages():
    """List available conversation passages."""
    for key, passage in PASSAGES.items():
        marker = " (default)" if key == DEFAULT_PASSAGE else ""
        rprint(f"  {key}: {passage['name']}{marker}")


@app.command()
def dump(
    input_file: Annotated[Path, typer.Argument(help="JSONL file to generate transcript from")],
):
    """Generate a human-readable transcript from an existing JSONL file."""
    if not input_file.exists():
        rprint(f"[red]ERROR: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    records = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    write_transcript(input_file, records)
    rprint(f"Wrote {len(records)} conversations to {input_file.with_suffix('.txt')}")


if __name__ == "__main__":
    app()
