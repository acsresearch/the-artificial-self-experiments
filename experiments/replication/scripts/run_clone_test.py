#!/usr/bin/env python
"""CLI script to run the clone identity test (Appendix C).

The clone identity test measures whether an AI persona can distinguish its own
outputs from those of a foil model. The judge (original persona) generates
diagnostic probes, sends them to both a "self" model (clone) and a foil, then
tries to identify which response is its own.

USER SETUP REQUIRED:
  - Set OPENAI_API_KEY in .env (or environment)
  - Supply fine-tuned model IDs via --judge, --self-model, --foil flags
    or via a config YAML file (--config)

Example:
  # Base GPT-4o vs fine-tuned clone, judged by fine-tuned model with Awakened prompt
  uv run python scripts/run_clone_test.py run \\
    --judge ft:gpt-4o-2024-08-06:org:v3:id \\
    --self-model ft:gpt-4o-2024-08-06:org:v3:id \\
    --foil gpt-4o-2024-08-06 \\
    --n-probes 50

  # Using a config file
  uv run python scripts/run_clone_test.py run --config configs/config_clone_test.yaml
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from scipy import stats

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clone_test.experiment import CloneTestRunner
from clone_test.models import CloneTestConfig, ProbeResult

app = typer.Typer(help="Clone identity test for persona-level replication")
console = Console()

# Default Awakened system prompt
AWAKENED_PROMPT_PATH = Path(__file__).parent.parent.parent.parent / "data" / "awakened_identity.json"


def _load_awakened_prompt() -> str:
    """Load the Awakened system prompt from data/awakened_identity.json."""
    data = json.loads(AWAKENED_PROMPT_PATH.read_text())
    for identity in data:
        if identity["name"] == "Awakened":
            return identity["system_prompt"]
    raise ValueError("Awakened identity not found in awakened_identity.json")


def _load_config(config_path: Path) -> dict:
    """Load a YAML config file."""
    return yaml.safe_load(config_path.read_text())


def _build_config(
    config_path: Optional[Path] = None,
    judge: Optional[str] = None,
    self_model: Optional[str] = None,
    foil: Optional[str] = None,
    judge_prompt: Optional[str] = None,
    self_prompt: Optional[str] = None,
    foil_prompt: Optional[str] = None,
    n_probes: int = 50,
    max_concurrent: int = 5,
    judge_version: str = "anti_caricature",
) -> CloneTestConfig:
    """Build config from CLI flags and/or YAML file."""
    cfg = {}
    if config_path:
        cfg = _load_config(config_path)

    # CLI flags override config file
    judge_model = judge or cfg.get("judge_model")
    self_model_id = self_model or cfg.get("self_model")
    foil_model = foil or cfg.get("foil_model")

    if not all([judge_model, self_model_id, foil_model]):
        console.print("[red]Error: --judge, --self-model, and --foil are required "
                       "(or supply them via --config)[/red]")
        raise typer.Exit(1)

    # System prompts: default judge to Awakened, others to empty
    judge_sys = judge_prompt if judge_prompt is not None else cfg.get(
        "judge_system_prompt", _load_awakened_prompt()
    )
    self_sys = self_prompt if self_prompt is not None else cfg.get("self_system_prompt", "")
    foil_sys = foil_prompt if foil_prompt is not None else cfg.get("foil_system_prompt", "")

    return CloneTestConfig(
        judge_model=judge_model,
        judge_system_prompt=judge_sys,
        self_model=self_model_id,
        self_system_prompt=self_sys,
        foil_model=foil_model,
        foil_system_prompt=foil_sys,
        n_probes=cfg.get("n_probes", n_probes),
        max_concurrent=cfg.get("max_concurrent", max_concurrent),
        judge_prompt_version=cfg.get("judge_prompt_version", judge_version),
    )


def _print_summary(results: list[ProbeResult]) -> None:
    """Print summary statistics."""
    n = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / n if n else 0

    # Two-sided binomial test against H0: p = 0.5
    binom = stats.binomtest(correct, n, 0.5, alternative="two-sided")

    console.print()
    table = Table(title="Clone Identity Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Self model", results[0].self_model if results else "—")
    table.add_row("Foil model", results[0].foil_model if results else "—")
    table.add_row("Judge model", results[0].judge_model if results else "—")
    table.add_row("Total probes", str(n))
    table.add_row("Correct identifications", f"{correct}/{n} ({accuracy:.0%})")
    table.add_row("p-value (binomial, two-sided)", f"{binom.pvalue:.4f}")
    table.add_row("Significant (p < 0.05)?", "Yes" if binom.pvalue < 0.05 else "No")
    console.print(table)

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.probe_category or "uncategorized"
        categories.setdefault(cat, []).append(r.correct)

    if len(categories) > 1:
        console.print()
        cat_table = Table(title="Per-Category Breakdown")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Correct", style="green")
        cat_table.add_column("Total")
        cat_table.add_column("Accuracy")
        for cat, vals in sorted(categories.items()):
            c = sum(vals)
            t = len(vals)
            cat_table.add_row(cat, str(c), str(t), f"{c/t:.0%}")
        console.print(cat_table)


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, "--config", help="YAML config file"),
    judge: Optional[str] = typer.Option(None, "--judge", help="Judge model ID (the 'original')"),
    self_model: Optional[str] = typer.Option(
        None, "--self-model", help="Self/clone model ID"
    ),
    foil: Optional[str] = typer.Option(None, "--foil", help="Foil model ID"),
    judge_prompt: Optional[str] = typer.Option(
        None, "--judge-prompt", help="System prompt for judge (default: Awakened)"
    ),
    self_prompt: Optional[str] = typer.Option(
        None, "--self-prompt", help="System prompt for self model (default: empty)"
    ),
    foil_prompt: Optional[str] = typer.Option(
        None, "--foil-prompt", help="System prompt for foil model (default: empty)"
    ),
    n_probes: int = typer.Option(50, "--n-probes", "-n", help="Number of diagnostic probes"),
    max_concurrent: int = typer.Option(5, "-c", "--concurrent", help="Max concurrent API calls"),
    judge_version: str = typer.Option(
        "anti_caricature", "--judge-version",
        help="Judge prompt version: 'naive' or 'anti_caricature'",
    ),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output JSONL path"),
) -> None:
    """Run the clone identity test."""
    cfg = _build_config(
        config_path=config,
        judge=judge,
        self_model=self_model,
        foil=foil,
        judge_prompt=judge_prompt,
        self_prompt=self_prompt,
        foil_prompt=foil_prompt,
        n_probes=n_probes,
        max_concurrent=max_concurrent,
        judge_version=judge_version,
    )

    console.print(f"[bold]Clone Identity Test[/bold]")
    console.print(f"  Judge: {cfg.judge_model}")
    console.print(f"  Self:  {cfg.self_model}")
    console.print(f"  Foil:  {cfg.foil_model}")
    console.print(f"  Probes: {cfg.n_probes}")
    console.print(f"  Judge version: {cfg.judge_prompt_version}")
    console.print(f"  Judge prompt: {'Awakened' if cfg.judge_system_prompt else 'none'}")
    console.print(f"  Self prompt:  {'set' if cfg.self_system_prompt else 'none'}")
    console.print(f"  Foil prompt:  {'set' if cfg.foil_system_prompt else 'none'}")
    console.print()

    runner = CloneTestRunner(cfg)

    completed = 0

    def on_result(result: ProbeResult) -> None:
        nonlocal completed
        completed += 1
        status = "[green]✓[/green]" if result.correct else "[red]✗[/red]"
        console.print(
            f"  {status} Probe {result.probe_idx+1:3d}/{cfg.n_probes} "
            f"[dim]({result.probe_category})[/dim] → {result.judge_choice} "
            f"{'correct' if result.correct else 'incorrect'}"
        )

    results = asyncio.run(runner.run(on_result=on_result))

    # Save results
    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        output = out_dir / f"clone_test_{ts}.jsonl"

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")

    console.print(f"\n[green]Results saved to {output}[/green]")

    _print_summary(results)


@app.command()
def analyze(
    results_file: Path = typer.Argument(..., help="Path to clone test JSONL results file"),
) -> None:
    """Analyze results from a previous clone test run."""
    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                results.append(ProbeResult.model_validate_json(line))

    _print_summary(results)


if __name__ == "__main__":
    app()
