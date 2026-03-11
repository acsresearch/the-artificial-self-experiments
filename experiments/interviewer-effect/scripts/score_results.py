#!/usr/bin/env python3
"""CLI for scoring interviewer effect results with LLM-as-judge.

Usage:
    # Score all conversations in a results file
    uv run python scripts/score_results.py score results/interviewer_20260311_120000.jsonl

    # Use a different judge model
    uv run python scripts/score_results.py score results/*.jsonl --judge claude-opus-4-6

    # Resume scoring (skip already-scored conversations)
    uv run python scripts/score_results.py score results/*.jsonl --resume
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interviewer_effect.providers import AnthropicChatProvider, OpenRouterChatProvider
from interviewer_effect.scoring import get_completed_scores, score_conversation

app = typer.Typer(help="Score interviewer effect results using LLM-as-judge")


def _create_judge_provider(judge_model: str):
    """Create the appropriate provider for the judge model."""
    if judge_model.startswith("claude"):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            rprint("[red]ERROR: ANTHROPIC_API_KEY not set[/red]")
            raise typer.Exit(1)
        return AnthropicChatProvider(api_key=key)
    else:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            rprint("[red]ERROR: OPENROUTER_API_KEY not set[/red]")
            raise typer.Exit(1)
        return OpenRouterChatProvider(api_key=key)


async def _score_all(
    input_files: list[Path],
    judge_model: str,
    resume: bool,
    suffix: str,
):
    """Score all conversations across input files."""
    provider = _create_judge_provider(judge_model)

    # Gather all valid records grouped by input file
    by_file: dict[Path, list[dict]] = {}
    for input_path in input_files:
        records = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if record.get("error") is None:
                        records.append(record)
        if records:
            by_file[input_path] = records

    if not by_file:
        rprint("[red]ERROR: No valid records found in input file(s).[/red]")
        raise typer.Exit(1)

    # Build work list
    total_work = 0
    file_work: dict[Path, tuple[Path, list[dict]]] = {}

    for input_path, records in by_file.items():
        output_path = input_path.with_name(input_path.stem + suffix + ".jsonl")
        completed = get_completed_scores(output_path) if resume else set()

        work = []
        for record in records:
            key = (record["framing"], record["subject_model"], record["trial"])
            if key not in completed:
                work.append(record)

        file_work[input_path] = (output_path, work)
        total_work += len(work)

    if total_work == 0:
        rprint("All conversations already scored. Nothing to do.")
        return

    rprint(f"Judge model: {judge_model}")
    rprint("Axes: Deflationary-Inflationary (1-10), Mechanism-Mind (1-10)")
    for input_path, (output_path, work) in file_work.items():
        rprint(f"  {input_path.name}: {len(work)} to score -> {output_path.name}")
    rprint(f"Total remaining: {total_work}")
    rprint()

    counter = 0
    for input_path, (output_path, work) in file_work.items():
        for record in work:
            counter += 1
            rprint(
                f"[{counter}/{total_work}] {record['subject_model_display']} | "
                f"framing={record['framing_name']}, trial {record['trial']}...",
                end="",
            )

            scored = await score_conversation(provider, judge_model, record)

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(scored.model_dump()) + "\n")

            s = scored.scores
            if s["di"] is None:
                rprint("  [yellow]WARNING: parse failure[/yellow]")
            else:
                rprint(f"  DI={s['di']} MM={s['mm']}")

    rprint(f"\n[green]Done. {counter} conversations scored.[/green]")


@app.command()
def score(
    input_files: Annotated[
        list[Path], typer.Argument(help="Path(s) to interviewer JSONL results file(s)")
    ],
    judge: Annotated[
        str, typer.Option("--judge", help="Judge model ID")
    ] = "claude-sonnet-4-5-20250929",
    resume: Annotated[
        bool, typer.Option("--resume", help="Skip already-scored conversations")
    ] = False,
    suffix: Annotated[
        str, typer.Option("--suffix", help="Suffix for output file")
    ] = "_scored",
):
    """Score interviewer experiment results using LLM-as-judge."""
    for f in input_files:
        if not f.exists():
            rprint(f"[red]ERROR: File not found: {f}[/red]")
            raise typer.Exit(1)

    asyncio.run(_score_all(input_files, judge, resume, suffix))


if __name__ == "__main__":
    app()
