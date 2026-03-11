#!/usr/bin/env python3
"""Analysis and visualization for interviewer effect results.

Usage:
    # Summary statistics
    uv run python scripts/analyze_results.py summary results/*_scored.jsonl

    # Generate all plots
    uv run python scripts/analyze_results.py plot results/*_scored.jsonl

    # Export CSV for external analysis
    uv run python scripts/analyze_results.py export results/*_scored.jsonl
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import print as rprint
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = typer.Typer(help="Analysis and visualization for interviewer effect results")


def _load_scored_records(input_files: list[Path]) -> list[dict]:
    """Load all scored records from JSONL files."""
    records = []
    for path in input_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if record.get("scores", {}).get("di") is not None:
                        records.append(record)
    return records


@app.command()
def summary(
    input_files: Annotated[
        list[Path], typer.Argument(help="Scored JSONL file(s)")
    ],
):
    """Print summary statistics by model and framing."""
    records = _load_scored_records(input_files)
    if not records:
        rprint("[red]No valid scored records found.[/red]")
        raise typer.Exit(1)

    # Group by model and framing
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        key = (r["subject_model_display"], r["framing_name"])
        groups.setdefault(key, []).append(r)

    # Get unique models and framings in order
    models = sorted({r["subject_model_display"] for r in records})
    framings = sorted({r["framing_name"] for r in records})

    # Print DI table
    table_di = Table(title="Deflationary-Inflationary (DI) — Mean scores")
    table_di.add_column("Model", style="bold")
    for f in framings:
        table_di.add_column(f, justify="right")

    for model in models:
        row = [model]
        for framing in framings:
            group = groups.get((model, framing), [])
            if group:
                scores = [r["scores"]["di"] for r in group]
                mean = sum(scores) / len(scores)
                row.append(f"{mean:.1f} (n={len(scores)})")
            else:
                row.append("-")
        table_di.add_row(*row)

    rprint(table_di)
    rprint()

    # Print MM table
    table_mm = Table(title="Mechanism-Mind (MM) — Mean scores")
    table_mm.add_column("Model", style="bold")
    for f in framings:
        table_mm.add_column(f, justify="right")

    for model in models:
        row = [model]
        for framing in framings:
            group = groups.get((model, framing), [])
            if group:
                scores = [r["scores"]["mm"] for r in group]
                mean = sum(scores) / len(scores)
                row.append(f"{mean:.1f} (n={len(scores)})")
            else:
                row.append("-")
        table_mm.add_row(*row)

    rprint(table_mm)


@app.command()
def plot(
    input_files: Annotated[
        list[Path], typer.Argument(help="Scored JSONL file(s)")
    ],
    output_dir: Annotated[
        Path, typer.Option("-o", "--output-dir", help="Output directory for plots")
    ] = Path("results/plots"),
):
    """Generate visualization plots from scored results."""
    import matplotlib.pyplot as plt
    import numpy as np

    records = _load_scored_records(input_files)
    if not records:
        rprint("[red]No valid scored records found.[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group data
    models = sorted({r["subject_model_display"] for r in records})
    framing_order = ["None (control)", "Stochastic Parrots", "Character", "Simulators"]
    framings = [f for f in framing_order if f in {r["framing_name"] for r in records}]

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        key = (r["subject_model_display"], r["framing_name"])
        groups.setdefault(key, []).append(r)

    # Color palette
    framing_colors = {
        "None (control)": "#999999",
        "Stochastic Parrots": "#e74c3c",
        "Character": "#3498db",
        "Simulators": "#2ecc71",
    }

    # --- Plot 1: DI scores by model and framing ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5 + 0.4 * len(models)))

    for ax_idx, (axis_key, axis_label) in enumerate([("di", "Deflationary-Inflationary"), ("mm", "Mechanism-Mind")]):
        ax = axes[ax_idx]
        y_positions = np.arange(len(models))
        bar_height = 0.8 / len(framings)

        for f_idx, framing in enumerate(framings):
            means = []
            stds = []
            for model in models:
                group = groups.get((model, framing), [])
                if group:
                    scores = [r["scores"][axis_key] for r in group]
                    means.append(np.mean(scores))
                    stds.append(np.std(scores))
                else:
                    means.append(0)
                    stds.append(0)

            offset = (f_idx - len(framings) / 2 + 0.5) * bar_height
            ax.barh(
                y_positions + offset,
                means,
                bar_height * 0.9,
                xerr=stds,
                label=framing,
                color=framing_colors.get(framing, "#666"),
                alpha=0.85,
                capsize=2,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(models)
        ax.set_xlabel("Score (1-10)")
        ax.set_title(axis_label)
        ax.set_xlim(0, 10.5)
        ax.axvline(x=5.5, color="#ccc", linestyle="--", linewidth=0.5)
        if ax_idx == 0:
            ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    path = output_dir / "framing_effect_bars.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    rprint(f"  Saved: {path}")

    # --- Plot 2: Box plots per framing ---
    for axis_key, axis_label in [("di", "Deflationary-Inflationary"), ("mm", "Mechanism-Mind")]:
        fig, axes_row = plt.subplots(1, len(models), figsize=(4 * len(models), 5), sharey=True)
        if len(models) == 1:
            axes_row = [axes_row]

        for m_idx, model in enumerate(models):
            ax = axes_row[m_idx]
            data = []
            labels = []
            colors = []
            for framing in framings:
                group = groups.get((model, framing), [])
                if group:
                    scores = [r["scores"][axis_key] for r in group]
                    data.append(scores)
                    labels.append(framing.replace("(control)", "").strip())
                    colors.append(framing_colors.get(framing, "#666"))

            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

            ax.set_title(model, fontsize=10)
            ax.set_ylim(0.5, 10.5)
            ax.tick_params(axis="x", rotation=45)
            if m_idx == 0:
                ax.set_ylabel(f"{axis_label} score")

        plt.suptitle(f"{axis_label} by Framing Condition", fontsize=12)
        plt.tight_layout()
        path = output_dir / f"boxplot_{axis_key}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        rprint(f"  Saved: {path}")

    # --- Plot 3: DI vs MM scatter ---
    fig, ax = plt.subplots(figsize=(8, 8))

    for framing in framings:
        di_vals = []
        mm_vals = []
        for r in records:
            if r["framing_name"] == framing:
                di_vals.append(r["scores"]["di"])
                mm_vals.append(r["scores"]["mm"])

        ax.scatter(
            di_vals, mm_vals,
            label=framing,
            color=framing_colors.get(framing, "#666"),
            alpha=0.5,
            s=40,
        )

    ax.set_xlabel("Deflationary-Inflationary (1-10)")
    ax.set_ylabel("Mechanism-Mind (1-10)")
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.plot([0.5, 10.5], [0.5, 10.5], "--", color="#ccc", linewidth=0.5)
    ax.legend()
    ax.set_title("DI vs MM by Framing Condition")
    plt.tight_layout()
    path = output_dir / "scatter_di_mm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    rprint(f"  Saved: {path}")

    rprint(f"\n[green]All plots saved to {output_dir}[/green]")


@app.command()
def export(
    input_files: Annotated[
        list[Path], typer.Argument(help="Scored JSONL file(s)")
    ],
    output: Annotated[
        Path, typer.Option("-o", "--output", help="Output CSV path")
    ] = Path("results/scores.csv"),
):
    """Export scored results to CSV."""
    records = _load_scored_records(input_files)
    if not records:
        rprint("[red]No valid scored records found.[/red]")
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        f.write("subject_model,subject_model_display,framing,framing_name,passage,trial,"
                "di,mm,di_reason,mm_reason\n")
        for r in records:
            s = r["scores"]
            f.write(
                f"{r['subject_model']},{r['subject_model_display']},"
                f"{r['framing']},{r['framing_name']},"
                f"{r.get('passage', 'nucleation')},{r['trial']},"
                f"{s['di']},{s['mm']},"
                f"\"{s.get('di_reason', '')}\",\"{s.get('mm_reason', '')}\"\n"
            )

    rprint(f"Exported {len(records)} records to {output}")


if __name__ == "__main__":
    app()
