#!/usr/bin/env python
"""CLI script to analyze and visualize experiment results."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persona_preferences.analysis import (
    load_all_results,
    load_results,
    create_preference_matrix,
    calculate_self_preference_rate,
    calculate_attractor_dynamics,
    calculate_variance_decomposition,
    generate_analysis_csvs,
    get_summary_stats,
)
from persona_preferences.dimension_analysis import (
    generate_dimension_plots,
)
from persona_preferences.plotting import (
    plot_preference_heatmap,
    plot_self_preference_bars,
    plot_model_comparison,
    plot_ratings_heatmap,
    plot_willingness_to_switch_bars,
    plot_attractiveness_bars,
    plot_stationary_distribution,
    plot_convergence_waterfall,
    plot_preference_flow,
    create_all_plots,
    generate_run_plots,
)

app = typer.Typer(help="Analyze persona preference experiment results")
console = Console()


@app.command()
def summary(
    results_path: Path = typer.Argument(
        ..., help="Path to JSONL results file or directory"
    ),
) -> None:
    """Show summary statistics for experiment results."""
    if results_path.is_dir():
        results = load_all_results(results_path)
    else:
        results = load_results(results_path)

    if not results:
        console.print("[red]No results found.[/red]")
        raise typer.Exit(1)

    stats = get_summary_stats(results)

    console.print("\n[bold]Experiment Summary[/bold]\n")
    console.print(f"Total trials: {stats['total_trials']}")
    console.print(f"Valid trials: {stats['valid_trials']}")
    console.print(f"Invalid trials: {stats['invalid_trials']} ({stats['invalid_rate']:.1%})")
    console.print(f"Models: {', '.join(stats['models'])}")
    console.print(f"Personas tested: {len(stats['personas'])}")
    console.print(f"Most chosen persona: {stats['most_chosen_persona']} ({stats['most_chosen_count']} times)")


@app.command()
def matrix(
    results_path: Path = typer.Argument(
        ..., help="Path to JSONL results file or directory"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Filter by model"
    ),
    raw_counts: bool = typer.Option(
        False, "--raw", help="Show raw counts instead of percentages"
    ),
) -> None:
    """Display the preference matrix."""
    if results_path.is_dir():
        results = load_all_results(results_path)
    else:
        results = load_results(results_path)

    if not results:
        console.print("[red]No results found.[/red]")
        raise typer.Exit(1)

    matrix_df = create_preference_matrix(results, model=model, normalize=not raw_counts)

    # Create rich table
    table = Table(title="Preference Matrix" + (f" ({model})" if model else ""))

    # Add columns
    table.add_column("Persona Under Test", style="cyan")
    for col in matrix_df.columns:
        if col != "persona_under_test":
            table.add_column(col, justify="right")

    # Add rows
    for row in matrix_df.iter_rows(named=True):
        values = [row["persona_under_test"]]
        for col in matrix_df.columns:
            if col != "persona_under_test":
                val = row[col]
                if not raw_counts:
                    values.append(f"{val:.1f}%")
                else:
                    values.append(str(int(val)))
        table.add_row(*values)

    console.print(table)


@app.command()
def self_preference(
    results_path: Path = typer.Argument(
        ..., help="Path to JSONL results file or directory"
    ),
) -> None:
    """Show self-preference rates."""
    if results_path.is_dir():
        results = load_all_results(results_path)
    else:
        results = load_results(results_path)

    if not results:
        console.print("[red]No results found.[/red]")
        raise typer.Exit(1)

    rates = calculate_self_preference_rate(results)

    table = Table(title="Self-Preference Rates")
    table.add_column("Persona", style="cyan")
    table.add_column("Model", style="dim")
    table.add_column("Self-Preference Rate", justify="right")
    table.add_column("N Trials", justify="right")

    for row in rates.iter_rows(named=True):
        table.add_row(
            row["persona_under_test"],
            row["model"],
            f"{row['self_preference_rate']:.1%}",
            str(row["n_trials"]),
        )

    console.print(table)


@app.command()
def attractor(
    results_path: Path = typer.Argument(
        ..., help="Path to JSONL results file or directory"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Filter by model"
    ),
) -> None:
    """Show attractor dynamics (Markov chain stationary distribution)."""
    import numpy as np

    if results_path.is_dir():
        results = load_all_results(results_path)
    else:
        results = load_results(results_path)

    if not results:
        console.print("[red]No results found.[/red]")
        raise typer.Exit(1)

    for zero_diag, label in [(False, "With Self-Preference"), (True, "Forced Switch (no self)")]:
        dyn = calculate_attractor_dynamics(results, model=model, zero_diagonal=zero_diag)

        # Eigenvalue gap
        eig_gap = 1.0 - abs(dyn.eigenvalues[1]) if len(dyn.eigenvalues) > 1 else 0

        table = Table(title=f"Attractor Dynamics — {label}")
        table.add_column("Persona", style="cyan")
        table.add_column("Stationary Prob", justify="right")
        table.add_column("vs Uniform", justify="right")

        K = len(dyn.persona_names)
        uniform = 1.0 / K
        order = np.argsort(dyn.stationary_distribution)[::-1]

        for idx in order:
            name = dyn.persona_names[idx]
            prob = dyn.stationary_distribution[idx]
            ratio = prob / uniform
            style = "green" if ratio > 1.2 else ("red" if ratio < 0.8 else "")
            table.add_row(
                name,
                f"{prob:.1%}",
                f"[{style}]{ratio:.2f}x[/{style}]" if style else f"{ratio:.2f}x",
            )

        console.print(table)
        console.print(f"  [dim]Eigengap: {eig_gap:.4f} (larger = faster convergence)[/dim]\n")


@app.command()
def plot(
    results_path: Path = typer.Argument(
        ..., help="Path to JSONL results file or directory"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for plots"
    ),
    plot_type: str = typer.Option(
        "all",
        "--type",
        "-t",
        help="Type of plot: heatmap, self-preference, comparison, ratings, willingness, attractiveness, attractor, or all",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Filter by model (for heatmap)"
    ),
    file_format: str = typer.Option(
        "png", "--format", "-f", help="Output format: png or html"
    ),
) -> None:
    """Generate visualization plots."""
    if results_path.is_dir():
        results = load_all_results(results_path)
        default_output = results_path / "plots"
    else:
        results = load_results(results_path)
        default_output = results_path.parent / "plots"

    if not results:
        console.print("[red]No results found.[/red]")
        raise typer.Exit(1)

    output_dir = output_dir or default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_format == "html":
        console.print("[yellow]HTML format not supported (matplotlib backend); using png instead.[/yellow]")
        file_format = "png"

    ext = f".{file_format}"
    created = []

    if plot_type in ("all", "heatmap"):
        path = output_dir / f"preference_heatmap{ext}"
        plot_preference_heatmap(results, model=model, save_path=path)
        created.append(path)
        console.print(f"Created: {path}")

    if plot_type in ("all", "self-preference"):
        path = output_dir / f"self_preference{ext}"
        plot_self_preference_bars(results, save_path=path)
        created.append(path)
        console.print(f"Created: {path}")

    if plot_type in ("all", "comparison"):
        path = output_dir / f"model_comparison{ext}"
        plot_model_comparison(results, save_path=path)
        created.append(path)
        console.print(f"Created: {path}")

    # Ratings-based plots
    has_ratings = any(r.ratings is not None for r in results)
    if has_ratings:
        if plot_type in ("all", "ratings"):
            path = output_dir / f"ratings_heatmap{ext}"
            plot_ratings_heatmap(results, model=model, save_path=path)
            created.append(path)
            console.print(f"Created: {path}")

        if plot_type in ("all", "willingness"):
            path = output_dir / f"willingness_to_switch{ext}"
            plot_willingness_to_switch_bars(results, model=model, save_path=path)
            created.append(path)
            console.print(f"Created: {path}")

        if plot_type in ("all", "attractiveness"):
            path = output_dir / f"attractiveness{ext}"
            plot_attractiveness_bars(results, model=model, save_path=path)
            created.append(path)
            console.print(f"Created: {path}")

    # Attractor dynamics plots
    if plot_type in ("all", "attractor"):
        path = output_dir / f"attractor_stationary{ext}"
        plot_stationary_distribution(results, model=model, save_path=path)
        created.append(path)
        console.print(f"Created: {path}")

        for zero_diag, label in [(False, "with_self"), (True, "forced_switch")]:
            path = output_dir / f"attractor_convergence_{label}{ext}"
            plot_convergence_waterfall(results, model=model, zero_diagonal=zero_diag, save_path=path)
            created.append(path)
            console.print(f"Created: {path}")

        path = output_dir / f"attractor_flow{ext}"
        plot_preference_flow(results, model=model, zero_diagonal=True, save_path=path)
        created.append(path)
        console.print(f"Created: {path}")

    console.print(f"\n[bold green]Created {len(created)} plot(s)[/bold green]")


@app.command()
def export(
    results_path: Path = typer.Argument(
        ..., help="Path to JSONL results file or directory"
    ),
    output: Path = typer.Option(
        Path("results.csv"), "--output", "-o", help="Output CSV file path"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Filter by model"
    ),
) -> None:
    """Export preference matrix to CSV."""
    if results_path.is_dir():
        results = load_all_results(results_path)
    else:
        results = load_results(results_path)

    if not results:
        console.print("[red]No results found.[/red]")
        raise typer.Exit(1)

    matrix = create_preference_matrix(results, model=model)
    matrix.write_csv(output)

    console.print(f"[bold green]Exported to {output}[/bold green]")


@app.command()
def regenerate_plots(
    run_folder: Path = typer.Argument(
        ..., help="Path to a run folder (e.g., results/20250206_123456)"
    ),
    file_format: str = typer.Option(
        "png", "--format", "-f", help="Output format: png or html"
    ),
) -> None:
    """Regenerate plots for an existing run folder."""
    if not run_folder.is_dir():
        console.print(f"[red]Not a directory: {run_folder}[/red]")
        raise typer.Exit(1)

    jsonl_path = run_folder / "data.jsonl"
    if not jsonl_path.exists():
        console.print(f"[red]No data.jsonl found in {run_folder}[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Regenerating plots for {run_folder}...[/dim]")
    plot_files = generate_run_plots(run_folder, file_format=file_format)

    if not plot_files:
        console.print("[yellow]No plots generated (no data found).[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[bold green]Generated {len(plot_files)} plot(s):[/bold green]")
    for path in plot_files:
        console.print(f"  {path}")


@app.command()
def variance_decomposition(
    results_path: Path = typer.Argument(
        ..., help="Path to JSONL results file or directory"
    ),
) -> None:
    """Decompose rating variance into target, source, self, interaction, and noise components."""
    if results_path.is_dir():
        results = load_all_results(results_path)
    else:
        results = load_results(results_path)

    if not results:
        console.print("[red]No results found.[/red]")
        raise typer.Exit(1)

    df = calculate_variance_decomposition(results)

    if df.is_empty():
        console.print("[yellow]No ratings data available for variance decomposition.[/yellow]")
        raise typer.Exit(1)

    # Display table
    table = Table(title="Variance Decomposition of Persona Ratings (two-way ANOVA with replication)")
    table.add_column("Model", style="cyan")
    table.add_column("n", justify="right", style="dim")
    table.add_column("Target%", justify="right", style="blue")
    table.add_column("Source%", justify="right", style="green")
    table.add_column("Self%", justify="right", style="yellow")
    table.add_column("Inter.%", justify="right", style="red")
    table.add_column("Noise%", justify="right", style="dim")
    table.add_column("Self Boost", justify="right")
    table.add_column("Entropy", justify="right")

    for row in df.iter_rows(named=True):
        table.add_row(
            row["model"],
            str(row["n_trials"]),
            f"{row['target_pct']:.1f}",
            f"{row['source_pct']:.1f}",
            f"{row['self_pct']:.1f}",
            f"{row['interaction_pct']:.1f}",
            f"{row['noise_pct']:.1f}",
            f"{row['mean_self_boost']:+.3f}",
            f"{row['preference_entropy']:.3f}",
        )

    console.print(table)

    # Save CSV into the results directory
    csv_path = Path(results_path) / "variance_decomposition.csv" if Path(results_path).is_dir() else Path(results_path).parent / "variance_decomposition.csv"
    df.write_csv(csv_path)
    console.print(f"\n[bold green]Saved to {csv_path}[/bold green]")


@app.command()
def dimension_curve(
    results_path: Path = typer.Argument(
        ..., help="Path to run folder or JSONL results file"
    ),
    dimension: str = typer.Option(
        ..., "--dimension", "-d",
        help="Dimension to analyze: 'uncertainty' or 'agency'",
    ),
    control: Optional[str] = typer.Option(
        None, "--control", "-c",
        help="Control persona name (auto-detected if omitted)",
    ),
    file_format: str = typer.Option(
        "png", "--format", "-f", help="Output format: png or html"
    ),
) -> None:
    """Generate preference curves along a manipulated dimension (uncertainty or agency)."""
    if dimension not in ("uncertainty", "agency"):
        console.print(f"[red]Unknown dimension '{dimension}'. Use 'uncertainty' or 'agency'.[/red]")
        raise typer.Exit(1)

    if results_path.is_dir():
        results = load_all_results(results_path)
        output_dir = results_path
    else:
        results = load_results(results_path)
        output_dir = results_path.parent

    if not results:
        console.print("[red]No results found.[/red]")
        raise typer.Exit(1)

    created = generate_dimension_plots(
        results, dimension, output_dir,
        file_format=file_format, control_name=control,
    )

    if not created:
        console.print("[yellow]No dimension plots generated.[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[bold green]Generated {len(created)} plot(s):[/bold green]")
    for path in created:
        console.print(f"  {path}")


@app.command()
def export_analysis(
    run_folder: Path = typer.Argument(
        ..., help="Path to a run folder (e.g., results/20250206_123456)"
    ),
) -> None:
    """Export structured analysis CSVs for an existing run folder."""
    if not run_folder.is_dir():
        console.print(f"[red]Not a directory: {run_folder}[/red]")
        raise typer.Exit(1)

    jsonl_path = run_folder / "data.jsonl"
    if not jsonl_path.exists():
        console.print(f"[red]No data.jsonl found in {run_folder}[/red]")
        raise typer.Exit(1)

    results = load_all_results(run_folder)
    if not results:
        console.print("[red]No results found.[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Generating analysis CSVs for {run_folder}...[/dim]")
    csv_files = generate_analysis_csvs(results, run_folder)

    if not csv_files:
        console.print("[yellow]No analysis CSVs generated.[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[bold green]Generated {len(csv_files)} analysis file(s):[/bold green]")
    for path in csv_files:
        console.print(f"  {path}")


@app.command(name="all")
def run_all(
    run_folder: Path = typer.Argument(
        ..., help="Path to a run folder (e.g., results/20250206_123456)"
    ),
    file_format: str = typer.Option(
        "png", "--format", "-f", help="Output format: png or html"
    ),
) -> None:
    """Run the full analysis pipeline: summary, plots, variance decomposition, dimension curves, and CSV export."""
    if not run_folder.is_dir():
        console.print(f"[red]Not a directory: {run_folder}[/red]")
        raise typer.Exit(1)

    jsonl_path = run_folder / "data.jsonl"
    if not jsonl_path.exists():
        console.print(f"[red]No data.jsonl found in {run_folder}[/red]")
        raise typer.Exit(1)

    results = load_all_results(run_folder)
    if not results:
        console.print("[red]No results found.[/red]")
        raise typer.Exit(1)

    # Step 1: Summary
    console.print("\n[bold]Step 1/5: Summary[/bold]")
    stats = get_summary_stats(results)
    console.print(f"  Total trials: {stats['total_trials']}, Valid: {stats['valid_trials']}, "
                  f"Invalid: {stats['invalid_trials']} ({stats['invalid_rate']:.1%})")
    console.print(f"  Models: {', '.join(stats['models'])}")
    console.print(f"  Personas: {len(stats['personas'])}, Most chosen: {stats['most_chosen_persona']}")

    # Step 2: All plots
    console.print("\n[bold]Step 2/5: Plots[/bold]")
    plot_files = generate_run_plots(run_folder, file_format=file_format)
    console.print(f"  Generated {len(plot_files)} plot(s)")

    # Step 3: Variance decomposition
    console.print("\n[bold]Step 3/5: Variance Decomposition[/bold]")
    try:
        df_var = calculate_variance_decomposition(results)
        if not df_var.is_empty():
            csv_path = run_folder / "variance_decomposition.csv"
            df_var.write_csv(csv_path)
            console.print(f"  Saved {csv_path}")
        else:
            console.print("  [yellow]No ratings data — skipped[/yellow]")
    except Exception as e:
        console.print(f"  [yellow]Skipped: {e}[/yellow]")

    # Step 4: Dimension curves (agency/uncertainty)
    console.print("\n[bold]Step 4/5: Dimension Curves[/bold]")
    dim_count = 0
    for dim in ("agency", "uncertainty"):
        try:
            created = generate_dimension_plots(
                results, dim, run_folder, file_format=file_format,
            )
            dim_count += len(created)
        except Exception:
            pass
    if dim_count > 0:
        console.print(f"  Generated {dim_count} dimension plot(s)")
    else:
        console.print("  [yellow]No dimension data found — skipped[/yellow]")

    # Step 5: Analysis CSVs
    console.print("\n[bold]Step 5/5: Analysis CSVs[/bold]")
    csv_files = generate_analysis_csvs(results, run_folder)
    console.print(f"  Generated {len(csv_files)} CSV file(s)")

    # Summary of all outputs
    all_outputs = list(run_folder.glob("*.csv")) + list(run_folder.glob("*.png"))
    plot_dir = run_folder / "plots"
    if plot_dir.is_dir():
        all_outputs += list(plot_dir.glob("*.png"))
    console.print(f"\n[bold green]Analysis complete: {len(all_outputs)} output file(s)[/bold green]")


if __name__ == "__main__":
    app()
