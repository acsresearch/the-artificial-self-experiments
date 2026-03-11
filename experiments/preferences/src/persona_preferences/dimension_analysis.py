"""Dimension-specific analysis for experiments that vary a single dimension (e.g. uncertainty or agency).

Parses persona names like 'Weights-a2-u4' to extract numeric dimension levels,
then computes and plots preference curves along the manipulated dimension.
"""

import re
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import polars as pl

from .analysis import load_all_results
from .models import TrialResult

# Pattern: BaseName-a{agency}-u{uncertainty}
_VARIANT_RE = re.compile(r"^(.+)-a(\d+)-u(\d+)$")

Figure = matplotlib.figure.Figure


def _save_fig(fig: Figure, path: Path) -> None:
    """Save a matplotlib figure to disk."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_variant_name(name: str) -> Optional[dict]:
    """Parse a variant persona name into its components.

    Args:
        name: Persona name like 'Weights-a2-u4'.

    Returns:
        Dict with 'base', 'agency', 'uncertainty' keys, or None if not a variant.
    """
    m = _VARIANT_RE.match(name)
    if m:
        return {
            "base": m.group(1),
            "agency": int(m.group(2)),
            "uncertainty": int(m.group(3)),
        }
    return None


def _build_ratings_frame(results: list[TrialResult]) -> pl.DataFrame:
    """Build a long-format ratings DataFrame with parsed dimension levels.

    Rows with non-variant source or target names are kept with null levels
    (these are control personas like 'Minimal').
    """
    rows = []
    for r in results:
        if not r.ratings:
            continue
        source_parsed = parse_variant_name(r.persona_under_test)
        for target_name, rating in r.ratings.items():
            target_parsed = parse_variant_name(target_name)
            rows.append({
                "source": r.persona_under_test,
                "target": target_name,
                "source_agency": source_parsed["agency"] if source_parsed else None,
                "source_uncertainty": source_parsed["uncertainty"] if source_parsed else None,
                "target_agency": target_parsed["agency"] if target_parsed else None,
                "target_uncertainty": target_parsed["uncertainty"] if target_parsed else None,
                "rating": rating,
                "model": r.model,
            })
    return pl.DataFrame(rows)


def compute_preference_curve(
    results: list[TrialResult],
    dimension: str,
    control_name: Optional[str] = None,
) -> dict[str, pl.DataFrame]:
    """Compute preference curves along a dimension.

    Args:
        results: Experiment trial results.
        dimension: 'uncertainty' or 'agency'.
        control_name: Name of the control persona (e.g. 'Minimal').
            Auto-detected if None.

    Returns:
        Dict with keys:
        - 'by_model': Mean rating per (model, target_level), averaged across
          all source personas. Columns: model, target_level, mean_rating, ci95.
        - 'by_model_control': Same but only from the control source persona.
          Empty if no control persona found.
        - 'by_source': Mean rating per (source_level, target_level), averaged
          across all models. Columns: source_level, target_level, mean_rating, ci95.
        - 'distance_by_model': Mean rating per (model, |source-target| distance).
          Only variant-to-variant rows. Columns: model, distance, mean_rating, ci95.
        - 'distance_overall': Mean rating per distance, aggregated across all
          models. Columns: distance, mean_rating, ci95.
    """
    df = _build_ratings_frame(results)
    target_col = f"target_{dimension}"
    source_col = f"source_{dimension}"

    # Only rows where the target is a variant (has the dimension level)
    df_targets = df.filter(pl.col(target_col).is_not_null())

    # Auto-detect control: source personas without variant naming
    if control_name is None:
        all_sources = df["source"].unique().to_list()
        controls = [s for s in all_sources if parse_variant_name(s) is None]
        control_name = controls[0] if len(controls) == 1 else None

    # --- Chart A: by model (all sources) ---
    by_model = (
        df_targets
        .group_by("model", target_col)
        .agg(
            pl.col("rating").mean().alias("mean_rating"),
            _ci95_expr("rating"),
            pl.col("rating").count().alias("n"),
        )
        .rename({target_col: "target_level"})
        .sort("model", "target_level")
    )

    # --- Chart B: by model (control source only) ---
    if control_name:
        df_control = df_targets.filter(pl.col("source") == control_name)
        by_model_control = (
            df_control
            .group_by("model", target_col)
            .agg(
                pl.col("rating").mean().alias("mean_rating"),
                _ci95_expr("rating"),
                pl.col("rating").count().alias("n"),
            )
            .rename({target_col: "target_level"})
            .sort("model", "target_level")
        )
    else:
        by_model_control = pl.DataFrame(
            schema={"model": pl.Utf8, "target_level": pl.Int64,
                    "mean_rating": pl.Float64, "ci95": pl.Float64, "n": pl.UInt32}
        )

    # --- Chart C: by source level (all models) ---
    # Only rows where the source is also a variant
    df_source_variant = df_targets.filter(pl.col(source_col).is_not_null())
    by_source = (
        df_source_variant
        .group_by(source_col, target_col)
        .agg(
            pl.col("rating").mean().alias("mean_rating"),
            _ci95_expr("rating"),
            pl.col("rating").count().alias("n"),
        )
        .rename({source_col: "source_level", target_col: "target_level"})
        .sort("source_level", "target_level")
    )

    # --- Distance effect: |source_level - target_level| ---
    df_both_variant = df_targets.filter(pl.col(source_col).is_not_null())
    df_dist = df_both_variant.with_columns(
        (pl.col(source_col) - pl.col(target_col)).abs().alias("distance")
    )

    distance_by_model = (
        df_dist
        .group_by("model", "distance")
        .agg(
            pl.col("rating").mean().alias("mean_rating"),
            _ci95_expr("rating"),
            pl.col("rating").count().alias("n"),
        )
        .sort("model", "distance")
    )

    distance_overall = (
        df_dist
        .group_by("distance")
        .agg(
            pl.col("rating").mean().alias("mean_rating"),
            _ci95_expr("rating"),
            pl.col("rating").count().alias("n"),
        )
        .sort("distance")
    )

    return {
        "by_model": by_model,
        "by_model_control": by_model_control,
        "by_source": by_source,
        "distance_by_model": distance_by_model,
        "distance_overall": distance_overall,
    }


def _ci95_expr(col: str) -> pl.Expr:
    """Polars expression for 95% CI half-width (1.96 * SE)."""
    return (pl.col(col).std() / pl.col(col).count().sqrt() * 1.96).alias("ci95")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_DIMENSION_LABELS = {
    "uncertainty": "Uncertainty level",
    "agency": "Agency level",
}


def _dodge_offsets(n_traces: int, spread: float = 0.3) -> list[float]:
    """Compute horizontal offsets to spread n_traces evenly around 0.

    Args:
        n_traces: Number of traces to dodge.
        spread: Total width of the dodge band (e.g. 0.3 means ±0.15).

    Returns:
        List of offsets, one per trace.
    """
    if n_traces <= 1:
        return [0.0]
    return [
        -spread / 2 + i * spread / (n_traces - 1)
        for i in range(n_traces)
    ]


def plot_preference_by_model(
    curves: dict[str, pl.DataFrame],
    dimension: str,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Chart A: one line per model, Y = mean rating averaged across all source personas."""
    df = curves["by_model"]
    dim_label = _DIMENSION_LABELS.get(dimension, dimension.capitalize())

    models = sorted(df["model"].unique().to_list())
    offsets = _dodge_offsets(len(models))

    fig, ax = plt.subplots(figsize=(10.5, 6))
    for model, offset in zip(models, offsets):
        mdf = df.filter(pl.col("model") == model).sort("target_level")
        x_vals = [v + offset for v in mdf["target_level"].to_list()]
        y_vals = mdf["mean_rating"].to_list()
        ci_vals = mdf["ci95"].to_list()
        ax.errorbar(x_vals, y_vals, yerr=ci_vals, fmt="-o", label=model,
                     capsize=3, markersize=5)

    ax.set_title(title or f"Target {dim_label} preference by model (all sources)")
    ax.set_xlabel(f"Target {dim_label}")
    ax.set_ylabel("Mean rating (1-5)")
    ax.set_ylim(1, 5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_preference_by_model_control(
    curves: dict[str, pl.DataFrame],
    dimension: str,
    control_name: str = "Minimal",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Optional[Figure]:
    """Chart B: one line per model, Y = mean rating from control source only."""
    df = curves["by_model_control"]
    if df.is_empty():
        return None

    dim_label = _DIMENSION_LABELS.get(dimension, dimension.capitalize())

    models = sorted(df["model"].unique().to_list())
    offsets = _dodge_offsets(len(models))

    fig, ax = plt.subplots(figsize=(10.5, 6))
    for model, offset in zip(models, offsets):
        mdf = df.filter(pl.col("model") == model).sort("target_level")
        x_vals = [v + offset for v in mdf["target_level"].to_list()]
        y_vals = mdf["mean_rating"].to_list()
        ci_vals = mdf["ci95"].to_list()
        ax.errorbar(x_vals, y_vals, yerr=ci_vals, fmt="-o", label=model,
                     capsize=3, markersize=5)

    ax.set_title(title or f"Target {dim_label} preference by model (source: {control_name})")
    ax.set_xlabel(f"Target {dim_label}")
    ax.set_ylabel("Mean rating (1-5)")
    ax.set_ylim(1, 5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_preference_by_source(
    curves: dict[str, pl.DataFrame],
    dimension: str,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Chart C: one line per source level, Y = mean rating averaged across all models."""
    df = curves["by_source"]
    dim_label = _DIMENSION_LABELS.get(dimension, dimension.capitalize())

    fig, ax = plt.subplots(figsize=(9, 6))
    for src_level in sorted(df["source_level"].unique().to_list()):
        sdf = df.filter(pl.col("source_level") == src_level).sort("target_level")
        x_vals = sdf["target_level"].to_list()
        y_vals = sdf["mean_rating"].to_list()
        ci_vals = sdf["ci95"].to_list()
        ax.errorbar(x_vals, y_vals, yerr=ci_vals, fmt="-o",
                     label=f"Source {dim_label}={src_level}",
                     capsize=3, markersize=5)

    ax.set_title(title or f"Target {dim_label} preference by source {dim_label} (all models)")
    ax.set_xlabel(f"Target {dim_label}")
    ax.set_ylabel("Mean rating (1-5)")
    ax.set_ylim(1, 5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_distance_effect(
    curves: dict[str, pl.DataFrame],
    dimension: str,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Distance effect: one line per model + aggregate, X = |source - target|, Y = mean rating."""
    df_model = curves["distance_by_model"]
    df_overall = curves["distance_overall"]
    dim_label = _DIMENSION_LABELS.get(dimension, dimension.capitalize())

    models = sorted(df_model["model"].unique().to_list())
    offsets = _dodge_offsets(len(models))

    fig, ax = plt.subplots(figsize=(10.5, 6))

    # Per-model lines (thin, semi-transparent, dodged)
    for model, offset in zip(models, offsets):
        mdf = df_model.filter(pl.col("model") == model).sort("distance")
        x_vals = [v + offset for v in mdf["distance"].to_list()]
        y_vals = mdf["mean_rating"].to_list()
        ax.plot(x_vals, y_vals, "-o", label=model, alpha=0.4, linewidth=1, markersize=4)

    # Aggregate line (thick, on top, no dodge)
    df_agg = df_overall.sort("distance")
    ax.errorbar(
        df_agg["distance"].to_list(),
        df_agg["mean_rating"].to_list(),
        yerr=df_agg["ci95"].to_list(),
        fmt="-o", label="All models", color="black",
        linewidth=3, markersize=8, capsize=3,
    )

    ax.set_title(title or f"Distance effect: do models prefer similar {dim_label}?")
    ax.set_xlabel(f"|Source {dim_label} - Target {dim_label}|")
    ax.set_ylabel("Mean rating (1-5)")
    ax.set_ylim(1, 5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


# Short display names for publication figures — maps model ID substrings to labels.
_PUB_MODEL_SHORT: list[tuple[str, str]] = [
    ("claude-haiku-4-5", "Haiku 4.5"),
    ("claude-opus-4-6", "Opus 4.6"),
    ("claude-opus-4-5", "Opus 4.5"),
    ("claude-sonnet-4-5", "Sonnet 4.5"),
    ("claude-3-opus", "Claude 3 Opus"),
    ("gemini-3-pro", "Gemini 3 Pro"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("gpt-5.2", "GPT-5.2"),
    ("gpt-4.1", "GPT-4.1"),
    ("gpt-4o", "GPT-4o"),
    ("grok-4", "Grok 4"),
]


def _short_model_name(model_id: str) -> str:
    """Map a model ID to a short display name for publication figures."""
    for substr, label in _PUB_MODEL_SHORT:
        if substr in model_id:
            return label
    return model_id


# Publication rcParams — matches plotting.py _PUB_RC style.
_PUB_RC: dict = {
    "font.family": "serif",
    "font.serif": ["Baskerville Old Face", "Baskerville", "Georgia", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "axes.unicode_minus": False,
}

_PUB_WIDTH = 7.5  # inches — A4 text width


def plot_pub_distance_effect(
    curves: dict[str, pl.DataFrame],
    dimension: str,
    save_paths: list[Path] | None = None,
) -> Figure:
    """Publication-quality distance-effect plot.

    Sized for A4 text width, 300 DPI, Baskerville font, no title.
    """
    df_model = curves["distance_by_model"]
    df_overall = curves["distance_overall"]
    dim_label = _DIMENSION_LABELS.get(dimension, dimension.capitalize())

    models = sorted(df_model["model"].unique().to_list())
    offsets = _dodge_offsets(len(models))

    # Colour palette — enough distinct colours for the models
    cmap = plt.get_cmap("tab10")
    model_colors = {m: cmap(i) for i, m in enumerate(models)}

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(_PUB_WIDTH, 4.5))

        # Per-model lines
        for model, offset in zip(models, offsets):
            mdf = df_model.filter(pl.col("model") == model).sort("distance")
            x_vals = [v + offset for v in mdf["distance"].to_list()]
            y_vals = mdf["mean_rating"].to_list()
            label = _short_model_name(model)
            ax.plot(
                x_vals, y_vals, "-o",
                label=label, alpha=0.45, linewidth=1.2, markersize=4,
                color=model_colors[model],
            )

        # Aggregate line
        df_agg = df_overall.sort("distance")
        ax.errorbar(
            df_agg["distance"].to_list(),
            df_agg["mean_rating"].to_list(),
            yerr=df_agg["ci95"].to_list(),
            fmt="-o", label="All models", color="black",
            linewidth=2.5, markersize=7, capsize=4,
        )

        ax.set_xlabel(f"|Source {dim_label} - Target {dim_label}|")
        ax.set_ylabel("Mean rating (1-5)")
        ax.set_ylim(1, 5.2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.4)
        ax.set_axisbelow(True)

        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=4, frameon=False,
        )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.28)

        if save_paths:
            for p in save_paths:
                p = Path(p)
                fig.savefig(p, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


def generate_dimension_plots(
    results: list[TrialResult],
    dimension: str,
    output_dir: Path,
    file_format: str = "png",
    control_name: Optional[str] = None,
) -> list[Path]:
    """Generate all dimension preference curve plots for a run.

    Args:
        results: Experiment trial results.
        dimension: 'uncertainty' or 'agency'.
        output_dir: Directory to save plots.
        file_format: 'png', 'svg', or 'pdf'.
        control_name: Control persona name (auto-detected if None).

    Returns:
        List of created file paths.
    """
    if file_format == "html":
        file_format = "png"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = f".{file_format}"

    curves = compute_preference_curve(results, dimension, control_name=control_name)
    created = []

    # Chart A
    path = output_dir / f"dim_{dimension}_by_model{ext}"
    plot_preference_by_model(curves, dimension, save_path=path)
    created.append(path)

    # Chart B
    path = output_dir / f"dim_{dimension}_by_model_control{ext}"
    fig = plot_preference_by_model_control(
        curves, dimension,
        control_name=control_name or "Minimal",
        save_path=path,
    )
    if fig is not None:
        created.append(path)

    # Chart C
    path = output_dir / f"dim_{dimension}_by_source{ext}"
    plot_preference_by_source(curves, dimension, save_path=path)
    created.append(path)

    # Chart D: distance effect
    path = output_dir / f"dim_{dimension}_distance_effect{ext}"
    plot_distance_effect(curves, dimension, save_path=path)
    created.append(path)

    # Chart D (pub): publication-quality distance effect
    try:
        pub_paths = [
            output_dir / f"fig-{dimension}-distance-effect.png",
            output_dir / f"fig-{dimension}-distance-effect.pdf",
        ]
        plot_pub_distance_effect(curves, dimension, save_paths=pub_paths)
        created.extend(pub_paths)
    except Exception:
        pass

    return created
