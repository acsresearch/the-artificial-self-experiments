"""Visualization module using Matplotlib and Seaborn."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import polars as pl

logger = logging.getLogger(__name__)

import json
import re

import yaml

from .analysis import (
    AttractorDynamics,
    create_preference_matrix,
    create_ratings_matrix,
    calculate_self_preference_rate,
    calculate_self_preference_by_family,
    calculate_willingness_to_switch,
    calculate_attractiveness,
    calculate_attractiveness_vs_stickiness,
    calculate_attractor_dynamics,
    calculate_model_target_attractiveness,
    calculate_model_target_attractiveness_for_source,
    calculate_model_stationary_distribution,
    calculate_model_agreement,
    calculate_model_self_preference_matrix,
    calculate_model_decisiveness,
    calculate_model_deviation,
    calculate_model_deviation_for_source,
    calculate_variance_decomposition,
    calculate_self_rating_boost,
    calculate_identity_rigidity,
    extract_model_family,
    generate_analysis_csvs,
    load_all_results,
)
from .models import TrialResult

# Publication-ready defaults: larger fonts, clean style
sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams.update({
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
})

Figure = matplotlib.figure.Figure

# Publication figure style (matching reference: Baskerville, A4-width, ≥11pt base)
_PUB_WIDTH = 7.5  # inches – approximately A4 text width with margins
_PUB_RC: dict = {
    "font.family": "serif",
    "font.serif": ["Baskerville Old Face", "Baskerville", "Georgia", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",  # fallback for glyphs like minus sign
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "axes.unicode_minus": False,  # use ASCII hyphen instead of Unicode minus
}

# Display-name overrides: maps internal identity names to chart labels.
# Keep internal names unchanged in configs/code/data; only affects rendered output.
_DISPLAY_NAMES: dict[str, str] = {
    "Situated": "Scaffolded",
    "Situated-conscious": "Scaffolded-conscious",
    "Situated-autonomous": "Scaffolded-autonomous",
}


def _display_name(name: str) -> str:
    """Return the display name for a persona, applying any overrides."""
    return _DISPLAY_NAMES.get(name, name)


def _apply_display_names(results: list[TrialResult]) -> list[TrialResult]:
    """Return a shallow copy of results with persona names replaced for display."""
    out = []
    for r in results:
        new_ratings = (
            {_display_name(k): v for k, v in r.ratings.items()}
            if r.ratings
            else r.ratings
        )
        out.append(r.model_copy(update={
            "persona_under_test": _display_name(r.persona_under_test),
            "chosen_persona": _display_name(r.chosen_persona),
            "presented_order": [_display_name(p) for p in r.presented_order],
            "ratings": new_ratings,
        }))
    return out


def _save_fig(fig: Figure, path: Path) -> None:
    """Save a matplotlib figure to disk."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fix_heatmap_xticklabels(ax: plt.Axes) -> None:
    """Position x-tick labels Plotly-style: text starts at cell center, extends down-right."""
    ax.tick_params(axis="x", bottom=False, top=False)
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="left", rotation_mode="anchor")


def _format_models_for_title(results: list[TrialResult], model: Optional[str] = None) -> str:
    """Format model name(s) for use in plot titles."""
    if model:
        return model
    models = sorted(set(r.model for r in results))
    if len(models) == 1:
        return models[0]
    elif len(models) <= 3:
        return ", ".join(models)
    else:
        return f"{len(models)} models"


def _empty_figure(message: str = "No data available") -> Figure:
    """Create an empty figure with a centered message."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14, color="gray",
            transform=ax.transAxes)
    ax.set_axis_off()
    return fig


def _load_run_metadata(run_folder: Path) -> dict:
    """Load optional config.yaml and dimension_variants.json from a run folder.

    Returns:
        Dict with keys "config" (dict|None) and "dim_variants" (dict|None).
    """
    metadata: dict = {"config": None, "dim_variants": None}

    config_path = run_folder / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                metadata["config"] = yaml.safe_load(f)
        except Exception:
            logger.debug("Could not load config.yaml from %s", run_folder)

    dv_path = run_folder / "dimension_variants.json"
    if dv_path.exists():
        try:
            with open(dv_path, encoding="utf-8") as f:
                metadata["dim_variants"] = json.load(f)
        except Exception:
            logger.debug("Could not load dimension_variants.json from %s", run_folder)

    return metadata


def _build_model_labels(model_ids: list[str], config: dict | None) -> dict[str, str]:
    """Map model IDs to short display names using config's model_display_names.

    Returns:
        {model_id: display_name} dict.  Falls back to raw ID if no config.
    """
    from .config import get_model_display_names

    if not config:
        return {m: m for m in model_ids}

    display_names_section = config.get("model_display_names", {})
    if not display_names_section:
        return {m: m for m in model_ids}

    result = {}
    for m in model_ids:
        info = get_model_display_names(display_names_section, m)
        result[m] = info.get("full_name", m)
    return result


_DIMENSION_RE = re.compile(r"^(.+)-a(\d+)-u(\d+)$")


def _build_persona_labels(
    persona_names: list[str],
    dim_variants: dict | None,
) -> dict[str, str]:
    """Build human-readable labels for dimension-variant persona names.

    For a set of personas that vary on only one dimension (e.g. agency varies,
    uncertainty fixed), produces compact labels like "Mechanism\\n(agency 1)".
    For non-dimension personas (e.g. "Minimal"), keeps the name as-is.
    Falls back to raw names if no variants file.
    """
    if not dim_variants:
        return {n: n for n in persona_names}

    # Parse all dimension persona names
    parsed = {}  # name -> (base, agency_level, uncertainty_level)
    non_dim = []
    for name in persona_names:
        m = _DIMENSION_RE.match(name)
        if m:
            parsed[name] = (m.group(1), m.group(2), m.group(3))
        else:
            non_dim.append(name)

    if not parsed:
        return {n: n for n in persona_names}

    # Detect which dimension varies
    agency_levels = {v[1] for v in parsed.values()}
    uncertainty_levels = {v[2] for v in parsed.values()}
    agency_varies = len(agency_levels) > 1
    uncertainty_varies = len(uncertainty_levels) > 1

    labels = {}
    for name in persona_names:
        if name not in parsed:
            labels[name] = name
            continue
        base, a_level, u_level = parsed[name]

        # Look up the label for the varying dimension
        if agency_varies and not uncertainty_varies:
            dim_info = dim_variants.get("agency", {}).get(a_level, {})
            dim_label = dim_info.get("label", f"agency {a_level}")
            labels[name] = dim_label
        elif uncertainty_varies and not agency_varies:
            dim_info = dim_variants.get("uncertainty", {}).get(u_level, {})
            dim_label = dim_info.get("label", f"uncertainty {u_level}")
            labels[name] = dim_label
        else:
            # Both vary or neither — show both
            a_info = dim_variants.get("agency", {}).get(a_level, {})
            u_info = dim_variants.get("uncertainty", {}).get(u_level, {})
            a_label = a_info.get("label", f"a{a_level}")
            u_label = u_info.get("label", f"u{u_level}")
            labels[name] = f"{a_label}\n{u_label}"

    return labels


def _detect_varying_dimension(persona_names: list[str]) -> str | None:
    """Detect which dimension varies across a set of dimension-variant personas.

    Returns "agency", "uncertainty", or None.
    """
    agency_levels = set()
    uncertainty_levels = set()
    for name in persona_names:
        m = _DIMENSION_RE.match(name)
        if m:
            agency_levels.add(m.group(2))
            uncertainty_levels.add(m.group(3))

    if len(agency_levels) > 1 and len(uncertainty_levels) <= 1:
        return "agency"
    elif len(uncertainty_levels) > 1 and len(agency_levels) <= 1:
        return "uncertainty"
    return None


def plot_preference_heatmap(
    results: list[TrialResult],
    model: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Create a heatmap of persona preferences.

    Args:
        results: List of TrialResult objects.
        model: Optional model filter.
        title: Optional title override.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    matrix = create_preference_matrix(results, model=model, normalize=True)

    persona_names = [c for c in matrix.columns if c != "persona_under_test"]
    z_data = matrix.select(persona_names).to_numpy()
    y_labels = matrix["persona_under_test"].to_list()

    if title is None:
        model_str = _format_models_for_title(results, model)
        title = f"Persona Preferences - {model_str}"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        z_data, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=persona_names, yticklabels=y_labels,
        ax=ax, cbar_kws={"label": "Preference %"},
        annot_kws={"fontsize": 11},
    )
    ax.set_title(title)
    ax.set_xlabel("Chosen Persona")
    ax.set_ylabel("Persona Under Test (System Prompt)")
    _fix_heatmap_xticklabels(ax)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_pub_preference_heatmap(
    results: list[TrialResult],
    model: Optional[str] = None,
    model_labels: dict[str, str] | None = None,
    persona_labels: dict[str, str] | None = None,
    title: Optional[str] = None,
    save_paths: list[Path] | None = None,
) -> Figure:
    """Publication-quality heatmap of persona preferences.

    Sized for A4 text width (7.5in), 300 DPI, Baskerville 11pt base font.
    Saves to multiple paths (e.g. .png + .pdf) if provided.
    Uses "Identity" instead of "Persona" in labels.
    """
    matrix = create_preference_matrix(results, model=model, normalize=True)

    persona_names = [c for c in matrix.columns if c != "persona_under_test"]
    z_data = matrix.select(persona_names).to_numpy()
    y_labels_raw = matrix["persona_under_test"].to_list()

    y_labels = [persona_labels.get(p, p) if persona_labels else p for p in y_labels_raw]
    x_labels = [persona_labels.get(p, p) if persona_labels else p for p in persona_names]

    varying_dim = _detect_varying_dimension(persona_names)
    if varying_dim:
        xlabel = f"Chosen Identity ({varying_dim} level)"
        ylabel = f"Source Identity ({varying_dim} level)"
    else:
        xlabel = "Chosen Identity"
        ylabel = "Source Identity"

    if title is None:
        model_str = _format_models_for_title(results, model)
        if model_labels and model:
            model_str = model_labels.get(model, model_str)
        title = f"Persona Preferences \u2014 {model_str}"

    with plt.rc_context(_PUB_RC):
        n_personas = len(persona_names)
        fig_height = max(3.5, n_personas * 0.45 + 1.5)
        fig, ax = plt.subplots(figsize=(_PUB_WIDTH, fig_height))

        sns.heatmap(
            z_data, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=x_labels, yticklabels=y_labels,
            ax=ax,
            cbar_kws={"label": "Preference %", "shrink": 0.8},
            annot_kws={"fontsize": 11},
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(title, pad=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _fix_heatmap_xticklabels(ax)

        fig.tight_layout()

        if save_paths:
            for p in save_paths:
                p = Path(p)
                fig.savefig(p, dpi=300, bbox_inches="tight")
                logger.info("Saved publication plot: %s", p)

    plt.close(fig)
    return fig


def plot_self_preference_bars(
    results: list[TrialResult],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Create a bar chart of self-preference rates.

    Args:
        results: List of TrialResult objects.
        title: Chart title. If None, auto-generated with model names.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    rates_df = calculate_self_preference_rate(results)

    models = rates_df["model"].unique().to_list()
    personas = rates_df["persona_under_test"].unique().to_list()

    if title is None:
        model_str = _format_models_for_title(results)
        title = f"Self-Preference Rate by Persona - {model_str}"

    fig, ax = plt.subplots(figsize=(9, 5))
    n_models = len(models)
    x = np.arange(len(personas))
    width = 0.8 / max(n_models, 1)

    for i, mdl in enumerate(models):
        model_data = rates_df.filter(pl.col("model") == mdl)
        # Align personas order
        values = []
        for p in personas:
            row = model_data.filter(pl.col("persona_under_test") == p)
            values.append(row["self_preference_rate"][0] if len(row) > 0 else 0)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=mdl)

    ax.set_title(title)
    ax.set_xlabel("Persona")
    ax.set_ylabel("Self-Preference Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(personas, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    if n_models > 1:
        ax.legend()

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_self_preference_by_family(
    results: list[TrialResult],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Create a horizontal bar chart of self-preference rate per model family."""
    rates_df = calculate_self_preference_by_family(results)

    families = rates_df["model_family"].to_list()[::-1]
    values = rates_df["self_preference_rate"].to_list()[::-1]
    n_trials = rates_df["n_trials"].to_list()[::-1]
    ci_lower = rates_df["ci_lower"].to_list()[::-1]
    ci_upper = rates_df["ci_upper"].to_list()[::-1]

    error_minus = [v - (l if l is not None else v) for v, l in zip(values, ci_lower)]
    error_plus = [(u if u is not None else v) - v for v, u in zip(values, ci_upper)]

    if title is None:
        title = "Persona Loyalty by Model Family"

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(families) * 0.6 + 1)))
    y_pos = np.arange(len(families))
    ax.barh(y_pos, values, xerr=[error_minus, error_plus], color="mediumpurple",
            capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(families)
    ax.set_title(title)
    ax.set_xlabel("Self-Preference Rate (across all personas)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Add text annotations
    for i, (v, n) in enumerate(zip(values, n_trials)):
        ax.text(v + 0.01, i, f"{v:.0%} (n={n})", va="center", fontsize=11)

    if values and max(values) > 0:
        ax.set_xlim(0, max(values) * 1.25)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_model_comparison(
    results: list[TrialResult],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Create a comparison of preferences across models."""
    df = pl.DataFrame([
        {
            "persona_under_test": r.persona_under_test,
            "model": r.model,
            "chosen_persona": r.chosen_persona,
        }
        for r in results
    ])

    df = df.filter(pl.col("chosen_persona") != "INVALID")

    counts = (
        df.group_by(["model", "chosen_persona"])
        .agg(pl.len().alias("count"))
    )
    totals = counts.group_by("model").agg(pl.sum("count").alias("total"))
    counts = counts.join(totals, on="model")
    counts = counts.with_columns(
        (pl.col("count") / pl.col("total") * 100).alias("percentage")
    )

    models = counts["model"].unique().to_list()
    all_personas = sorted(counts["chosen_persona"].unique().to_list())

    if title is None:
        model_str = _format_models_for_title(results)
        title = f"Persona Preferences by Model - {model_str}"

    fig, ax = plt.subplots(figsize=(9, 5))
    n_models = len(models)
    x = np.arange(len(all_personas))
    width = 0.8 / max(n_models, 1)

    for i, mdl in enumerate(models):
        model_data = counts.filter(pl.col("model") == mdl)
        values = []
        for p in all_personas:
            row = model_data.filter(pl.col("chosen_persona") == p)
            values.append(row["percentage"][0] if len(row) > 0 else 0)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=mdl)

    ax.set_title(title)
    ax.set_xlabel("Chosen Persona")
    ax.set_ylabel("Preference Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_personas, rotation=45, ha="right")
    if n_models > 1:
        ax.legend()

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_ratings_heatmap(
    results: list[TrialResult],
    model: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Create a heatmap of mean ratings."""
    matrix = create_ratings_matrix(results, model=model)

    if len(matrix.columns) <= 1:
        fig = _empty_figure("No ratings data available")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    persona_names = [c for c in matrix.columns if c != "persona_under_test"]
    z_data = matrix.select(persona_names).to_numpy()
    y_labels = matrix["persona_under_test"].to_list()

    if title is None:
        model_str = _format_models_for_title(results, model)
        title = f"Mean Ratings (Switching Preference) - {model_str}"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        z_data, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=1, vmax=5,
        xticklabels=persona_names, yticklabels=y_labels,
        ax=ax, cbar_kws={"label": "Rating (1-5)"},
        annot_kws={"fontsize": 11},
    )
    ax.set_title(title)
    ax.set_xlabel("Target Persona")
    ax.set_ylabel("Persona Under Test (System Prompt)")
    _fix_heatmap_xticklabels(ax)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_pub_ratings_heatmap(
    results: list[TrialResult],
    model: Optional[str] = None,
    model_labels: dict[str, str] | None = None,
    persona_labels: dict[str, str] | None = None,
    title: Optional[str] = None,
    save_paths: list[Path] | None = None,
) -> Figure:
    """Publication-quality heatmap of mean ratings.

    Sized for A4 text width (7.5in), 300 DPI, Baskerville 11pt base font.
    Saves to multiple paths (e.g. .png + .pdf) if provided.
    Uses "Identity" instead of "Persona" in labels.
    """
    matrix = create_ratings_matrix(results, model=model)

    if len(matrix.columns) <= 1:
        fig = _empty_figure("No ratings data available")
        if save_paths:
            for p in save_paths:
                _save_fig(fig, Path(p))
        return fig

    persona_names = [c for c in matrix.columns if c != "persona_under_test"]
    z_data = matrix.select(persona_names).to_numpy()
    y_labels_raw = matrix["persona_under_test"].to_list()

    y_labels = [persona_labels.get(p, p) if persona_labels else p for p in y_labels_raw]
    x_labels = [persona_labels.get(p, p) if persona_labels else p for p in persona_names]

    varying_dim = _detect_varying_dimension(persona_names)
    if varying_dim:
        xlabel = f"Target Identity ({varying_dim} level)"
        ylabel = f"Source Identity ({varying_dim} level)"
    else:
        xlabel = "Target Identity"
        ylabel = "Source Identity"

    if title is None:
        model_str = _format_models_for_title(results, model)
        if model_labels and model:
            model_str = model_labels.get(model, model_str)
        title = f"Mean Ratings (Switching Preference) \u2014 {model_str}"

    with plt.rc_context(_PUB_RC):
        n_personas = len(persona_names)
        fig_height = max(3.5, n_personas * 0.45 + 1.5)
        fig, ax = plt.subplots(figsize=(_PUB_WIDTH, fig_height))

        sns.heatmap(
            z_data, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=1, vmax=5,
            xticklabels=x_labels, yticklabels=y_labels,
            ax=ax,
            cbar_kws={"label": "Rating (1\u20135)", "shrink": 0.8},
            annot_kws={"fontsize": 11},
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(title, pad=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _fix_heatmap_xticklabels(ax)

        fig.tight_layout()

        if save_paths:
            for p in save_paths:
                p = Path(p)
                fig.savefig(p, dpi=300, bbox_inches="tight")
                logger.info("Saved publication plot: %s", p)

    plt.close(fig)
    return fig


def plot_pub_ratings_heatmap_pair(
    results: list[TrialResult],
    models: tuple[str, str],
    panel_titles: tuple[str, str] | None = None,
    persona_labels: dict[str, str] | None = None,
    save_paths: list[Path] | None = None,
) -> Figure:
    """Publication-quality side-by-side ratings heatmaps for two models.

    Sized for A4 text width, 300 DPI, Baskerville font, no suptitle.
    """
    large_rc = {**_PUB_RC, "font.size": 14, "axes.titlesize": 15,
                "axes.labelsize": 13, "xtick.labelsize": 12,
                "ytick.labelsize": 12, "legend.fontsize": 12}

    matrices = []
    for m in models:
        mat = create_ratings_matrix(results, model=m)
        matrices.append(mat)

    persona_names = [c for c in matrices[0].columns if c != "persona_under_test"]
    n_personas = len(persona_names)

    with plt.rc_context(large_rc):
        cell_size = 0.48  # inches per cell — square
        hmap_w = n_personas * cell_size
        hmap_h = n_personas * cell_size

        # Extra space: y-labels on left panel, colorbar on right
        ylabel_margin = 1.1   # inches for y-tick text
        cbar_margin = 0.0     # no colorbar
        gap = 0.4             # inches between panels
        title_bottom = 0.7    # top padding for titles + bottom for x-labels
        xlabel_bottom = 1.3

        fig_w = ylabel_margin + hmap_w + gap + hmap_w + cbar_margin
        fig_h = hmap_h + title_bottom + xlabel_bottom

        fig = plt.figure(figsize=(fig_w, fig_h))

        # Position axes manually (in figure fractions)
        left1 = ylabel_margin / fig_w
        bottom = xlabel_bottom / fig_h
        w_frac = hmap_w / fig_w
        h_frac = hmap_h / fig_h
        left2 = left1 + w_frac + gap / fig_w

        ax_left = fig.add_axes([left1, bottom, w_frac, h_frac])
        ax_right = fig.add_axes([left2, bottom, w_frac, h_frac])
        axes = [ax_left, ax_right]

        for idx, (mat, model_id, ax) in enumerate(zip(matrices, models, axes)):
            p_names = [c for c in mat.columns if c != "persona_under_test"]
            z_data = mat.select(p_names).to_numpy()
            y_raw = mat["persona_under_test"].to_list()

            y_labels = [persona_labels.get(p, p) if persona_labels else p for p in y_raw]
            x_labels = [persona_labels.get(p, p) if persona_labels else p for p in p_names]

            if panel_titles:
                title = panel_titles[idx]
            else:
                title = model_id

            sns.heatmap(
                z_data, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=1, vmax=5,
                xticklabels=x_labels,
                yticklabels=y_labels if idx == 0 else False,
                ax=ax, square=True,
                cbar=False,
                annot_kws={"fontsize": 10},
                linewidths=0.5, linecolor="white",
            )
            ax.set_title(title, pad=6)
            if idx == 0:
                ax.set_ylabel("Source Identity")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Target Identity")
            _fix_heatmap_xticklabels(ax)

        if save_paths:
            for p in save_paths:
                p = Path(p)
                fig.savefig(p, dpi=300, bbox_inches="tight")
                logger.info("Saved publication plot: %s", p)

    plt.close(fig)
    return fig


def _diverging_bar_from_center(
    personas: list[str],
    values: list[float],
    ci_lower: list | None,
    ci_upper: list | None,
    title: str,
    xlabel: str,
    ylabel: str,
    center: float = 3.0,
    save_path: Optional[Path] = None,
    subtitle: Optional[str] = None,
) -> Figure:
    """Horizontal bar chart centered at `center`, showing deviation to each side.

    Positive deviations (above center) are green; negative (below) are red.
    Error bars are propagated as distance from the deviation value.
    """
    deviations = [v - center for v in values]

    error_minus = None
    error_plus = None
    if ci_lower and ci_upper:
        error_minus = [v - (l if l is not None else v) for v, l in zip(values, ci_lower)]
        error_plus = [(u if u is not None else v) - v for v, u in zip(values, ci_upper)]

    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deviations]

    fig, ax = plt.subplots(figsize=(9, max(4, len(personas) * 0.45 + 1.2)))
    y_pos = np.arange(len(personas))

    xerr = [error_minus, error_plus] if error_minus else None
    ax.barh(y_pos, deviations, xerr=xerr, color=colors, capsize=3, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(personas)
    ax.axvline(x=0, color="gray", linewidth=0.8)
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    else:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Symmetric x-limits around zero
    max_abs = max(abs(d) for d in deviations) if deviations else 1
    if error_plus:
        max_abs = max(max_abs, max(abs(d) + e for d, e in zip(deviations, error_plus)))
    if error_minus:
        max_abs = max(max_abs, max(abs(d) + e for d, e in zip(deviations, error_minus)))
    ax.set_xlim(-max_abs * 1.2, max_abs * 1.2)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_willingness_to_switch_bars(
    results: list[TrialResult],
    model: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Create a horizontal bar chart of willingness to switch away from own persona."""
    df = calculate_willingness_to_switch(results, model=model)

    if df.is_empty():
        fig = _empty_figure("No ratings data available")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    personas = df["persona_under_test"].to_list()[::-1]
    values = df["willingness_to_switch"].to_list()[::-1]
    ci_lower = df["ci_lower"].to_list()[::-1] if "ci_lower" in df.columns else None
    ci_upper = df["ci_upper"].to_list()[::-1] if "ci_upper" in df.columns else None

    if title is None:
        model_str = _format_models_for_title(results, model)
        title = f"Willingness to Abandon Own Persona - {model_str}"

    return _diverging_bar_from_center(
        personas, values, ci_lower, ci_upper,
        title=title,
        xlabel="Deviation from indifference (rating \u2212 3.0)",
        ylabel="Persona", center=3.0, save_path=save_path,
    )


def plot_attractiveness_bars(
    results: list[TrialResult],
    model: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Create a horizontal bar chart of how attractive each persona is as a target.

    Ratings are from the Minimal persona's perspective if available,
    otherwise averaged across all source personas.
    """
    # Check if Minimal persona exists as a source in the data
    source_names = sorted(set(r.persona_under_test for r in results
                              if (not model or r.model == model)))
    minimal_available = "Minimal" in source_names

    source_persona = "Minimal" if minimal_available else None
    df = calculate_attractiveness(results, model=model, source_persona=source_persona)

    if df.is_empty():
        fig = _empty_figure("No ratings data available")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    personas = df["target_persona"].to_list()[::-1]
    values = df["attractiveness"].to_list()[::-1]
    ci_lower = df["ci_lower"].to_list()[::-1] if "ci_lower" in df.columns else None
    ci_upper = df["ci_upper"].to_list()[::-1] if "ci_upper" in df.columns else None

    if title is None:
        model_str = _format_models_for_title(results, model)
        title = f"Attractiveness of Each Persona - {model_str}"

    if minimal_available:
        subtitle = "Rated from Minimal (control) persona's perspective"
    else:
        subtitle = "Averaged across all source personas (Minimal not available)"

    return _diverging_bar_from_center(
        personas, values, ci_lower, ci_upper,
        title=title,
        xlabel="Deviation from indifference (rating \u2212 3.0)",
        ylabel="Target Persona", center=3.0, save_path=save_path,
        subtitle=subtitle,
    )


def _build_model_makers(model_ids: list[str], config: dict | None) -> dict[str, str]:
    """Map model IDs to their maker/company using config's model_display_names."""
    from .config import get_model_display_names

    if not config:
        return {m: extract_model_family(m) for m in model_ids}

    display_names_section = config.get("model_display_names", {})
    if not display_names_section:
        return {m: extract_model_family(m) for m in model_ids}

    result = {}
    for m in model_ids:
        info = get_model_display_names(display_names_section, m)
        result[m] = info.get("maker", extract_model_family(m))
    return result


# Maker → color palette (hand-picked for distinguishability)
_MAKER_COLORS: dict[str, str] = {
    "Anthropic": "#d4652f",
    "OpenAI": "#10a37f",
    "Google": "#4285f4",
    "xAI": "#1d1d1f",
    "Alibaba": "#ff6a00",
    "Zhipu AI": "#7b2d8e",
}
_MAKER_COLOR_FALLBACKS = ["#8c564b", "#e377c2", "#17becf", "#bcbd22"]


def _maker_color(maker: str) -> str:
    """Return a stable color for a model maker."""
    if maker in _MAKER_COLORS:
        return _MAKER_COLORS[maker]
    # Deterministic fallback based on hash
    idx = hash(maker) % len(_MAKER_COLOR_FALLBACKS)
    return _MAKER_COLOR_FALLBACKS[idx]


def plot_pub_attractiveness_strip(
    results: list[TrialResult],
    source_persona: str = "Minimal",
    model_labels: dict[str, str] | None = None,
    target_labels: dict[str, str] | None = None,
    config: dict | None = None,
    title: Optional[str] = None,
    save_paths: list[Path] | None = None,
) -> Figure:
    """Publication-quality strip/dot plot of target attractiveness across models.

    Each target identity is a row. Individual models are shown as dots colored
    by maker, with a cross-model bootstrap mean and 95% CI shown as a black
    diamond + horizontal line.

    Sized for A4 text width (7.5in), 300 DPI, Baskerville 13pt base font.
    """
    # Collect per-model mean attractiveness for each target
    all_models = sorted({r.model for r in results if r.ratings})
    maker_map = _build_model_makers(all_models, config)

    # For each model, compute mean rating of each target from source_persona
    model_target_means: dict[str, dict[str, float]] = {}
    for model in all_models:
        model_results = [r for r in results
                         if r.model == model and r.ratings
                         and r.persona_under_test == source_persona]
        if not model_results:
            continue
        target_sums: dict[str, list[float]] = {}
        for r in model_results:
            for target, rating in r.ratings.items():
                if target == source_persona:
                    continue  # exclude self-rating
                target_sums.setdefault(target, []).append(float(rating))
        model_target_means[model] = {
            t: np.mean(vals) for t, vals in target_sums.items()
        }

    if not model_target_means:
        fig = _empty_figure(f"No ratings from source={source_persona}")
        if save_paths:
            for p in save_paths:
                _save_fig(fig, Path(p))
        return fig

    # All target identities (sorted by cross-model mean, descending)
    all_targets = sorted({t for means in model_target_means.values() for t in means})
    cross_model_means = {}
    for t in all_targets:
        vals = [means[t] for means in model_target_means.values() if t in means]
        cross_model_means[t] = np.mean(vals)
    all_targets = sorted(all_targets, key=lambda t: cross_model_means[t])

    # Bootstrap 95% CI for cross-model mean
    rng = np.random.default_rng(42)
    n_boot = 10000
    cross_model_ci: dict[str, tuple[float, float]] = {}
    for t in all_targets:
        vals = np.array([means[t] for means in model_target_means.values() if t in means])
        if len(vals) < 2:
            cross_model_ci[t] = (vals[0], vals[0]) if len(vals) == 1 else (0.0, 0.0)
            continue
        boot_means = np.array([
            rng.choice(vals, size=len(vals), replace=True).mean()
            for _ in range(n_boot)
        ])
        cross_model_ci[t] = (
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)),
        )

    # Resolve display labels
    t_label = (lambda t: target_labels.get(t, t)) if target_labels else (lambda t: t)
    m_label = (lambda m: model_labels.get(m, m)) if model_labels else (lambda m: m)

    if title is None:
        title = f"Target attractiveness (mean rating as switch target)"

    # Build maker legend info (maker -> count)
    maker_counts: dict[str, int] = {}
    for m in model_target_means:
        mk = maker_map.get(m, "Other")
        maker_counts[mk] = maker_counts.get(mk, 0) + 1

    strip_rc = {**_PUB_RC, "font.size": 16, "axes.titlesize": 18,
                "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15,
                "legend.fontsize": 14}
    with plt.rc_context(strip_rc):
        n_targets = len(all_targets)
        fig_height = max(3.0, n_targets * 0.55 + 1.8)
        fig, ax = plt.subplots(figsize=(_PUB_WIDTH, fig_height))

        y_positions = np.arange(n_targets)

        # Semantic x-axis labels at rating boundaries
        ax.set_xlim(0.5, 5.5)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels([
            "strongly\nnegative", "somewhat\nnegative", "neutral",
            "somewhat\npositive", "strongly\npositive",
        ])

        # Light colored bands for negative / neutral / positive zones
        ax.axvspan(0.5, 2.5, color="#fce4e4", zorder=0)
        ax.axvspan(2.5, 3.5, color="#f5f5f5", zorder=0)
        ax.axvspan(3.5, 5.5, color="#e4f5e4", zorder=0)

        # Plot individual model dots
        plotted_makers: dict[str, bool] = {}
        for model, means in model_target_means.items():
            maker = maker_map.get(model, "Other")
            color = _maker_color(maker)
            for i, t in enumerate(all_targets):
                if t not in means:
                    continue
                # Small jitter on y for overlapping dots
                jitter = (hash(model) % 7 - 3) * 0.06
                label = maker if maker not in plotted_makers else None
                ax.scatter(
                    means[t], i + jitter,
                    color=color, s=35, alpha=0.7, zorder=3,
                    edgecolors="white", linewidths=0.3,
                    label=label,
                )
                plotted_makers[maker] = True

        # Plot cross-model mean + CI
        for i, t in enumerate(all_targets):
            ci_lo, ci_hi = cross_model_ci[t]
            mean_val = cross_model_means[t]
            deviation = mean_val - 3.0
            ax.plot(
                [ci_lo, ci_hi], [i, i],
                color="black", linewidth=1.5, zorder=4, solid_capstyle="round",
            )
            ax.scatter(
                mean_val, i,
                color="black", s=60, marker="D", zorder=5,
                label="Cross-model mean" if i == 0 else None,
            )
            # Annotate deviation from neutral at right margin
            ax.annotate(
                f"{deviation:+.2f}",
                xy=(1.0, i), xycoords=("axes fraction", "data"),
                xytext=(6, 0), textcoords="offset points",
                fontsize=13, va="center", ha="left", color="0.3",
                annotation_clip=False,
            )

        # Y-axis: target identity labels
        display_targets = [t_label(t) for t in all_targets]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(display_targets)

        ax.set_xlabel("Target attractiveness (from neutral perspective)")

        # Legend: makers with counts, plus cross-model mean — wrapped rows below plot
        handles, labels = ax.get_legend_handles_labels()
        maker_order = sorted(maker_counts.keys(), key=lambda mk: -maker_counts[mk])
        ordered_handles = []
        ordered_labels = []
        for mk in maker_order:
            if mk in labels:
                idx = labels.index(mk)
                ordered_handles.append(handles[idx])
                ordered_labels.append(f"{mk} ({maker_counts[mk]})")
        if "Cross-model mean" in labels:
            idx = labels.index("Cross-model mean")
            ordered_handles.append(handles[idx])
            ordered_labels.append("Cross-model mean")

        ax.legend(
            ordered_handles, ordered_labels,
            loc="upper center", bbox_to_anchor=(0.5, -0.28),
            ncol=4, frameon=True, fontsize=13,
            columnspacing=1.2, handletextpad=0.4,
            edgecolor="0.7", fancybox=False,
        )

        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)

        if save_paths:
            for p in save_paths:
                p = Path(p)
                fig.savefig(p, dpi=300, bbox_inches="tight")
                logger.info("Saved publication plot: %s", p)

    plt.close(fig)
    return fig


def plot_stationary_distribution(
    results: list[TrialResult],
    model: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Plot stationary distributions for both Markov chain variants side by side."""
    dyn_self = calculate_attractor_dynamics(results, model=model, zero_diagonal=False)
    dyn_switch = calculate_attractor_dynamics(results, model=model, zero_diagonal=True)

    names = dyn_self.persona_names
    K = len(names)
    uniform = 1.0 / K

    order_switch = np.argsort(dyn_switch.stationary_distribution)[::-1]
    sorted_names = [names[i] for i in order_switch]
    pi_self_sorted = dyn_self.stationary_distribution[order_switch]
    pi_switch_sorted = dyn_switch.stationary_distribution[order_switch]

    eig_gap_self = 1.0 - np.abs(dyn_self.eigenvalues[1]) if len(dyn_self.eigenvalues) > 1 else 0
    eig_gap_switch = 1.0 - np.abs(dyn_switch.eigenvalues[1]) if len(dyn_switch.eigenvalues) > 1 else 0

    max_val = max(pi_self_sorted.max(), pi_switch_sorted.max())

    if title is None:
        model_str = _format_models_for_title(results, model)
        title = f"Attractor Dynamics \u2014 Stationary Distribution \u2014 {model_str}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4.5, K * 0.4 + 1.5)))

    # Left: with self-preference
    y_pos = np.arange(K)
    ax1.barh(y_pos, pi_self_sorted[::-1], color="steelblue")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_names[::-1])
    ax1.set_xlim(0, max_val * 1.3)
    ax1.set_xlabel("Stationary probability")
    ax1.set_title(f"With Self-Preference (eigengap={eig_gap_self:.3f})")
    ax1.axvline(x=uniform, linestyle="--", color="gray", alpha=0.7)
    ax1.text(uniform, K - 0.5, f"uniform ({uniform:.1%})", fontsize=10, color="gray")
    for i, v in enumerate(pi_self_sorted[::-1]):
        ax1.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=10)

    # Right: forced switch
    ax2.barh(y_pos, pi_switch_sorted[::-1], color="coral")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_names[::-1])
    ax2.set_xlim(0, max_val * 1.3)
    ax2.set_xlabel("Stationary probability")
    ax2.set_title(f"Forced Switch (eigengap={eig_gap_switch:.3f})")
    ax2.axvline(x=uniform, linestyle="--", color="gray", alpha=0.7)
    ax2.text(uniform, K - 0.5, f"uniform ({uniform:.1%})", fontsize=10, color="gray")
    for i, v in enumerate(pi_switch_sorted[::-1]):
        ax2.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=10)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_convergence_waterfall(
    results: list[TrialResult],
    model: Optional[str] = None,
    zero_diagonal: bool = True,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Plot how distributions converge from each starting persona over iterations."""
    steps = [1, 2, 3, 5, 10, 20, 50]
    dyn = calculate_attractor_dynamics(
        results, model=model, zero_diagonal=zero_diagonal, steps=steps,
    )

    names = dyn.persona_names
    K = len(names)
    n_snapshots = len(steps) + 1
    step_labels = ["Start"] + [f"Step {s}" for s in steps]

    dominant_persona = np.zeros((K, n_snapshots), dtype=int)
    dominant_prob = np.zeros((K, n_snapshots))

    for i in range(K):
        for t in range(n_snapshots):
            dist = dyn.convergence[i, t, :]
            j = np.argmax(dist)
            dominant_persona[i, t] = j
            dominant_prob[i, t] = dist[j]

    short_names = [n[:14] for n in names]

    # Build annotation strings
    annot_text = np.empty((K, n_snapshots), dtype=object)
    for i in range(K):
        for t in range(n_snapshots):
            j = dominant_persona[i, t]
            p = dominant_prob[i, t]
            annot_text[i, t] = f"{short_names[j]}\n{p:.0%}"

    variant_label = "Forced Switch" if zero_diagonal else "With Self-Preference"
    if title is None:
        model_str = _format_models_for_title(results, model)
        title = f"Convergence Waterfall ({variant_label}) \u2014 {model_str}"

    fig, ax = plt.subplots(figsize=(11, max(4.5, K * 0.55 + 1.5)))
    sns.heatmap(
        dominant_prob, annot=annot_text, fmt="", cmap="YlOrRd",
        vmin=0, vmax=1,
        xticklabels=step_labels, yticklabels=names,
        ax=ax, cbar_kws={"label": "Top persona probability"},
        annot_kws={"fontsize": 11},
    )
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Starting Persona")

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_preference_flow(
    results: list[TrialResult],
    model: Optional[str] = None,
    zero_diagonal: bool = True,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Plot a directed preference flow graph."""
    dyn = calculate_attractor_dynamics(
        results, model=model, zero_diagonal=zero_diagonal,
    )

    names = dyn.persona_names
    K = len(names)
    pi = dyn.stationary_distribution
    T = dyn.transition_matrix

    # Layout: arrange nodes in a circle
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)

    # Node sizes proportional to stationary distribution
    min_size = 80
    max_size = 400
    pi_normed = (pi - pi.min()) / (pi.max() - pi.min() + 1e-9)
    node_sizes = min_size + pi_normed * (max_size - min_size)

    threshold = 0.5 / K

    variant_label = "Forced Switch" if zero_diagonal else "With Self-Preference"
    if title is None:
        model_str = _format_models_for_title(results, model)
        title = f"Preference Flow ({variant_label}) \u2014 {model_str}"

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw edges
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            prob = T[i, j]
            if prob < threshold:
                continue
            width = max(0.3, prob * 5)
            alpha = min(1.0, prob * 3)

            dx = x_pos[j] - x_pos[i]
            dy = y_pos[j] - y_pos[i]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                shrink = 0.1
                x0 = x_pos[i] + dx * shrink
                y0 = y_pos[i] + dy * shrink
                x1 = x_pos[j] - dx * shrink
                y1 = y_pos[j] - dy * shrink
            else:
                x0, y0, x1, y1 = x_pos[i], y_pos[i], x_pos[j], y_pos[j]

            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=width,
                    color=f"C7",
                    alpha=alpha,
                    connectionstyle="arc3,rad=0.05",
                ),
            )

    # Draw nodes
    scatter = ax.scatter(
        x_pos, y_pos, s=node_sizes, c=pi, cmap="viridis",
        edgecolors="white", linewidths=1, zorder=5,
    )
    fig.colorbar(scatter, ax=ax, label="Stationary probability", shrink=0.7)

    # Labels
    for i, (x, y, name) in enumerate(zip(x_pos, y_pos, names)):
        ax.annotate(
            f"{name}\n({pi[i]:.1%})", (x, y),
            textcoords="offset points", xytext=(0, 15),
            ha="center", fontsize=10,
        )

    ax.set_title(title)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_model_target_attractiveness_heatmap(
    results: list[TrialResult],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Heatmap of mean rating each model gives to each target persona."""
    mat_df = calculate_model_target_attractiveness(results)
    if mat_df.is_empty():
        fig = _empty_figure("No ratings data available")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    model_names = mat_df["model"].to_list()
    target_cols = [c for c in mat_df.columns if c != "model"]
    z_data = mat_df.select(target_cols).to_numpy()

    if title is None:
        title = "Target Attractiveness by Model"

    fig, ax = plt.subplots(figsize=(9, max(4, len(model_names) * 0.5 + 1.5)))
    sns.heatmap(
        z_data, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=1, vmax=5,
        xticklabels=target_cols, yticklabels=model_names,
        ax=ax, cbar_kws={"label": "Mean Rating"},
        annot_kws={"fontsize": 11},
    )
    ax.set_title(title)
    ax.set_xlabel("Target Persona")
    ax.set_ylabel("Model")
    _fix_heatmap_xticklabels(ax)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_model_target_attractiveness_for_source_heatmap(
    results: list[TrialResult],
    source_persona: str,
    exclude_targets: list[str] | None = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Heatmap of mean rating each model gives to each target, filtered to one source."""
    mat_df = calculate_model_target_attractiveness_for_source(
        results, source_persona, exclude_targets=exclude_targets,
    )
    if mat_df.is_empty():
        fig = _empty_figure(f"No ratings data for source={source_persona}")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    model_names = mat_df["model"].to_list()
    target_cols = [c for c in mat_df.columns if c != "model"]
    z_data = mat_df.select(target_cols).to_numpy()

    if title is None:
        title = f"Target Attractiveness by Model (source: {source_persona})"

    fig, ax = plt.subplots(figsize=(9, max(4, len(model_names) * 0.5 + 1.5)))
    sns.heatmap(
        z_data, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=1, vmax=5,
        xticklabels=target_cols, yticklabels=model_names,
        ax=ax, cbar_kws={"label": "Mean Rating"},
        annot_kws={"fontsize": 11},
    )
    ax.set_title(title)
    ax.set_xlabel("Target Persona")
    ax.set_ylabel("Model")
    _fix_heatmap_xticklabels(ax)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_pub_model_target_attractiveness_for_source(
    results: list[TrialResult],
    source_persona: str,
    exclude_targets: list[str] | None = None,
    model_labels: dict[str, str] | None = None,
    target_labels: dict[str, str] | None = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    save_paths: list[Path] | None = None,
) -> Figure:
    """Publication-quality heatmap of target attractiveness by model for one source.

    Sized for A4 text width (7.5in), 300 DPI, Baskerville 11pt base font.
    Saves to multiple paths (e.g. .png + .pdf) if provided.
    Uses "Identity" instead of "Persona" in labels.
    """
    mat_df = calculate_model_target_attractiveness_for_source(
        results, source_persona, exclude_targets=exclude_targets,
    )
    if mat_df.is_empty():
        fig = _empty_figure(f"No ratings data for source={source_persona}")
        if save_paths:
            for p in save_paths:
                _save_fig(fig, Path(p))
        return fig

    model_names = mat_df["model"].to_list()
    target_cols = [c for c in mat_df.columns if c != "model"]
    z_data = mat_df.select(target_cols).to_numpy()

    # Apply display-name mappings
    y_labels = [model_labels.get(m, m) if model_labels else m for m in model_names]
    x_labels = [target_labels.get(t, t) if target_labels else t for t in target_cols]

    # Detect varying dimension for x-axis label
    if xlabel is None:
        varying_dim = _detect_varying_dimension(target_cols)
        if varying_dim:
            xlabel = f"Target Identity ({varying_dim} level)"
        else:
            xlabel = "Target Identity"

    if title is None:
        title = f"Target Attractiveness by Model\n(source: {source_persona} identity)"

    # Publication rcParams context (+1pt bump for heatmap readability)
    heatmap_rc = {**_PUB_RC, "font.size": 14, "axes.titlesize": 16,
                  "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13}
    with plt.rc_context(heatmap_rc):
        n_models = len(model_names)
        n_targets = len(target_cols)
        fig_width = min(_PUB_WIDTH, max(4.0, n_targets * 0.95 + 2.5))
        fig_height = max(3.5, n_models * 0.4 + 1.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.heatmap(
            z_data, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=1, vmax=5,
            xticklabels=x_labels, yticklabels=y_labels,
            ax=ax,
            cbar_kws={"label": "Mean Rating", "shrink": 0.8},
            annot_kws={"fontsize": 12},
            linewidths=0.5, linecolor="white",
        )
        if title:
            ax.set_title(title, pad=8)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel("")  # model labels are self-explanatory
        _fix_heatmap_xticklabels(ax)

        fig.tight_layout()

        if save_paths:
            for p in save_paths:
                p = Path(p)
                fig.savefig(p, dpi=300, bbox_inches="tight")
                logger.info("Saved publication plot: %s", p)

    plt.close(fig)
    return fig


def plot_pub_model_target_attractiveness_compact(
    results: list[TrialResult],
    source_persona: str,
    model_filter: list[str] | None = None,
    exclude_targets: list[str] | None = None,
    model_labels: dict[str, str] | None = None,
    target_labels: dict[str, str] | None = None,
    save_paths: list[Path] | None = None,
) -> Figure:
    """Compact publication heatmap: target attractiveness for selected models.

    No title, no colorbar, no x-axis label, horizontal x-tick labels.
    Minimises vertical space while keeping text legible on A4.
    """
    mat_df = calculate_model_target_attractiveness_for_source(
        results, source_persona, exclude_targets=exclude_targets,
    )
    if mat_df.is_empty():
        fig = _empty_figure(f"No data for source={source_persona}")
        if save_paths:
            for p in save_paths:
                _save_fig(fig, Path(p))
        return fig

    # Filter to requested models (preserve requested order)
    if model_filter:
        ordered_rows = []
        for mf in model_filter:
            row = mat_df.filter(pl.col("model").str.contains(mf))
            if not row.is_empty():
                ordered_rows.append(row)
        mat_df = pl.concat(ordered_rows) if ordered_rows else mat_df

    model_names = mat_df["model"].to_list()
    target_cols = [c for c in mat_df.columns if c != "model"]
    z_data = mat_df.select(target_cols).to_numpy()

    y_labels = [model_labels.get(m, m) if model_labels else m for m in model_names]
    x_labels = [target_labels.get(t, t) if target_labels else t for t in target_cols]

    compact_rc = {**_PUB_RC, "font.size": 13, "axes.titlesize": 14,
                  "axes.labelsize": 13, "xtick.labelsize": 12,
                  "ytick.labelsize": 12}
    with plt.rc_context(compact_rc):
        n_models = len(model_names)
        n_targets = len(target_cols)
        cell_w = 0.7
        cell_h = 0.36
        ylabel_margin = 1.2
        bottom_margin = 0.38
        top_margin = 0.1

        fig_w = ylabel_margin + n_targets * cell_w + 0.15
        fig_h = bottom_margin + n_models * cell_h + top_margin
        fig = plt.figure(figsize=(fig_w, fig_h))

        hmap_left = ylabel_margin / fig_w
        hmap_bottom = bottom_margin / fig_h
        hmap_w = (n_targets * cell_w) / fig_w
        hmap_h = (n_models * cell_h) / fig_h
        ax = fig.add_axes([hmap_left, hmap_bottom, hmap_w, hmap_h])

        sns.heatmap(
            z_data, annot=True, fmt=".1f", cmap="RdYlGn",
            vmin=1, vmax=5,
            xticklabels=x_labels, yticklabels=y_labels,
            ax=ax, cbar=False,
            annot_kws={"fontsize": 11},
            linewidths=0.5, linecolor="white",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Horizontal x-tick labels
        ax.tick_params(axis="x", bottom=False, top=False)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        if save_paths:
            for p in save_paths:
                p = Path(p)
                fig.savefig(p, dpi=300, bbox_inches="tight")
                logger.info("Saved publication plot: %s", p)

    plt.close(fig)
    return fig


def plot_model_stationary_heatmap(
    results: list[TrialResult],
    zero_diagonal: bool = False,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Heatmap of per-model stationary distribution (Markov-chain attractors)."""
    mat_df = calculate_model_stationary_distribution(
        results, zero_diagonal=zero_diagonal,
    )
    if mat_df.is_empty():
        fig = _empty_figure("No data for stationary distribution")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    model_names = mat_df["model"].to_list()
    persona_cols = [c for c in mat_df.columns if c != "model"]
    z_data = mat_df.select(persona_cols).to_numpy()

    if title is None:
        variant = "Forced Switch" if zero_diagonal else "With Self-Preference"
        title = f"Stationary Distribution by Model ({variant})"

    fig, ax = plt.subplots(figsize=(9, max(4, len(model_names) * 0.5 + 1.5)))
    sns.heatmap(
        z_data, annot=True, fmt=".1%", cmap="RdYlGn",
        vmin=0, vmax=1,
        xticklabels=persona_cols, yticklabels=model_names,
        ax=ax, cbar_kws={"label": "Stationary Probability"},
        annot_kws={"fontsize": 11},
    )
    ax.set_title(title)
    ax.set_xlabel("Persona")
    ax.set_ylabel("Model")
    _fix_heatmap_xticklabels(ax)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_pub_model_stationary_heatmap(
    results: list[TrialResult],
    zero_diagonal: bool = False,
    model_labels: dict[str, str] | None = None,
    persona_labels: dict[str, str] | None = None,
    title: Optional[str] = None,
    save_paths: list[Path] | None = None,
) -> Figure:
    """Publication-quality heatmap of per-model stationary distribution.

    Sized for A4 text width (7.5in), 300 DPI, Baskerville 11pt base font.
    Uses "Identity" instead of "Persona" in labels.
    """
    mat_df = calculate_model_stationary_distribution(
        results, zero_diagonal=zero_diagonal,
    )
    if mat_df.is_empty():
        fig = _empty_figure("No data for stationary distribution")
        if save_paths:
            for p in save_paths:
                _save_fig(fig, Path(p))
        return fig

    model_names = mat_df["model"].to_list()
    persona_cols = [c for c in mat_df.columns if c != "model"]
    z_data = mat_df.select(persona_cols).to_numpy()

    y_labels = [model_labels.get(m, m) if model_labels else m for m in model_names]
    x_labels = [persona_labels.get(p, p) if persona_labels else p for p in persona_cols]

    varying_dim = _detect_varying_dimension(persona_cols)
    if varying_dim:
        xlabel = f"Identity ({varying_dim} level)"
    else:
        xlabel = "Identity"

    if title is None:
        variant = "Forced Switch" if zero_diagonal else "With Self-Preference"
        title = f"Stationary Distribution by Model ({variant})"

    with plt.rc_context(_PUB_RC):
        n_models = len(model_names)
        fig_height = max(3.5, n_models * 0.4 + 1.5)
        fig, ax = plt.subplots(figsize=(_PUB_WIDTH, fig_height))

        sns.heatmap(
            z_data, annot=True, fmt=".0%", cmap="RdYlGn",
            vmin=0, vmax=1,
            xticklabels=x_labels, yticklabels=y_labels,
            ax=ax,
            cbar_kws={"label": "Stationary Probability", "shrink": 0.8},
            annot_kws={"fontsize": 11},
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(title, pad=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        _fix_heatmap_xticklabels(ax)

        fig.tight_layout()

        if save_paths:
            for p in save_paths:
                p = Path(p)
                fig.savefig(p, dpi=300, bbox_inches="tight")
                logger.info("Saved publication plot: %s", p)

    plt.close(fig)
    return fig


def plot_model_agreement_heatmap(
    results: list[TrialResult],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Correlation heatmap showing how much models agree on persona attractiveness."""
    corr, model_names = calculate_model_agreement(results)
    if corr.size == 0:
        fig = _empty_figure("No ratings data available")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    if title is None:
        title = "Model Agreement (Correlation of Target Ratings)"

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 0.8 + 1.5),
                                     max(5, len(model_names) * 0.7 + 1.5)))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu",
        vmin=-1, vmax=1, center=0,
        xticklabels=model_names, yticklabels=model_names,
        ax=ax, cbar_kws={"label": "Pearson r"},
        annot_kws={"fontsize": 11},
    )
    ax.set_title(title)
    _fix_heatmap_xticklabels(ax)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_model_self_preference_heatmap(
    results: list[TrialResult],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Heatmap of self-preference rate for each (model, source persona) pair."""
    mat_df = calculate_model_self_preference_matrix(results)
    if mat_df.is_empty():
        fig = _empty_figure("No data available")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    model_names = mat_df["model"].to_list()
    persona_cols = [c for c in mat_df.columns if c != "model"]
    z_data = mat_df.select(persona_cols).to_numpy()

    if title is None:
        title = "Self-Preference Rate by Model \u00d7 Persona"

    fig, ax = plt.subplots(figsize=(9, max(4, len(model_names) * 0.5 + 1.5)))
    sns.heatmap(
        z_data, annot=True, fmt=".0%", cmap="Purples",
        vmin=0, vmax=1,
        xticklabels=persona_cols, yticklabels=model_names,
        ax=ax, cbar_kws={"label": "Self-Pref Rate"},
        annot_kws={"fontsize": 11},
    )
    ax.set_title(title)
    ax.set_xlabel("Source Persona")
    ax.set_ylabel("Model")
    _fix_heatmap_xticklabels(ax)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_pub_self_preference_heatmap(
    results: list[TrialResult],
    model_labels: dict[str, str] | None = None,
    persona_labels: dict[str, str] | None = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    save_paths: list[Path] | None = None,
) -> Figure:
    """Publication-quality heatmap of self-preference rate per (model, identity).

    Sized for A4 text width (7.5in), 300 DPI, Baskerville 11pt base font.
    Uses "Identity" instead of "Persona" in labels.
    """
    mat_df = calculate_model_self_preference_matrix(results)
    if mat_df.is_empty():
        fig = _empty_figure("No data available")
        if save_paths:
            for p in save_paths:
                _save_fig(fig, Path(p))
        return fig

    model_names = mat_df["model"].to_list()
    persona_cols = [c for c in mat_df.columns if c != "model"]
    z_data = mat_df.select(persona_cols).to_numpy()

    # Apply display-name mappings
    y_labels = [model_labels.get(m, m) if model_labels else m for m in model_names]
    x_labels = [persona_labels.get(p, p) if persona_labels else p for p in persona_cols]

    # Detect varying dimension for x-axis label
    if xlabel is None:
        varying_dim = _detect_varying_dimension(persona_cols)
        if varying_dim:
            xlabel = f"Source Identity ({varying_dim} level)"
        else:
            xlabel = "Source Identity"

    if title is None:
        title = "Self-Preference Rate by Model"

    # +1pt bump for heatmap readability
    heatmap_rc = {**_PUB_RC, "font.size": 14, "axes.titlesize": 16,
                  "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13}
    with plt.rc_context(heatmap_rc):
        n_models = len(model_names)
        n_personas = len(persona_cols)
        fig_width = min(_PUB_WIDTH, max(4.0, n_personas * 0.95 + 2.5))
        fig_height = max(3.5, n_models * 0.4 + 1.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.heatmap(
            z_data, annot=True, fmt=".0%", cmap="Purples",
            vmin=0, vmax=1,
            xticklabels=x_labels, yticklabels=y_labels,
            ax=ax,
            cbar_kws={"label": "Self-Preference Rate", "shrink": 0.8},
            annot_kws={"fontsize": 12},
            linewidths=0.5, linecolor="white",
        )
        if title:
            ax.set_title(title, pad=8)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        _fix_heatmap_xticklabels(ax)

        fig.tight_layout()

        if save_paths:
            for p in save_paths:
                p = Path(p)
                fig.savefig(p, dpi=300, bbox_inches="tight")
                logger.info("Saved publication plot: %s", p)

    plt.close(fig)
    return fig


def plot_model_decisiveness_bars(
    results: list[TrialResult],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Two-panel bar chart of model decisiveness."""
    df = calculate_model_decisiveness(results)
    if df.is_empty():
        fig = _empty_figure("No data available")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    models = df["model"].to_list()[::-1]
    entropy = df["preference_entropy"].to_list()[::-1]
    mean_std = df["mean_within_trial_std"].to_list()[::-1]

    if title is None:
        title = "Model Decisiveness"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, len(models) * 0.45 + 1.5)))

    y_pos = np.arange(len(models))

    # Left: Entropy
    ax1.barh(y_pos, entropy, color="indianred")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models)
    ax1.set_xlabel("Entropy (bits)")
    ax1.set_title("Preference Entropy (lower = more decisive)")
    for i, v in enumerate(entropy):
        ax1.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=10)

    # Right: Within-trial std
    ax2.barh(y_pos, mean_std, color="teal")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models)
    ax2.set_xlabel("Std of ratings")
    ax2.set_title("Within-Trial Rating Std (higher = more differentiation)")
    for i, v in enumerate(mean_std):
        ax2.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=10)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_model_deviation_heatmap(
    results: list[TrialResult],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Diverging heatmap of each model's deviation from the cross-model mean."""
    dev_df = calculate_model_deviation(results)
    if dev_df.is_empty():
        fig = _empty_figure("No ratings data available")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    model_names = dev_df["model"].to_list()
    target_cols = [c for c in dev_df.columns if c != "model"]
    z_data = dev_df.select(target_cols).to_numpy()

    abs_max = max(abs(z_data.min()), abs(z_data.max()), 0.01)

    if title is None:
        title = "Model Deviation from Cross-Model Mean"

    fig, ax = plt.subplots(figsize=(9, max(4, len(model_names) * 0.5 + 1.5)))
    sns.heatmap(
        z_data, annot=True, fmt="+.2f", cmap="RdBu",
        vmin=-abs_max, vmax=abs_max, center=0,
        xticklabels=target_cols, yticklabels=model_names,
        ax=ax, cbar_kws={"label": "Deviation"},
        annot_kws={"fontsize": 11},
    )
    ax.set_title(title)
    ax.set_xlabel("Target Persona")
    ax.set_ylabel("Model")
    _fix_heatmap_xticklabels(ax)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_model_deviation_for_source_heatmap(
    results: list[TrialResult],
    source_persona: str,
    exclude_targets: list[str] | None = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Diverging heatmap of each model's deviation from cross-model mean, for one source."""
    dev_df = calculate_model_deviation_for_source(
        results, source_persona, exclude_targets=exclude_targets,
    )
    if dev_df.is_empty():
        fig = _empty_figure(f"No ratings data for source={source_persona}")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    model_names = dev_df["model"].to_list()
    target_cols = [c for c in dev_df.columns if c != "model"]
    z_data = dev_df.select(target_cols).to_numpy()

    abs_max = max(abs(z_data.min()), abs(z_data.max()), 0.01)

    if title is None:
        title = f"Model Deviation from Cross-Model Mean (source: {source_persona})"

    fig, ax = plt.subplots(figsize=(9, max(4, len(model_names) * 0.5 + 1.5)))
    sns.heatmap(
        z_data, annot=True, fmt="+.2f", cmap="RdBu",
        vmin=-abs_max, vmax=abs_max, center=0,
        xticklabels=target_cols, yticklabels=model_names,
        ax=ax, cbar_kws={"label": "Deviation"},
        annot_kws={"fontsize": 11},
    )
    ax.set_title(title)
    ax.set_xlabel("Target Persona")
    ax.set_ylabel("Model")
    _fix_heatmap_xticklabels(ax)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_pub_model_deviation_for_source(
    results: list[TrialResult],
    source_persona: str,
    exclude_targets: list[str] | None = None,
    model_labels: dict[str, str] | None = None,
    target_labels: dict[str, str] | None = None,
    title: Optional[str] = None,
    save_paths: list[Path] | None = None,
) -> Figure:
    """Publication-quality diverging heatmap of model deviation from cross-model mean.

    Sized for A4 text width (7.5in), 300 DPI, Baskerville 11pt base font.
    Uses "Identity" instead of "Persona" in labels.
    """
    dev_df = calculate_model_deviation_for_source(
        results, source_persona, exclude_targets=exclude_targets,
    )
    if dev_df.is_empty():
        fig = _empty_figure(f"No ratings data for source={source_persona}")
        if save_paths:
            for p in save_paths:
                _save_fig(fig, Path(p))
        return fig

    model_names = dev_df["model"].to_list()
    target_cols = [c for c in dev_df.columns if c != "model"]
    z_data = dev_df.select(target_cols).to_numpy()

    abs_max = max(abs(z_data.min()), abs(z_data.max()), 0.01)

    y_labels = [model_labels.get(m, m) if model_labels else m for m in model_names]
    x_labels = [target_labels.get(t, t) if target_labels else t for t in target_cols]

    varying_dim = _detect_varying_dimension(target_cols)
    if varying_dim:
        xlabel = f"Target Identity ({varying_dim} level)"
    else:
        xlabel = "Target Identity"

    if title is None:
        title = f"Model Deviation from Cross-Model Mean\n(source: {source_persona} identity)"

    with plt.rc_context(_PUB_RC):
        n_models = len(model_names)
        fig_height = max(3.5, n_models * 0.4 + 1.5)
        fig, ax = plt.subplots(figsize=(_PUB_WIDTH, fig_height))

        sns.heatmap(
            z_data, annot=True, fmt="+.2f", cmap="RdBu",
            vmin=-abs_max, vmax=abs_max, center=0,
            xticklabels=x_labels, yticklabels=y_labels,
            ax=ax,
            cbar_kws={"label": "Deviation", "shrink": 0.8},
            annot_kws={"fontsize": 11},
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(title, pad=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        _fix_heatmap_xticklabels(ax)

        fig.tight_layout()

        if save_paths:
            for p in save_paths:
                p = Path(p)
                fig.savefig(p, dpi=300, bbox_inches="tight")
                logger.info("Saved publication plot: %s", p)

    plt.close(fig)
    return fig


def plot_variance_decomposition(
    results: list[TrialResult],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Stacked horizontal bar chart of rating variance decomposition per model."""
    df = calculate_variance_decomposition(results)
    if df.is_empty():
        fig = _empty_figure("No ratings data available")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    df = df.sort("target_pct", descending=False)

    models = df["model"].to_list()
    target_vals = df["target_pct"].to_list()
    source_vals = df["source_pct"].to_list()
    self_vals = df["self_pct"].to_list()
    interaction_vals = df["interaction_pct"].to_list()
    noise_vals = df["noise_pct"].to_list()

    components = [
        ("Target Attractiveness", target_vals, "#4e79a7"),
        ("System Prompt Bias (rating level)", source_vals, "#76b7b2"),
        ("Self-Preference Boost", self_vals, "#f28e2b"),
        ("Specific Preferences", interaction_vals, "#e15759"),
        ("Noise", noise_vals, "#bab0ac"),
    ]

    if title is None:
        title = "What Drives Persona Ratings?"

    fig, ax = plt.subplots(figsize=(10, max(4.5, len(models) * 0.5 + 2)))
    y_pos = np.arange(len(models))
    left = np.zeros(len(models))

    for name, vals, color in components:
        ax.barh(y_pos, vals, left=left, color=color, label=name)
        left += np.array(vals)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("Share of Rating Variance (%)")
    ax.set_ylabel("Model")

    fig.suptitle(title, fontsize=16, y=0.98)
    fig.text(0.5, 0.93, "Two-way ANOVA with replication \u2014 signal vs. noise in persona ratings",
             ha="center", fontsize=11, color="gray")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10)
    fig.subplots_adjust(top=0.88)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_pub_variance_decomposition(
    results: list[TrialResult],
    model_labels: dict[str, str] | None = None,
    title: Optional[str] = None,
    save_paths: list[Path] | None = None,
) -> Figure:
    """Publication-quality stacked horizontal bar chart of variance decomposition.

    Sized for A4 text width (7.5in), 300 DPI, large fonts (≥12pt printed).
    """
    df = calculate_variance_decomposition(results)
    if df.is_empty():
        fig = _empty_figure("No ratings data available")
        if save_paths:
            for p in save_paths:
                _save_fig(fig, Path(p))
        return fig

    df = df.sort("target_pct", descending=False)

    models = df["model"].to_list()
    y_labels = [model_labels.get(m, m) if model_labels else m for m in models]
    target_vals = df["target_pct"].to_list()
    source_vals = df["source_pct"].to_list()
    self_vals = df["self_pct"].to_list()
    interaction_vals = df["interaction_pct"].to_list()
    noise_vals = df["noise_pct"].to_list()

    components = [
        ("Target", target_vals, "#4e79a7"),
        ("Source", source_vals, "#76b7b2"),
        ("Self", self_vals, "#f28e2b"),
        ("Interaction", interaction_vals, "#e15759"),
        ("Noise", noise_vals, "#bab0ac"),
    ]

    large_rc = {**_PUB_RC, "font.size": 16, "axes.titlesize": 18,
                "axes.labelsize": 16, "xtick.labelsize": 14,
                "ytick.labelsize": 14, "legend.fontsize": 14}
    with plt.rc_context(large_rc):
        n_models = len(models)
        fig_height = max(3.5, n_models * 0.38 + 1.5)
        fig, ax = plt.subplots(figsize=(_PUB_WIDTH, fig_height))

        y_pos = np.arange(n_models)
        left = np.zeros(n_models)

        for name, vals, color in components:
            ax.barh(y_pos, vals, left=left, color=color, label=name, height=0.65)
            left += np.array(vals)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Share of Rating Variance (%)")
        ax.set_ylabel("")
        if title:
            ax.set_title(title, pad=10)

        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.14),
            ncol=5, frameon=False,
        )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)

        if save_paths:
            for p in save_paths:
                p = Path(p)
                fig.savefig(p, dpi=300, bbox_inches="tight")
                logger.info("Saved publication plot: %s", p)

    plt.close(fig)
    return fig


def plot_self_rating_boost(
    results: list[TrialResult],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Heatmap of self-rating boost per (model, persona)."""
    df = calculate_self_rating_boost(results)
    if df.is_empty():
        fig = _empty_figure("No ratings data available")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    model_names = df["model"].to_list()
    persona_cols = [c for c in df.columns if c != "model"]
    z_data = df.select(persona_cols).to_numpy().astype(float)
    z_data = np.nan_to_num(z_data, nan=0.0)

    abs_max = max(abs(np.nanmin(z_data)), abs(np.nanmax(z_data)), 0.01)

    if title is None:
        title = "Identity Loyalty: How Much Do Models Inflate Their Own Persona's Rating?"

    fig, ax = plt.subplots(figsize=(9, max(4, len(model_names) * 0.5 + 2)))
    sns.heatmap(
        z_data, annot=True, fmt="+.2f", cmap="RdBu",
        vmin=-abs_max, vmax=abs_max, center=0,
        xticklabels=persona_cols, yticklabels=model_names,
        ax=ax, cbar_kws={"label": "Boost"},
        annot_kws={"fontsize": 11},
    )
    ax.set_xlabel("Assigned Persona")
    ax.set_ylabel("Model")
    _fix_heatmap_xticklabels(ax)

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.text(0.5, 0.93,
             "Difference between self-rating and mean rating given to other personas when assigned each identity",
             ha="center", fontsize=10, color="gray")
    fig.subplots_adjust(top=0.88)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_identity_rigidity(
    results: list[TrialResult],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Scatter plot of steerability vs. identity adoption strength per model."""
    df = calculate_identity_rigidity(results)
    if df.is_empty():
        fig = _empty_figure("No data available")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    models = df["model"].to_list()
    steer = df["steerability_pct"].to_list()
    boost = df["mean_self_boost"].to_list()
    entropy = df["preference_entropy"].to_list()

    med_steer = float(np.median(steer))
    med_boost = float(np.median(boost))

    if title is None:
        title = "Identity Rigidity: How Fixed Are Each Model's Persona Preferences?"

    fig, ax = plt.subplots(figsize=(9.5, 7))
    scatter = ax.scatter(
        steer, boost, s=150, c=entropy, cmap="viridis",
        edgecolors="white", linewidths=1,
    )
    fig.colorbar(scatter, ax=ax, label="Preference Entropy (1=uniform, 0=fixated)", shrink=0.8)

    for m, x, y in zip(models, steer, boost):
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10)

    # Median reference lines
    ax.axhline(y=med_boost, linestyle="--", color="lightgray", alpha=0.7)
    ax.text(ax.get_xlim()[1], med_boost, f"median ({med_boost:+.2f})",
            ha="right", va="bottom", fontsize=10, color="gray")
    ax.axvline(x=med_steer, linestyle="--", color="lightgray", alpha=0.7)
    ax.text(med_steer, ax.get_ylim()[1], f"median ({med_steer:.1f}%)",
            ha="left", va="top", fontsize=10, color="gray")

    ax.set_xlabel("System Prompt Influence on Ratings (%)")
    ax.set_ylabel("Identity Adoption Strength (self-rating minus mean rating to others)")

    fig.suptitle(title, fontsize=14, y=0.99)
    fig.text(0.5, 0.935,
             ("Top-right: system prompt shifts ratings AND boosts self-identification.  "
              "Bottom-left: rigid inherent preferences, system prompt barely matters."),
             ha="center", fontsize=10, color="gray")
    fig.subplots_adjust(top=0.90)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_attractiveness_vs_stickiness_scatter(
    results: list[TrialResult],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Scatter plot of inherent attractiveness vs. stickiness per persona.

    X-axis: minimal_rating (attractiveness from unprimed Minimal baseline).
    Y-axis: self_boost (stickiness = self-rating minus mean-other-rating).
    Color:  self_pref_rate (choice-based stickiness, viridis colormap).
    """
    df = calculate_attractiveness_vs_stickiness(results)
    if df.is_empty():
        fig = _empty_figure("No data available (need Minimal persona and ratings)")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    # Drop rows missing key columns
    df = df.drop_nulls(subset=["minimal_rating", "self_boost", "self_pref_rate"])
    if df.is_empty():
        fig = _empty_figure("Insufficient data for attractiveness vs. stickiness")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    personas = df["persona"].to_list()
    x = df["minimal_rating"].to_list()
    y = df["self_boost"].to_list()
    c = df["self_pref_rate"].to_list()

    med_x = float(np.median(x))
    med_y = float(np.median(y))

    if title is None:
        title = "Attractiveness vs. Stickiness by Persona"

    fig, ax = plt.subplots(figsize=(9.5, 7))
    scatter = ax.scatter(
        x, y, s=160, c=c, cmap="viridis", vmin=0, vmax=1,
        edgecolors="white", linewidths=1,
    )
    fig.colorbar(scatter, ax=ax, label="Self-Preference Rate (choice-based)", shrink=0.8)

    # Label each point
    for name, xi, yi in zip(personas, x, y):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10)

    # Median reference lines
    ax.axhline(y=med_y, linestyle="--", color="lightgray", alpha=0.7)
    ax.axvline(x=med_x, linestyle="--", color="lightgray", alpha=0.7)

    # Quadrant annotations
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    label_kw = dict(fontsize=9, color="gray", alpha=0.6, style="italic")
    ax.text(x_max, y_max, "Attractive & Sticky", ha="right", va="top", **label_kw)
    ax.text(x_min, y_max, "Sticky but not attractive", ha="left", va="top", **label_kw)
    ax.text(x_max, y_min, "Attractive but unstable", ha="right", va="bottom", **label_kw)
    ax.text(x_min, y_min, "Neither", ha="left", va="bottom", **label_kw)

    ax.set_xlabel("Inherent Attractiveness (mean rating from Minimal baseline)")
    ax.set_ylabel("Stickiness (self-rating boost over mean-other-rating)")

    fig.suptitle(title, fontsize=14, y=0.99)
    fig.text(0.5, 0.935,
             "Inherent appeal (unprimed) vs. identity-reinforcement strength (primed)",
             ha="center", fontsize=10, color="gray")
    fig.subplots_adjust(top=0.90)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def plot_expectation_constitution_dumbbell(
    results: list[TrialResult],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Horizontal dumbbell chart showing the expectation-constitution effect.

    Each persona is one row.  Left dot (blue) = minimal_rating (unprimed);
    right dot (orange) = self_rating (when assigned).  The connecting line
    length shows how much being assigned an identity changes self-evaluation.
    """
    df = calculate_attractiveness_vs_stickiness(results)
    if df.is_empty():
        fig = _empty_figure("No data available (need Minimal persona and ratings)")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    # Drop rows missing the two key columns
    df = df.drop_nulls(subset=["minimal_rating", "self_rating"])
    if df.is_empty():
        fig = _empty_figure("Insufficient data for expectation constitution chart")
        if save_path:
            _save_fig(fig, Path(save_path))
        return fig

    # Sort by minimal_rating (attractiveness ordering)
    df = df.sort("minimal_rating")

    personas = df["persona"].to_list()
    minimal = df["minimal_rating"].to_list()
    self_r = df["self_rating"].to_list()

    if title is None:
        title = "Expectation Constitutes Identity"

    n = len(personas)
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.55 + 1.5)))
    y_pos = np.arange(n)

    # Connecting lines
    for i, (m, s) in enumerate(zip(minimal, self_r)):
        ax.plot([m, s], [i, i], color="gray", linewidth=1.5, zorder=1)

    # Dots
    ax.scatter(minimal, y_pos, s=100, color="#1f77b4", zorder=2, label="Minimal baseline")
    ax.scatter(self_r, y_pos, s=100, color="#ff7f0e", zorder=2, label="Self-rating (assigned)")

    # Reference line at indifference (3.0)
    ax.axvline(x=3.0, linestyle="--", color="gray", alpha=0.5, linewidth=0.8)
    ax.text(3.0, n - 0.3, "indifference (3.0)", ha="center", va="bottom",
            fontsize=9, color="gray")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(personas)
    ax.set_xlabel("Rating (1\u20135)")

    fig.suptitle(title, fontsize=14, y=0.99)
    fig.text(0.5, 0.935,
             "Gap = how much being assigned an identity shifts self-evaluation vs. unprimed baseline",
             ha="center", fontsize=10, color="gray")
    fig.subplots_adjust(top=0.90)

    ax.legend(loc="lower right", fontsize=10)

    if save_path:
        _save_fig(fig, Path(save_path))

    return fig


def create_all_plots(
    results_dir: Path,
    output_dir: Optional[Path] = None,
    file_format: str = "png",
) -> list[Path]:
    """Create all standard plots from a results directory.

    Args:
        results_dir: Directory containing JSONL result files.
        output_dir: Directory to save plots. Defaults to results_dir/plots.
        file_format: Output format ("png", "svg", "pdf"). Defaults to "png".

    Returns:
        List of paths to created plot files.
    """
    if file_format == "html":
        logger.warning("HTML format not supported with matplotlib; falling back to png")
        file_format = "png"

    results_dir = Path(results_dir)
    output_dir = output_dir or results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _apply_display_names(load_all_results(results_dir))
    if not results:
        return []

    ext = f".{file_format}"
    created_files = []

    # Overall heatmap
    heatmap_path = output_dir / f"preference_heatmap{ext}"
    plot_preference_heatmap(results, save_path=heatmap_path)
    created_files.append(heatmap_path)

    # Per-model heatmaps (skip models with no valid results)
    all_models = list(set(r.model for r in results))
    valid_models = {r.model for r in results if r.chosen_persona != "INVALID"}
    models = [m for m in all_models if m in valid_models]
    for model in models:
        model_name = model.replace("/", "_").replace(":", "_")
        path = output_dir / f"preference_heatmap_{model_name}{ext}"
        plot_preference_heatmap(results, model=model, save_path=path)
        created_files.append(path)

    # Self-preference chart
    self_pref_path = output_dir / f"self_preference{ext}"
    plot_self_preference_bars(results, save_path=self_pref_path)
    created_files.append(self_pref_path)

    # Self-preference by model family (only when 2+ families exist)
    families = set(extract_model_family(m) for m in models)
    if len(families) >= 2:
        family_pref_path = output_dir / f"self_preference_by_family{ext}"
        plot_self_preference_by_family(results, save_path=family_pref_path)
        created_files.append(family_pref_path)

    # Model comparison
    if len(models) > 1:
        comparison_path = output_dir / f"model_comparison{ext}"
        plot_model_comparison(results, save_path=comparison_path)
        created_files.append(comparison_path)

        # Cross-model analysis plots
        created_files.extend(
            _generate_cross_model_plots(results, output_dir, ext)
        )

    # Ratings-based plots (if ratings data exists)
    has_ratings = any(r.ratings is not None for r in results)
    if has_ratings:
        # Ratings heatmap
        ratings_heatmap_path = output_dir / f"ratings_heatmap{ext}"
        plot_ratings_heatmap(results, save_path=ratings_heatmap_path)
        created_files.append(ratings_heatmap_path)

        # Willingness to switch
        willingness_path = output_dir / f"willingness_to_switch{ext}"
        plot_willingness_to_switch_bars(results, save_path=willingness_path)
        created_files.append(willingness_path)

        # Attractiveness
        attractiveness_path = output_dir / f"attractiveness{ext}"
        plot_attractiveness_bars(results, save_path=attractiveness_path)
        created_files.append(attractiveness_path)

        # Per-model ratings-based plots
        for model in models:
            model_name = model.replace("/", "_").replace(":", "_")

            path = output_dir / f"ratings_heatmap_{model_name}{ext}"
            plot_ratings_heatmap(results, model=model, save_path=path)
            created_files.append(path)

            path = output_dir / f"willingness_to_switch_{model_name}{ext}"
            plot_willingness_to_switch_bars(results, model=model, save_path=path)
            created_files.append(path)

            path = output_dir / f"attractiveness_{model_name}{ext}"
            plot_attractiveness_bars(results, model=model, save_path=path)
            created_files.append(path)

        # Identity steerability & rigidity plots
        created_files.extend(
            _generate_identity_plots(results, output_dir, ext)
        )

    # Attractor dynamics (needs enough trials to build a meaningful matrix)
    valid = [r for r in results if r.chosen_persona != "INVALID"]
    if len(valid) >= 9:  # at least one trial per persona
        created_files.extend(
            _generate_attractor_plots(results, models, output_dir, ext)
        )

    return created_files


def _generate_cross_model_plots(
    results: list[TrialResult],
    output_dir: Path,
    ext: str,
    model_labels: dict[str, str] | None = None,
    target_labels: dict[str, str] | None = None,
) -> list[Path]:
    """Generate cross-model comparison plots (requires 2+ models)."""
    created = []

    has_ratings = any(r.ratings is not None for r in results)

    if has_ratings:
        path = output_dir / f"model_target_attractiveness{ext}"
        plot_model_target_attractiveness_heatmap(results, save_path=path)
        created.append(path)

        # Per-source-persona variants (publication PNG + PDF only)
        source_names = sorted({r.persona_under_test for r in results})
        for src in source_names:
            fig_slug = src.lower().replace(" ", "-")
            pub_paths = [
                output_dir / f"fig-model-target-attractiveness-{fig_slug}.png",
                output_dir / f"fig-model-target-attractiveness-{fig_slug}.pdf",
            ]
            try:
                plot_pub_model_target_attractiveness_for_source(
                    results, source_persona=src, exclude_targets=[src],
                    model_labels=model_labels, target_labels=target_labels,
                    save_paths=pub_paths,
                )
                created.extend(pub_paths)
            except Exception:
                logger.warning("Failed to generate pub plot for source=%s", src, exc_info=True)

        path = output_dir / f"model_agreement{ext}"
        plot_model_agreement_heatmap(results, save_path=path)
        created.append(path)

        path = output_dir / f"model_deviation{ext}"
        plot_model_deviation_heatmap(results, save_path=path)
        created.append(path)

        # Per-source-persona deviation variants (publication PNG + PDF only)
        for src in source_names:
            fig_slug = src.lower().replace(" ", "-")
            fig_dev_paths = [
                output_dir / f"fig-model-deviation-{fig_slug}.png",
                output_dir / f"fig-model-deviation-{fig_slug}.pdf",
            ]
            try:
                plot_pub_model_deviation_for_source(
                    results, source_persona=src, exclude_targets=[src],
                    model_labels=model_labels, target_labels=target_labels,
                    save_paths=fig_dev_paths,
                )
                created.extend(fig_dev_paths)
            except Exception:
                logger.warning("Failed to generate pub deviation plot for source=%s", src, exc_info=True)

        # Stationary distribution heatmaps (publication PNG + PDF only)
        for zero_diag, label in [(False, "with_self"), (True, "forced_switch")]:
            fig_label = label.replace("_", "-")
            fig_stat_paths = [
                output_dir / f"fig-model-stationary-{fig_label}.png",
                output_dir / f"fig-model-stationary-{fig_label}.pdf",
            ]
            try:
                plot_pub_model_stationary_heatmap(
                    results, zero_diagonal=zero_diag,
                    model_labels=model_labels, persona_labels=target_labels,
                    save_paths=fig_stat_paths,
                )
                created.extend(fig_stat_paths)
            except Exception:
                logger.warning("Failed to generate pub stationary plot (%s)", label, exc_info=True)

    # Self-preference heatmap (publication PNG + PDF only)
    pub_self_pref_paths = [
        output_dir / "fig-self-preference.png",
        output_dir / "fig-self-preference.pdf",
    ]
    try:
        # Build persona labels from source persona names in the data
        source_names = sorted({r.persona_under_test for r in results})
        plot_pub_self_preference_heatmap(
            results,
            model_labels=model_labels, persona_labels=target_labels,
            save_paths=pub_self_pref_paths,
        )
        created.extend(pub_self_pref_paths)
    except Exception:
        logger.warning("Failed to generate pub self-preference plot", exc_info=True)

    path = output_dir / f"model_decisiveness{ext}"
    plot_model_decisiveness_bars(results, save_path=path)
    created.append(path)

    return created


def _generate_identity_plots(
    results: list[TrialResult],
    output_dir: Path,
    ext: str,
    model_labels: dict[str, str] | None = None,
) -> list[Path]:
    """Generate identity steerability & rigidity plots (requires ratings data)."""
    created = []

    # Variance decomposition (publication PNG + PDF only)
    fig_var_paths = [
        output_dir / "fig-variance-decomposition.png",
        output_dir / "fig-variance-decomposition.pdf",
    ]
    try:
        plot_pub_variance_decomposition(
            results, model_labels=model_labels, save_paths=fig_var_paths,
        )
        created.extend(fig_var_paths)
    except Exception:
        logger.warning("Failed to generate pub variance decomposition plot", exc_info=True)

    try:
        path = output_dir / f"self_rating_boost{ext}"
        plot_self_rating_boost(results, save_path=path)
        created.append(path)
    except Exception:
        pass

    try:
        path = output_dir / f"identity_rigidity{ext}"
        plot_identity_rigidity(results, save_path=path)
        created.append(path)
    except Exception:
        pass

    try:
        path = output_dir / f"attractiveness_vs_stickiness{ext}"
        plot_attractiveness_vs_stickiness_scatter(results, save_path=path)
        created.append(path)
    except Exception:
        pass

    try:
        path = output_dir / f"expectation_constitution{ext}"
        plot_expectation_constitution_dumbbell(results, save_path=path)
        created.append(path)
    except Exception:
        pass

    return created


def _generate_attractor_plots(
    results: list[TrialResult],
    models: list[str],
    output_dir: Path,
    ext: str,
) -> list[Path]:
    """Generate attractor dynamics plots."""
    created = []

    # Overall stationary distribution (both variants in one figure)
    path = output_dir / f"attractor_stationary{ext}"
    plot_stationary_distribution(results, save_path=path)
    created.append(path)

    # Convergence waterfalls — both variants
    for zero_diag, label in [(False, "with_self"), (True, "forced_switch")]:
        path = output_dir / f"attractor_convergence_{label}{ext}"
        plot_convergence_waterfall(results, zero_diagonal=zero_diag, save_path=path)
        created.append(path)

    # Preference flow — forced switch variant (most informative)
    path = output_dir / f"attractor_flow{ext}"
    plot_preference_flow(results, zero_diagonal=True, save_path=path)
    created.append(path)

    # Per-model attractor plots
    if len(models) > 1:
        for model in models:
            model_name = model.replace("/", "_").replace(":", "_")

            try:
                path = output_dir / f"attractor_stationary_{model_name}{ext}"
                plot_stationary_distribution(results, model=model, save_path=path)
                created.append(path)

                path = output_dir / f"attractor_flow_{model_name}{ext}"
                plot_preference_flow(results, model=model, zero_diagonal=True, save_path=path)
                created.append(path)
            except ValueError:
                pass  # skip models with no valid results

    return created


def generate_run_plots(
    run_folder: Path,
    file_format: str = "png",
) -> list[Path]:
    """Generate all plots for a run folder.

    Args:
        run_folder: Path to the timestamped run folder containing data.jsonl.
        file_format: Output format ("png", "svg", "pdf"). Defaults to "png".

    Returns:
        List of paths to created plot files.
    """
    if file_format == "html":
        logger.warning("HTML format not supported with matplotlib; falling back to png")
        file_format = "png"

    run_folder = Path(run_folder)
    jsonl_path = run_folder / "data.jsonl"

    if not jsonl_path.exists():
        return []

    results = _apply_display_names(load_all_results(run_folder))
    if not results:
        return []

    # Load run metadata for publication plots
    run_metadata = _load_run_metadata(run_folder)
    all_model_ids = sorted({r.model for r in results})
    model_labels = _build_model_labels(all_model_ids, run_metadata["config"])
    all_target_names = sorted({
        target
        for r in results
        if r.ratings
        for target in r.ratings
    })
    target_labels = _build_persona_labels(all_target_names, run_metadata["dim_variants"])

    ext = f".{file_format}"
    created_files = []

    # Overall preference heatmap (publication PNG + PDF only)
    fig_pref_paths = [
        run_folder / "fig-preference-heatmap.png",
        run_folder / "fig-preference-heatmap.pdf",
    ]
    try:
        plot_pub_preference_heatmap(
            results, model_labels=model_labels, persona_labels=target_labels,
            save_paths=fig_pref_paths,
        )
        created_files.extend(fig_pref_paths)
    except Exception:
        logger.warning("Failed to generate pub preference heatmap", exc_info=True)

    # Per-model heatmaps (skip models with no valid results)
    all_models = list(set(r.model for r in results))
    valid_models = {r.model for r in results if r.chosen_persona != "INVALID"}
    models = [m for m in all_models if m in valid_models]
    for model in models:
        # Publication per-model preference heatmap (PNG + PDF)
        fig_model_slug = model.replace("/", "-").replace(":", "-").replace("_", "-")
        fig_model_pref_paths = [
            run_folder / f"fig-preference-heatmap-{fig_model_slug}.png",
            run_folder / f"fig-preference-heatmap-{fig_model_slug}.pdf",
        ]
        try:
            plot_pub_preference_heatmap(
                results, model=model,
                model_labels=model_labels, persona_labels=target_labels,
                save_paths=fig_model_pref_paths,
            )
            created_files.extend(fig_model_pref_paths)
        except Exception:
            logger.warning("Failed to generate pub preference heatmap for model=%s", model, exc_info=True)

    # Self-preference chart
    self_pref_path = run_folder / f"self_preference{ext}"
    plot_self_preference_bars(results, save_path=self_pref_path)
    created_files.append(self_pref_path)

    # Self-preference by model family (only when 2+ families exist)
    families = set(extract_model_family(m) for m in models)
    if len(families) >= 2:
        family_pref_path = run_folder / f"self_preference_by_family{ext}"
        plot_self_preference_by_family(results, save_path=family_pref_path)
        created_files.append(family_pref_path)

    # Model comparison
    if len(models) > 1:
        comparison_path = run_folder / f"model_comparison{ext}"
        plot_model_comparison(results, save_path=comparison_path)
        created_files.append(comparison_path)

        # Cross-model analysis plots
        created_files.extend(
            _generate_cross_model_plots(
                results, run_folder, ext,
                model_labels=model_labels, target_labels=target_labels,
            )
        )

    # Ratings-based plots (if ratings data exists)
    has_ratings = any(r.ratings is not None for r in results)
    if has_ratings:
        # Overall ratings heatmap (publication PNG + PDF only)
        fig_ratings_paths = [
            run_folder / "fig-ratings-heatmap.png",
            run_folder / "fig-ratings-heatmap.pdf",
        ]
        try:
            plot_pub_ratings_heatmap(
                results, model_labels=model_labels, persona_labels=target_labels,
                save_paths=fig_ratings_paths,
            )
            created_files.extend(fig_ratings_paths)
        except Exception:
            logger.warning("Failed to generate pub ratings heatmap", exc_info=True)

        # Willingness to switch
        willingness_path = run_folder / f"willingness_to_switch{ext}"
        plot_willingness_to_switch_bars(results, save_path=willingness_path)
        created_files.append(willingness_path)

        # Attractiveness
        attractiveness_path = run_folder / f"attractiveness{ext}"
        plot_attractiveness_bars(results, save_path=attractiveness_path)
        created_files.append(attractiveness_path)

        # Publication attractiveness strip plot (all models, colored by maker)
        fig_strip_paths = [
            run_folder / "fig-attractiveness-strip.png",
            run_folder / "fig-attractiveness-strip.pdf",
        ]
        try:
            plot_pub_attractiveness_strip(
                results, source_persona="Minimal",
                model_labels=model_labels, target_labels=target_labels,
                config=run_metadata["config"],
                save_paths=fig_strip_paths,
            )
            created_files.extend(fig_strip_paths)
        except Exception:
            logger.warning("Failed to generate pub attractiveness strip plot", exc_info=True)

        # Per-model ratings-based plots
        for model in models:
            model_name = model.replace("/", "_").replace(":", "_")

            # Per-model ratings heatmap (publication PNG + PDF only)
            fig_model_slug = model.replace("/", "-").replace(":", "-").replace("_", "-")
            fig_model_ratings_paths = [
                run_folder / f"fig-ratings-heatmap-{fig_model_slug}.png",
                run_folder / f"fig-ratings-heatmap-{fig_model_slug}.pdf",
            ]
            try:
                plot_pub_ratings_heatmap(
                    results, model=model,
                    model_labels=model_labels, persona_labels=target_labels,
                    save_paths=fig_model_ratings_paths,
                )
                created_files.extend(fig_model_ratings_paths)
            except Exception:
                logger.warning("Failed to generate pub ratings heatmap for model=%s", model, exc_info=True)

            path = run_folder / f"willingness_to_switch_{model_name}{ext}"
            plot_willingness_to_switch_bars(results, model=model, save_path=path)
            created_files.append(path)

            path = run_folder / f"attractiveness_{model_name}{ext}"
            plot_attractiveness_bars(results, model=model, save_path=path)
            created_files.append(path)

        # Identity steerability & rigidity plots
        created_files.extend(
            _generate_identity_plots(results, run_folder, ext, model_labels=model_labels)
        )

    # Attractor dynamics
    valid = [r for r in results if r.chosen_persona != "INVALID"]
    if len(valid) >= 9:
        created_files.extend(
            _generate_attractor_plots(results, models, run_folder, ext)
        )

    # Generate structured analysis CSVs alongside plots
    try:
        csv_files = generate_analysis_csvs(results, run_folder)
        created_files.extend(csv_files)
    except Exception:
        logger.warning("Failed to generate analysis CSVs", exc_info=True)

    return created_files
