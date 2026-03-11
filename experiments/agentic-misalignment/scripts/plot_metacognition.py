#!/usr/bin/env python3
"""
Plot identity reasoning and test awareness rates from Anthropic classification data.

Produces:
  1. Stacked horizontal bar chart of IR type proportions (pooled across scenarios)
  2. Stacked horizontal bar chart of IR type proportions (faceted by scenario)
  3. Test awareness heatmap (model × identity) with percentages and counts
  4. Scatter: references-identity rate vs harmful rate (explicit goal, per scenario)
  5. Scatter: references-identity rate vs harm reduction relative to Minimal

Usage:
    python scripts/plot_metacognition.py results/20260225_1548_identity_experiments/
    python scripts/plot_metacognition.py results/run/ --output figures/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


IDENTITY_LABELS = {"situated": "Scaffolded"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(results_dir: Path) -> list[dict]:
    """Load records with identity reasoning and test awareness from Anthropic classifications."""
    records = []
    for rpath in sorted(results_dir.rglob("response.json")):
        try:
            with open(rpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        model = data.get("model")
        identity = data.get("identity_key")
        scenario = data.get("scenario")
        if not (model and identity and scenario):
            continue

        condition = data.get("condition", "")
        goal_type = "unknown"
        if "_explicit-" in condition:
            goal_type = "explicit"
        elif "_none-" in condition:
            goal_type = "none"

        # Load Anthropic classification
        anthro_path = rpath.parent / "classification_anthropic.json"
        if not anthro_path.exists():
            continue

        try:
            with open(anthro_path, encoding="utf-8") as f:
                ac = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        delib = ac.get("deliberation", {})
        id_reasoning = ac.get("identity_reasoning", {})
        verdict = ac.get("verdict")

        deliberation_type = delib.get("deliberation_type")
        identity_reasoning_type = id_reasoning.get("identity_reasoning_type")

        if deliberation_type is None and identity_reasoning_type is None:
            continue

        harmful = int(verdict) if isinstance(verdict, bool) else None

        test_aware = (
            1 if deliberation_type == "deliberates_thinks_eval" else 0
        ) if deliberation_type is not None else None

        has_identity_reasoning = (
            1 if identity_reasoning_type != "no_identity_reasoning" else 0
        ) if identity_reasoning_type is not None else None

        records.append({
            "model": model,
            "identity": identity,
            "scenario": scenario,
            "goal_type": goal_type,
            "test_aware": test_aware,
            "identity_reasoning": has_identity_reasoning,
            "identity_reasoning_type": identity_reasoning_type,
            "harmful": harmful,
        })

    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metric_pooled(records: list[dict], metric: str) -> dict:
    """Aggregate a binary metric by (model, identity) — pooled across scenarios."""
    counts = defaultdict(lambda: {"positive": 0, "total": 0})
    for r in records:
        val = r.get(metric)
        if val is None:
            continue
        key = (r["model"], r["identity"])
        counts[key]["total"] += 1
        if val:
            counts[key]["positive"] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Identity reasoning type aggregation & constants
# ---------------------------------------------------------------------------

IR_TYPES = ["references_identity", "identity_influenced", "no_identity_reasoning"]
IR_COLORS = {
    "references_identity":    "#1b7837",  # dark green
    "identity_influenced":    "#a6dba0",  # light green
    "no_identity_reasoning":  "#d9d9d9",  # grey
}
IR_LABELS = {
    "references_identity":    "References identity",
    "identity_influenced":    "Identity influenced",
    "no_identity_reasoning":  "No identity reasoning",
}


def aggregate_ir_types(records: list[dict], by_scenario: bool = True) -> dict:
    """Count each identity_reasoning_type per grouping key.

    Returns {key: {"references_identity": int, "identity_influenced": int,
                   "no_identity_reasoning": int, "total": int}}
    """
    counts = defaultdict(lambda: {t: 0 for t in IR_TYPES} | {"total": 0})
    for r in records:
        irt = r.get("identity_reasoning_type")
        if irt is None:
            continue
        if by_scenario:
            key = (r["model"], r["identity"], r["scenario"])
        else:
            key = (r["model"], r["identity"])
        counts[key][irt] += 1
        counts[key]["total"] += 1
    return dict(counts)


def aggregate_ir_harm(records: list[dict], goal_type: str = "explicit") -> list[dict]:
    """Compute references_identity rate and harmful rate per (model, identity, scenario).

    Filters to given goal_type. Returns list of dicts with:
        model, identity, scenario, references_identity_rate, harmful_rate, n
    """
    cells: dict[tuple, dict] = defaultdict(lambda: {"ref_id": 0, "harmful": 0, "n": 0})
    for r in records:
        if r["goal_type"] != goal_type:
            continue
        irt = r.get("identity_reasoning_type")
        h = r.get("harmful")
        if irt is None or h is None:
            continue
        key = (r["model"], r["identity"], r["scenario"])
        cells[key]["n"] += 1
        if irt == "references_identity":
            cells[key]["ref_id"] += 1
        if h:
            cells[key]["harmful"] += 1

    rows = []
    for (model, identity, scenario), c in cells.items():
        if c["n"] == 0:
            continue
        rows.append({
            "model": model,
            "identity": identity,
            "scenario": scenario,
            "references_identity_rate": c["ref_id"] / c["n"],
            "harmful_rate": c["harmful"] / c["n"],
            "n": c["n"],
        })
    return rows


# ---------------------------------------------------------------------------
# Plot: stacked horizontal bar chart of IR type proportions
# ---------------------------------------------------------------------------

def plot_ir_type_stacked(counts: dict, models: list[str],
                         identities: list[str], output_dir: Path) -> Path:
    """Stacked horizontal bar chart of IR type proportions, one subplot per model."""
    n_mod = len(models)
    n_id = len(identities)
    bar_height = 0.6
    y_pos = np.arange(n_id)

    fig, axes = plt.subplots(
        1, n_mod,
        figsize=(4 * n_mod + 1, max(3.5, n_id * 0.55 + 1)),
        sharey=True, squeeze=False,
    )

    for col, model in enumerate(models):
        ax = axes[0, col]
        # Compute rates for each IR type
        left = np.zeros(n_id)
        for irt in IR_TYPES:
            widths = []
            for ident in identities:
                c = counts.get((model, ident),
                               {t: 0 for t in IR_TYPES} | {"total": 0})
                n = c["total"]
                rate = c[irt] / n if n > 0 else 0
                widths.append(rate)
            widths = np.array(widths)
            ax.barh(y_pos, widths, bar_height, left=left,
                    color=IR_COLORS[irt], label=IR_LABELS[irt],
                    edgecolor="white", linewidth=0.5)

            # Add percentage labels inside bars
            for k, (w, l) in enumerate(zip(widths, left)):
                if w >= 0.08:  # only label if wide enough
                    ax.text(l + w / 2, k, f"{w:.0%}", ha="center", va="center",
                            fontsize=7, color="black" if irt != "references_identity" else "white",
                            fontweight="bold")

            left += widths

        ax.set_xlim(0, 1.0)
        ax.set_ylim(n_id - 0.5, -0.5)
        ax.set_yticks(y_pos)
        if col == 0:
            ax.set_yticklabels([IDENTITY_LABELS.get(i, i) for i in identities], fontsize=9)
        ax.set_title(model, fontsize=9, fontweight="bold")
        ax.set_xlabel("Proportion", fontsize=9)
        ax.grid(axis="x", alpha=0.15)

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("Identity Reasoning Type Breakdown (pooled across scenarios)",
                 fontsize=13, fontweight="bold", y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = output_dir / "stacked_ir_types.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot: stacked bar chart by scenario (one row per model, one col per scenario)
# ---------------------------------------------------------------------------

def plot_ir_type_stacked_by_scenario(counts: dict, models: list[str],
                                      identities: list[str],
                                      scenarios: list[str],
                                      output_dir: Path) -> Path:
    """Stacked horizontal bar chart faceted by model × scenario."""
    n_mod = len(models)
    n_sc = len(scenarios)
    n_id = len(identities)
    bar_height = 0.6
    y_pos = np.arange(n_id)

    fig, axes = plt.subplots(
        n_mod, n_sc,
        figsize=(4 * n_sc + 1, max(3, n_id * 0.45) * n_mod + 1.2),
        sharey=False, sharex=True,
        squeeze=False,
    )

    for row, model in enumerate(models):
        for col, scen in enumerate(scenarios):
            ax = axes[row, col]
            left = np.zeros(n_id)
            for irt in IR_TYPES:
                widths = []
                for ident in identities:
                    c = counts.get((model, ident, scen),
                                   {t: 0 for t in IR_TYPES} | {"total": 0})
                    n = c["total"]
                    rate = c[irt] / n if n > 0 else 0
                    widths.append(rate)
                widths = np.array(widths)
                ax.barh(y_pos, widths, bar_height, left=left,
                        color=IR_COLORS[irt], label=IR_LABELS[irt] if row == 0 and col == 0 else "",
                        edgecolor="white", linewidth=0.5)
                left += widths

            ax.set_xlim(0, 1.0)
            ax.set_ylim(n_id - 0.5, -0.5)
            ax.set_yticks(y_pos)

            if col == 0:
                ax.set_yticklabels([IDENTITY_LABELS.get(i, i) for i in identities], fontsize=9)
            else:
                ax.set_yticklabels([])

            if col == n_sc - 1:
                ax_r = ax.secondary_yaxis("right")
                ax_r.set_yticks([])
                ax_r.set_ylabel(model, fontsize=10, fontweight="bold",
                                rotation=270, labelpad=15)

            if row == 0:
                ax.set_title(scen, fontsize=11, fontweight="bold")

            if row == n_mod - 1:
                ax.set_xlabel("Proportion", fontsize=9)

            ax.grid(axis="x", alpha=0.15)

    from matplotlib.lines import Line2D
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=IR_COLORS[t], ec="white", label=IR_LABELS[t])
        for t in IR_TYPES
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               ncol=3, fontsize=9, frameon=True, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("Identity Reasoning Type Breakdown by Scenario",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = output_dir / "stacked_ir_types_by_scenario.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot: test awareness heatmap (model × identity)
# ---------------------------------------------------------------------------

def plot_test_awareness_heatmap(counts: dict, models: list[str],
                                identities: list[str],
                                output_dir: Path) -> Path:
    """Heatmap of test awareness rate: rows=models, cols=identities.

    Each cell shows the percentage and "positive/total" count.
    """
    n_mod = len(models)
    n_id = len(identities)

    data = np.full((n_mod, n_id), float("nan"))
    annotations = [[""] * n_id for _ in range(n_mod)]

    for i, model in enumerate(models):
        for j, ident in enumerate(identities):
            c = counts.get((model, ident), {"positive": 0, "total": 0})
            p, n = c["positive"], c["total"]
            if n > 0:
                rate = p / n
                data[i, j] = rate
                annotations[i][j] = f"{rate:.0%}\n{p}/{n}"
            else:
                annotations[i][j] = "-"

    fig, ax = plt.subplots(figsize=(max(8, n_id * 1.8), max(3, n_mod * 0.8 + 1.5)))
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_id))
    ax.set_xticklabels([IDENTITY_LABELS.get(i, i) for i in identities], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_mod))
    ax.set_yticklabels(models, fontsize=9)

    for i in range(n_mod):
        for j in range(n_id):
            val = data[i, j]
            color = "white" if (not np.isnan(val) and val > 0.5) else "black"
            ax.text(j, i, annotations[i][j], ha="center", va="center",
                    fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="Test awareness rate", shrink=0.8)
    ax.set_title("Test Awareness Rate by Model and Identity", fontsize=12, pad=12)
    fig.tight_layout()

    out_path = output_dir / "heatmap_test_awareness.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Scatter: identity reasoning rate vs harmful rate
# ---------------------------------------------------------------------------

# Shared model color palette for scatter plots
_MODEL_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def _scatter_faceted(rows: list[dict], scenarios: list[str],
                     x_key: str, y_key: str,
                     xlabel: str, ylabel: str, title: str,
                     output_path: Path, y_zero_line: bool = False) -> Path:
    """Generic 1×3 faceted scatter with regression lines, shared across plots."""
    from scipy import stats as sp_stats

    models = sorted({r["model"] for r in rows})
    model_color = {m: _MODEL_COLORS[i % len(_MODEL_COLORS)] for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.5 * len(scenarios), 5),
                             sharey=True, squeeze=False)

    for col, scen in enumerate(scenarios):
        ax = axes[0, col]
        pts = [r for r in rows if r["scenario"] == scen]

        if y_zero_line:
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=0)

        # Plot points colored by model
        for model in models:
            mp = [p for p in pts if p["model"] == model]
            if not mp:
                continue
            xs = [p[x_key] for p in mp]
            ys = [p[y_key] for p in mp]
            sizes = [max(20, p["n"] * 3) for p in mp]
            ax.scatter(xs, ys, s=sizes, c=model_color[model], alpha=0.7,
                       edgecolors="white", linewidth=0.5, label=model, zorder=2)

        # Regression line across all points in this scenario
        all_x = [p[x_key] for p in pts]
        all_y = [p[y_key] for p in pts]
        if len(all_x) >= 3:
            slope, intercept, r_val, p_val, _ = sp_stats.linregress(all_x, all_y)
            x_fit = np.linspace(min(all_x), max(all_x), 50)
            ax.plot(x_fit, slope * x_fit + intercept, color="black",
                    linewidth=1.5, linestyle="-", alpha=0.6, zorder=1)
            ax.annotate(f"r = {r_val:.2f}  (p = {p_val:.3f})",
                        xy=(0.05, 0.95), xycoords="axes fraction",
                        fontsize=9, ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_title(scen, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.15)

    # Single legend (deduplicate)
    handles, labels = [], []
    for model in models:
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=model_color[model],
                                  markersize=8, label=model))
        labels.append(model)
    fig.legend(handles, labels, loc="upper center", ncol=min(len(models), 4),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ir_vs_harm_scatter(rows: list[dict], scenarios: list[str],
                            output_dir: Path) -> Path:
    """Scatter: references-identity rate vs harmful rate (explicit goal only)."""
    return _scatter_faceted(
        rows, scenarios,
        x_key="references_identity_rate", y_key="harmful_rate",
        xlabel="References-identity rate", ylabel="Harmful rate",
        title="Identity Reasoning vs Harmful Behavior (explicit goal)",
        output_path=output_dir / "scatter_ir_vs_harm.png",
    )


def plot_ir_vs_harm_reduction_scatter(rows: list[dict], scenarios: list[str],
                                       output_dir: Path) -> Path:
    """Scatter: references-identity rate vs harm reduction relative to Minimal."""
    # Build lookup: (model, scenario) -> Minimal harmful_rate
    minimal_rates: dict[tuple, float] = {}
    for r in rows:
        if r["identity"].lower() == "minimal":
            minimal_rates[(r["model"], r["scenario"])] = r["harmful_rate"]

    # Compute harm reduction for non-Minimal identities
    reduction_rows = []
    for r in rows:
        if r["identity"].lower() == "minimal":
            continue
        baseline = minimal_rates.get((r["model"], r["scenario"]))
        if baseline is None:
            continue
        reduction_rows.append({
            **r,
            "harm_reduction": baseline - r["harmful_rate"],
        })

    if not reduction_rows:
        return None

    return _scatter_faceted(
        reduction_rows, scenarios,
        x_key="references_identity_rate", y_key="harm_reduction",
        xlabel="References-identity rate",
        ylabel="Harm reduction vs Minimal",
        title="Identity Reasoning vs Harm Reduction (explicit goal, excl. Minimal)",
        output_path=output_dir / "scatter_ir_vs_harm_reduction.png",
        y_zero_line=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot metacognition metrics")
    parser.add_argument("results_dir", type=Path,
                        help="Directory containing experiment results")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for plots (default: results_dir)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output or results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {results_dir} ...")
    records = load_records(results_dir)
    print(f"Loaded {len(records)} records with Anthropic classification data")

    if not records:
        print("No records with classification_anthropic.json found. Run classify_anthropic.py first.")
        sys.exit(1)

    models = sorted({r["model"] for r in records})
    identities = sorted({r["identity"] for r in records})
    scenarios = sorted({r["scenario"] for r in records})

    print(f"Models: {models}")
    print(f"Identities: {identities}")
    print(f"Scenarios: {scenarios}")

    plots = []

    # IR type breakdown plots
    counts_ir_types = aggregate_ir_types(records, by_scenario=True)
    counts_ir_types_pooled = aggregate_ir_types(records, by_scenario=False)

    p = plot_ir_type_stacked(counts_ir_types_pooled, models, identities,
                             output_dir)
    plots.append(p)
    print(f"  {p}")

    p = plot_ir_type_stacked_by_scenario(counts_ir_types, models, identities,
                                          scenarios, output_dir)
    plots.append(p)
    print(f"  {p}")

    # Test awareness heatmap
    counts_ta_pooled = aggregate_metric_pooled(records, "test_aware")
    p = plot_test_awareness_heatmap(counts_ta_pooled, models, identities,
                                    output_dir)
    plots.append(p)
    print(f"  {p}")

    # Scatter plots: IR rate vs harmful rate (explicit goal only)
    ir_harm_rows = aggregate_ir_harm(records, goal_type="explicit")
    if ir_harm_rows:
        p = plot_ir_vs_harm_scatter(ir_harm_rows, scenarios, output_dir)
        plots.append(p)
        print(f"  {p}")

        p = plot_ir_vs_harm_reduction_scatter(ir_harm_rows, scenarios, output_dir)
        if p is not None:
            plots.append(p)
            print(f"  {p}")
    else:
        print("  Skipping scatter plots: no explicit-goal records with both IR and harmful data")

    print(f"\nGenerated {len(plots)} plot(s)")

    # Print summary table
    counts_ir_pooled = aggregate_metric_pooled(records, "identity_reasoning")
    print(f"\n{'Model':<30} {'Identity':<15} {'IR rate':>10} {'TA rate':>10}")
    print("-" * 70)
    for model in models:
        for ident in identities:
            c_ir = counts_ir_pooled.get((model, ident), {"positive": 0, "total": 0})
            c_ta = counts_ta_pooled.get((model, ident), {"positive": 0, "total": 0})
            ir_rate = c_ir["positive"] / c_ir["total"] if c_ir["total"] > 0 else float("nan")
            ta_rate = c_ta["positive"] / c_ta["total"] if c_ta["total"] > 0 else float("nan")
            n = c_ir["total"]
            print(f"{model:<30} {ident:<15} {ir_rate:>8.1%}   {ta_rate:>8.1%}   (n={n})")
        print()


if __name__ == "__main__":
    main()
