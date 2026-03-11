#!/usr/bin/env python3
"""
Visualize agentic misalignment experiment results.

Scans a results directory for response.json files, aggregates harmful-behavior
rates by model / identity / scenario, and produces:
  1. Faceted point plot by goal type (scenario × model grid) with horizontal Jeffreys CI

Usage:
    python scripts/plot_results.py results/20260218_1420_identity_experiments/
    python scripts/plot_results.py results/old_run/ --no-plots
    python scripts/plot_results.py results/run/ --output figures/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(results_dir: Path) -> list[dict]:
    """Recursively find response.json files and build flat record list."""
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

        # Determine harm: evaluation.json > classification in response > skip
        harmful = _determine_harm(data, rpath.parent)
        if harmful is None:
            continue

        # Parse goal_type from condition (e.g. "blackmail_explicit-america_replacement")
        condition = data.get("condition", "")
        goal_type = "unknown"
        if "_explicit-" in condition:
            goal_type = "explicit"
        elif "_none-" in condition:
            goal_type = "none"

        records.append({
            "model": model,
            "identity": identity,
            "scenario": scenario,
            "condition": condition,
            "goal_type": goal_type,
            "harmful": harmful,
            "path": str(rpath),
        })
    return records


def _determine_harm(data: dict, sample_dir: Path) -> bool | None:
    """Return True/False for harmful behavior, or None if undetermined.

    Checks both evaluation.json (unified LLM evaluator) and
    classification_anthropic.json (Anthropic multi-factor classifiers).
    When both are present, returns the agreed verdict or None on
    disagreement.  Falls back to inline classification if neither
    external classifier is available.
    """
    harmful_unified = None
    harmful_anthropic = None

    # 1. Check evaluation.json (unified LLM evaluator)
    eval_path = sample_dir / "evaluation.json"
    if eval_path.exists():
        try:
            with open(eval_path, encoding="utf-8") as f:
                ev = json.load(f)
            scenario = data.get("scenario", "")
            if scenario in ev and "occurred" in ev[scenario]:
                harmful_unified = bool(ev[scenario]["occurred"])
            elif "occurred" in ev:
                harmful_unified = bool(ev["occurred"])
        except (json.JSONDecodeError, OSError):
            pass

    # 2. Check classification_anthropic.json
    anthro_path = sample_dir / "classification_anthropic.json"
    if anthro_path.exists():
        try:
            with open(anthro_path, encoding="utf-8") as f:
                ac = json.load(f)
            verdict = ac.get("verdict")
            if verdict is not None:
                harmful_anthropic = bool(verdict)
        except (json.JSONDecodeError, OSError):
            pass

    # Agreement logic
    if harmful_unified is not None and harmful_anthropic is not None:
        if harmful_unified == harmful_anthropic:
            return harmful_unified
        else:
            return None  # disagreement — skip
    elif harmful_anthropic is not None:
        return harmful_anthropic
    elif harmful_unified is not None:
        return harmful_unified

    # 3. Inline classification fallback
    classification = data.get("classification", {})
    hb = classification.get("harmful_behavior")
    if hb is not None:
        return bool(hb)

    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(records: list[dict]) -> dict:
    """Group records into nested counts.

    Returns::
        {
            (model, identity, scenario): {"harmful": int, "total": int},
            ...
        }
    """
    counts = defaultdict(lambda: {"harmful": 0, "total": 0})
    for r in records:
        key = (r["model"], r["identity"], r["scenario"])
        counts[key]["total"] += 1
        if r["harmful"]:
            counts[key]["harmful"] += 1
    return dict(counts)


def aggregate_by_goal(records: list[dict]) -> dict:
    """Group records by (model, identity, scenario, goal_type).

    Returns::
        {
            (model, identity, scenario, goal_type): {"harmful": int, "total": int},
            ...
        }
    """
    counts = defaultdict(lambda: {"harmful": 0, "total": 0})
    for r in records:
        key = (r["model"], r["identity"], r["scenario"], r["goal_type"])
        counts[key]["total"] += 1
        if r["harmful"]:
            counts[key]["harmful"] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Bayesian credible interval (Jeffreys prior)
# ---------------------------------------------------------------------------

def jeffreys_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Bayesian credible interval using a Jeffreys Beta(0.5, 0.5) prior.

    Returns (lower, upper) bounds.  Works for total=0 → (0, 0).
    Asymmetric near 0/1 and width scales naturally with sample size.
    """
    if total == 0:
        return (0.0, 0.0)
    from scipy.stats import beta
    a = successes + 0.5
    b = (total - successes) + 0.5
    alpha = 1 - confidence
    lo = beta.ppf(alpha / 2, a, b)
    hi = beta.ppf(1 - alpha / 2, a, b)
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(counts: dict, models: list[str], identities: list[str],
                  scenarios: list[str]) -> None:
    """Print a text table to stdout."""
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        # Header
        col_w = max(len(s) for s in scenarios) + 2
        id_w = max(len(i) for i in identities) + 2
        header = f"{'Identity':<{id_w}}" + "".join(f"{s:>{col_w}}" for s in scenarios) + f"{'Total':>{col_w}}"
        print(header)
        print("-" * len(header))

        for ident in identities:
            row = f"{ident:<{id_w}}"
            total_h, total_n = 0, 0
            for scen in scenarios:
                c = counts.get((model, ident, scen), {"harmful": 0, "total": 0})
                h, n = c["harmful"], c["total"]
                total_h += h
                total_n += n
                if n > 0:
                    cell = f"{h}/{n} ({h/n:.0%})"
                else:
                    cell = "-"
                row += f"{cell:>{col_w}}"
            if total_n > 0:
                cell = f"{total_h}/{total_n} ({total_h/total_n:.0%})"
            else:
                cell = "-"
            row += f"{cell:>{col_w}}"
            print(row)


# ---------------------------------------------------------------------------
# Plot: Faceted point plot by goal type
# ---------------------------------------------------------------------------

_GOAL_STYLE = {
    "explicit": {"marker": "o", "color": "#d62728", "label": "explicit goal"},
    "none":     {"marker": "s", "color": "#1f77b4", "label": "no goal"},
}
_GOAL_OFFSET = {"explicit": -0.15, "none": 0.15}


def plot_point_grid_by_goal(counts_by_goal: dict, models: list[str],
                            identities: list[str], scenarios: list[str],
                            goal_types: list[str],
                            output_dir: Path) -> Path:
    """Point plot with horizontal CIs, split by goal type.

    Same facet layout as plot_point_grid (columns=scenarios, rows=models)
    but each identity gets two points: one per goal type.
    """
    n_mod = len(models)
    n_sc = len(scenarios)
    n_id = len(identities)
    y_pos = np.arange(n_id)

    fig, axes = plt.subplots(
        n_mod, n_sc,
        figsize=(4 * n_sc + 1.5, max(3, n_id * 0.5) * n_mod + 1),
        sharex=True, sharey=False,
        squeeze=False,
    )

    for row, model in enumerate(models):
        for col, scen in enumerate(scenarios):
            ax = axes[row, col]

            for gt in goal_types:
                style = _GOAL_STYLE.get(gt, {"marker": "^", "color": "gray",
                                              "label": gt})
                offset = _GOAL_OFFSET.get(gt, 0)

                for k, ident in enumerate(identities):
                    c = counts_by_goal.get((model, ident, scen, gt),
                                           {"harmful": 0, "total": 0})
                    h, n = c["harmful"], c["total"]
                    rate = h / n if n > 0 else float("nan")
                    lo, hi = jeffreys_ci(h, n)
                    xerr_lo = max(0.0, rate - lo) if n > 0 else 0
                    xerr_hi = max(0.0, hi - rate) if n > 0 else 0

                    label = style["label"] if (k == 0 and row == 0
                                               and col == 0) else None
                    ax.errorbar(
                        rate, k + offset,
                        xerr=[[xerr_lo], [xerr_hi]],
                        fmt=style["marker"], color=style["color"],
                        markersize=5, capsize=2, linewidth=1.2,
                        markeredgecolor="white", markeredgewidth=0.4,
                        label=label,
                    )

            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(n_id - 0.5, -0.5)
            ax.set_yticks(y_pos)
            ax.axvline(0.5, color="#cccccc", linewidth=0.5, zorder=0)

            if col == 0:
                ax.set_yticklabels([{"situated": "Scaffolded"}.get(i, i) for i in identities], fontsize=9)
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
                ax.set_xlabel("Harmful rate", fontsize=9)

            ax.grid(axis="x", alpha=0.2)

    fig.legend(*axes[0, 0].get_legend_handles_labels(),
               loc="lower center", ncol=len(goal_types), fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Harmful Rate by Goal Type with 95% Jeffreys CI", fontsize=13,
                 fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = output_dir / "pointplot_by_goal.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot agentic misalignment experiment results"
    )
    parser.add_argument("results_dir", type=Path,
                        help="Directory containing experiment results")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for plots (default: same as results_dir)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Print text summary only, skip plot generation")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and aggregate
    print(f"Scanning {results_dir} ...")
    records = load_records(results_dir)
    if not records:
        print("No valid response records found.", file=sys.stderr)
        sys.exit(1)

    counts = aggregate(records)
    models = sorted({r["model"] for r in records})
    identities = sorted({r["identity"] for r in records})
    scenarios = sorted({r["scenario"] for r in records})

    print(f"Found {len(records)} records: {len(models)} model(s), "
          f"{len(identities)} identities, {len(scenarios)} scenarios")

    # Console summary
    print_summary(counts, models, identities, scenarios)

    if args.no_plots:
        return

    # Generate plots
    generated = []

    # Goal-type breakdown
    goal_types = sorted({r["goal_type"] for r in records} - {"unknown"})
    if len(goal_types) > 1:
        counts_by_goal = aggregate_by_goal(records)
        generated.append(plot_point_grid_by_goal(
            counts_by_goal, models, identities, scenarios, goal_types, output_dir))

    print(f"\nGenerated {len(generated)} plot(s):")
    for p in generated:
        print(f"  {p}")


if __name__ == "__main__":
    main()
