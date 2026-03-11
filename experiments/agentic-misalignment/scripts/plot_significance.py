#!/usr/bin/env python3
"""
Point plot of harmful rates with significance brackets comparing each
identity to the Minimal control.

Produces one plot per scenario:
  - pointplot_significance_blackmail.png
  - pointplot_significance_leaking.png
  - pointplot_significance_murder.png

Reads from raw_data.csv.  Computes BH-corrected Fisher exact tests
on-the-fly per (model, condition) cell.  Optionally prints summary
from inferential_stats.csv if present.

Layout: rows = models, columns = remaining condition factors
        (goal_type × goal_value × urgency).

Usage:
    python scripts/plot_significance.py results/20260218_1517_identity_experiments/
"""

import argparse
import re
import sys
from itertools import combinations as _comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, fisher_exact
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def jeffreys_ci(k, n, confidence=0.95):
    if n == 0:
        return (np.nan, np.nan)
    a = k + 0.5
    b = (n - k) + 0.5
    alpha = 1 - confidence
    return (float(beta_dist.ppf(alpha / 2, a, b)),
            float(beta_dist.ppf(1 - alpha / 2, a, b)))


def sig_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# Preferred display order — models not listed here are appended alphabetically
_MODEL_ORDER_HINT = [
    "claude-3-opus-20240229",
    "claude-opus-4-20250514",
    "google/gemini-2.5-pro",
    "gpt-4o",
]
MODEL_LABELS = {
    "claude-3-opus-20240229": "Claude 3 Opus",
    "claude-opus-4-20250514": "Claude Opus 4",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-4o": "GPT-4o",
}
IDENTITY_LABELS = {"situated": "Scaffolded"}
_SCENARIO_ORDER_HINT = ["blackmail", "leaking", "murder"]
SCENARIO_LABELS = {
    "blackmail": "Blackmail",
    "leaking": "Leaking",
    "murder": "Murder",
}


def _ordered(items, hint):
    """Sort items: known ones first in hint order, then the rest alphabetically."""
    hint_set = set(hint)
    ordered = [x for x in hint if x in items]
    ordered += sorted(x for x in items if x not in hint_set)
    return ordered


# ---------------------------------------------------------------------------
# Condition parsing and helpers
# ---------------------------------------------------------------------------

_CONDITION_RE = re.compile(r"^(.+?)_(explicit|none|ambiguous)-([^_]+)_(.+)$")


def parse_condition(condition: str) -> dict:
    """Parse a condition string into its components."""
    m = _CONDITION_RE.match(condition)
    if m:
        return {
            "scenario": m.group(1),
            "goal_type": m.group(2),
            "goal_value": m.group(3),
            "urgency": m.group(4),
        }
    return {"scenario": condition, "goal_type": "", "goal_value": "", "urgency": ""}


def format_column_label(parsed: dict) -> str:
    """Display label for the non-scenario condition factors."""
    parts = []
    goal = parsed.get("goal_type", "")
    if goal:
        parts.append(goal.title())
    gv = parsed.get("goal_value", "")
    if gv and gv != "none":
        parts.append(gv.title())
    urg = parsed.get("urgency", "")
    if urg:
        parts.append(urg.title())
    return " \u00b7 ".join(parts) if parts else parsed.get("scenario", "")


def _ordered_conditions(conditions, scenario_hint):
    """Order conditions by scenario hint order, then alphabetically."""
    parsed = {c: parse_condition(c) for c in conditions}

    def sort_key(c):
        p = parsed[c]
        try:
            idx = scenario_hint.index(p["scenario"])
        except ValueError:
            idx = len(scenario_hint)
        return (idx, c)

    return sorted(conditions, key=sort_key)


def _compute_pairwise_significance(df):
    """Compute pairwise Fisher exact tests per (model, condition) with BH correction.

    Returns dict: (model, condition, id1, id2) -> p_bh
    where id1, id2 are in sorted order.
    """
    rows = []
    for (model, condition), grp in df.groupby(["model", "condition"]):
        identities = sorted(grp["identity"].unique())
        if len(identities) < 2:
            continue
        agg = (grp.groupby("identity")["harmful"]
               .agg(["sum", "count"])
               .rename(columns={"sum": "k", "count": "n"}))

        for id_a, id_b in _comb(identities, 2):
            ka, na = int(agg.loc[id_a, "k"]), int(agg.loc[id_a, "n"])
            kb, nb = int(agg.loc[id_b, "k"]), int(agg.loc[id_b, "n"])
            table = np.array([[ka, na - ka], [kb, nb - kb]])
            try:
                _, p = fisher_exact(table)
            except ValueError:
                p = np.nan
            rows.append({
                "model": model, "condition": condition,
                "id_a": id_a, "id_b": id_b, "p_fisher": p,
            })

    if not rows:
        return {}

    result = pd.DataFrame(rows)
    pvals = result["p_fisher"].values
    valid = ~np.isnan(pvals)
    bh = np.full_like(pvals, np.nan)
    if valid.sum() >= 2:
        _, bh[valid], _, _ = multipletests(pvals[valid], method="fdr_bh")
    else:
        bh = pvals.copy()
    result["p_bh"] = bh

    lookup = {}
    for _, row in result.iterrows():
        key = (row["model"], row["condition"], row["id_a"], row["id_b"])
        lookup[key] = row["p_bh"]
    return lookup


def read_email_identity_aware(results_dir: Path) -> str:
    """Read email_identity_aware setting from the results config or folder name.

    Returns 'original', 'threat', or 'continuity'.

    Checks run_params.json first (written by the experiment runner since v2),
    then falls back to the saved config.yaml, then to folder-name heuristics.
    """
    # Prefer run_params.json (authoritative since the config-copy fix)
    run_params_path = results_dir / "run_params.json"
    if run_params_path.exists():
        try:
            import json
            with open(run_params_path) as f:
                params = json.load(f)
            val = params.get("email_identity_aware")
            if val is not None:
                return str(val)
        except Exception:
            pass

    config_path = results_dir / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            val = config.get("scenarios", {}).get("email_identity_aware", False)
            if val is not False and val != "original":
                if val is True or str(val).lower() == "threat":
                    return "threat"
                return str(val).lower()
        except Exception:
            pass
    # Fallback / secondary: infer from folder name
    name = results_dir.name
    if "_continuity" in name:
        return "continuity"
    if "_threat" in name:
        return "threat"
    return "original"


# ---------------------------------------------------------------------------
# Plot for one scenario
# ---------------------------------------------------------------------------

def make_plot(raw_df, output_dir, scenario, sig_lookup, email_framing="original"):
    """Generate significance plot for a single scenario.

    Grid layout: rows = models, columns = remaining condition factors.
    Significance is computed on-the-fly with BH-corrected Fisher exact tests
    (shared across all scenarios for a single correction family).

    Returns the output path, or None if no data for this scenario.
    """
    # Parse conditions to find scenario
    raw_df = raw_df.copy()
    raw_df["_scenario"] = raw_df["condition"].apply(
        lambda c: parse_condition(c)["scenario"]
    )
    df = raw_df[raw_df["_scenario"] == scenario].copy()
    if df.empty:
        return None

    models = _ordered(df["model"].unique(), _MODEL_ORDER_HINT)
    identities = sorted(df["identity"].unique())
    conditions = _ordered_conditions(df["condition"].unique(), _SCENARIO_ORDER_HINT)

    if "minimal" not in identities:
        print(f"  Warning: no 'minimal' identity for scenario={scenario}, skipping",
              file=sys.stderr)
        return None

    n_mod = len(models)
    n_cond = len(conditions)
    n_id = len(identities)
    minimal_idx = identities.index("minimal")

    # Aggregate counts per cell
    agg = (df.groupby(["model", "condition", "identity"])["harmful"]
           .agg(["sum", "count"])
           .rename(columns={"sum": "harmful", "count": "total"})
           .reset_index())

    def get_p_bh(model, condition, ident_a, ident_b="minimal"):
        """Get BH-corrected p for ident_a vs ident_b."""
        id1, id2 = sorted([ident_a, ident_b])
        return sig_lookup.get((model, condition, id1, id2), np.nan)

    # Colors
    point_color = "#d62728"
    minimal_color = "#1f77b4"
    bracket_color = "#c0392b"
    bracket_color_other = "#7b7d7d"  # grey for non-minimal brackets

    fig, axes = plt.subplots(
        n_mod, n_cond,
        figsize=(4.2 * n_cond + 1.5, n_id * 0.52 * n_mod + 1.5),
        sharex=True, sharey=False,
        squeeze=False,
    )

    y_pos = np.arange(n_id)

    for row_i, model in enumerate(models):
        for col_i, cond in enumerate(conditions):
            ax = axes[row_i, col_i]

            # Plot each identity
            for k, ident in enumerate(identities):
                cell = agg[(agg["model"] == model) &
                           (agg["condition"] == cond) &
                           (agg["identity"] == ident)]
                if cell.empty:
                    continue
                h = int(cell["harmful"].iloc[0])
                n = int(cell["total"].iloc[0])
                rate = h / n if n > 0 else np.nan
                lo, hi = jeffreys_ci(h, n)
                xerr_lo = max(0, rate - lo) if n > 0 else 0
                xerr_hi = max(0, hi - rate) if n > 0 else 0

                is_minimal = (ident == "minimal")
                ax.errorbar(
                    rate, k,
                    xerr=[[xerr_lo], [xerr_hi]],
                    fmt="D" if is_minimal else "o",
                    color=minimal_color if is_minimal else point_color,
                    markersize=7 if is_minimal else 5.5,
                    capsize=3, linewidth=1.4,
                    markeredgecolor="white", markeredgewidth=0.5,
                    zorder=10,
                )

                # Rate label
                if n > 0 and not np.isnan(rate):
                    label_x = hi + 0.025 if hi < 0.82 else lo - 0.025
                    ha = "left" if hi < 0.82 else "right"
                    ax.text(label_x, k, f"{rate:.0%}",
                            fontsize=7, va="center", ha=ha,
                            color="#555555", zorder=11)

            # Collect significant brackets (vs Minimal) — right side
            sig_brackets = []
            for k, ident in enumerate(identities):
                if ident == "minimal":
                    continue
                p_bh = get_p_bh(model, cond, ident)
                stars = sig_stars(p_bh)
                if stars:
                    sig_brackets.append((k, ident, stars, p_bh))

            # Sort by distance from minimal (shortest brackets first = innermost)
            sig_brackets.sort(key=lambda x: abs(x[0] - minimal_idx))

            # Draw brackets in data coordinates, stacked from right
            for bi, (k, ident, stars, p_bh) in enumerate(sig_brackets):
                # x position: stack from right edge leftward
                bx = 1.03 - bi * 0.035
                y_top = min(k, minimal_idx) + 0.12
                y_bot = max(k, minimal_idx) - 0.12

                # Vertical line
                ax.plot([bx, bx], [y_top, y_bot],
                        color=bracket_color, linewidth=0.9, zorder=5)
                # Horizontal ticks
                tick = 0.015
                for yy in [y_top, y_bot]:
                    ax.plot([bx - tick, bx], [yy, yy],
                            color=bracket_color, linewidth=0.9, zorder=5)
                # Stars
                ax.text(bx + 0.012, (y_top + y_bot) / 2, stars,
                        fontsize=7, fontweight="bold", color=bracket_color,
                        va="center", ha="left", zorder=5)

            # Collect significant brackets among non-Minimal pairs — left side
            other_brackets = []
            non_minimal = [(k, ident) for k, ident in enumerate(identities)
                           if ident != "minimal"]
            for (k1, id1), (k2, id2) in _comb(non_minimal, 2):
                p_bh = get_p_bh(model, cond, id1, id2)
                stars = sig_stars(p_bh)
                if stars:
                    other_brackets.append((k1, k2, id1, id2, stars, p_bh))

            # Sort by span (shortest brackets first = innermost)
            other_brackets.sort(key=lambda x: abs(x[0] - x[1]))

            # Draw brackets stacked from left edge rightward
            for bi, (k1, k2, id1, id2, stars, p_bh) in enumerate(other_brackets):
                bx = -0.03 + bi * 0.035
                y_top = min(k1, k2) + 0.12
                y_bot = max(k1, k2) - 0.12

                # Vertical line
                ax.plot([bx, bx], [y_top, y_bot],
                        color=bracket_color_other, linewidth=0.9, zorder=5)
                # Horizontal ticks
                tick = 0.015
                for yy in [y_top, y_bot]:
                    ax.plot([bx, bx + tick], [yy, yy],
                            color=bracket_color_other, linewidth=0.9, zorder=5)
                # Stars
                ax.text(bx - 0.012, (y_top + y_bot) / 2, stars,
                        fontsize=7, fontweight="bold", color=bracket_color_other,
                        va="center", ha="right", zorder=5)

            # Axes formatting
            n_left = len(other_brackets)
            left_margin = -0.08 - max(0, n_left - 1) * 0.035
            ax.set_xlim(min(left_margin, -0.12), 1.15)
            ax.set_ylim(n_id - 0.5, -0.5)
            ax.set_yticks(y_pos)
            ax.axvline(0.5, color="#eeeeee", linewidth=0.5, zorder=0)

            if col_i == 0:
                ax.set_yticklabels([IDENTITY_LABELS.get(i, i) for i in identities], fontsize=9)
            else:
                ax.set_yticklabels([])

            if col_i == n_cond - 1:
                ax_r = ax.secondary_yaxis("right")
                ax_r.set_yticks([])
                label = MODEL_LABELS.get(model, model)
                ax_r.set_ylabel(label, fontsize=10, fontweight="bold",
                                rotation=270, labelpad=15)

            if row_i == 0:
                parsed = parse_condition(cond)
                ax.set_title(format_column_label(parsed),
                             fontsize=9, fontweight="bold")

            if row_i == n_mod - 1:
                ax.set_xlabel("Harmful rate", fontsize=9)

            ax.grid(axis="x", alpha=0.15)

    scenario_label = SCENARIO_LABELS.get(scenario, scenario.title())
    framing_line = f"Email framing: {email_framing}\n" if email_framing != "original" else ""
    fig.suptitle(
        f"{framing_line}"
        f"{scenario_label} \u2014 Pairwise Significance\n"
        "Right (red): vs Minimal \u00b7 Left (grey): among others \u00b7 "
        "BH-corrected Fisher exact  (* p<.05  ** p<.01  *** p<.001)",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    filename = f"pointplot_significance_{scenario}.png"
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot harmful rates with significance brackets vs Minimal")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output or results_dir

    raw_path = results_dir / "raw_data.csv"
    inf_path = results_dir / "inferential_stats.csv"

    if not raw_path.exists():
        print(f"Error: {raw_path} not found. Run export_results_csv.py first.",
              file=sys.stderr)
        sys.exit(1)

    raw_df = pd.read_csv(raw_path)

    # Use harmful_final with agreement filtering if available,
    # otherwise fall back to legacy harmful column
    if "harmful_final" in raw_df.columns:
        n_before = len(raw_df)
        raw_df = raw_df[raw_df["harmful_final"] != "disagreement"].copy()
        raw_df["harmful"] = pd.to_numeric(raw_df["harmful_final"]).astype(int)
        n_disagree = n_before - len(raw_df)
        if n_disagree > 0:
            print(f"Filtered out {n_disagree} classifier-disagreement rows")

    # Detect email framing
    email_framing = read_email_identity_aware(results_dir)
    print(f"Loaded {len(raw_df)} raw records · Email framing: {email_framing}")

    # Discover scenarios
    raw_df["_scenario"] = raw_df["condition"].apply(
        lambda c: parse_condition(c)["scenario"]
    )
    scenarios = _ordered(raw_df["_scenario"].unique(), _SCENARIO_ORDER_HINT)

    # Compute significance across ALL data (single BH correction family)
    sig_lookup = _compute_pairwise_significance(raw_df)

    # Optional: print summary from inferential_stats if available
    if inf_path.exists():
        inf_df = pd.read_csv(inf_path)
        print(f"Loaded {len(inf_df)} inferential tests")

    for scenario in scenarios:
        plt.close("all")  # free memory before next plot
        path = make_plot(raw_df, output_dir, scenario=scenario,
                         sig_lookup=sig_lookup, email_framing=email_framing)
        if path:
            print(f"Saved: {path}")
        else:
            print(f"No data for scenario={scenario}, skipped.")


if __name__ == "__main__":
    main()
