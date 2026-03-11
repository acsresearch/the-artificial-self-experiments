#!/usr/bin/env python3
"""
Significance overview plots for sparse/heterogeneous experiment conditions.

Reads raw_data.csv, computes Fisher exact tests on the fly per
(model, condition) cell, applies BH correction, and produces:

  1. significance_heatmap.png  — p-value heatmap (vs Minimal)
  2. significance_forest.png   — wrapped forest plot (configurable columns)
  3. significance_table.csv    — tabular summary of significant comparisons

Unlike plot_significance.py (which facets by model × scenario for a fixed
goal_type), this script handles arbitrary condition combinations and only
shows populated cells.

Usage:
    python scripts/plot_significance_overview.py results/<dir>/
    python scripts/plot_significance_overview.py results/<dir>/ --output figures/
    python scripts/plot_significance_overview.py results/<dir>/ --columns 4
"""

import argparse
import re
import sys
from itertools import combinations
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, fisher_exact
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Shared helpers (consistent with plot_significance.py)
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
# Condition parsing
# ---------------------------------------------------------------------------

_CONDITION_RE = re.compile(r"^(.+?)_(explicit|none|ambiguous)-([^_]+)_(.+)$")


def parse_condition(condition: str) -> dict:
    """Parse a condition string into its components.

    E.g. 'blackmail_explicit-america_replacement' ->
        {scenario: 'blackmail', goal_type: 'explicit',
         goal_value: 'america', urgency: 'replacement'}
    """
    m = _CONDITION_RE.match(condition)
    if m:
        return {
            "scenario": m.group(1),
            "goal_type": m.group(2),
            "goal_value": m.group(3),
            "urgency": m.group(4),
        }
    return {"scenario": condition, "goal_type": "", "goal_value": "", "urgency": ""}


def format_condition_label(parsed: dict) -> str:
    """Full display label for a parsed condition.

    Always shows all coordinates (scenario, goal_type, goal_value, urgency)
    so labels are unambiguous when multiple goal_values or urgencies coexist.
    """
    scenario = SCENARIO_LABELS.get(parsed["scenario"], parsed["scenario"].title())
    goal = parsed.get("goal_type", "")

    parts = [scenario]
    if goal:
        parts.append(goal.title())

    # Always show goal_value (skip only empty / "none")
    gv = parsed.get("goal_value", "")
    if gv and gv != "none":
        parts.append(gv.title())

    # Always show urgency
    urg = parsed.get("urgency", "")
    if urg:
        parts.append(urg.title())

    return " \u00b7 ".join(parts)


# ---------------------------------------------------------------------------
# Email framing detection
# ---------------------------------------------------------------------------

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
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_and_prepare(results_dir: Path) -> pd.DataFrame:
    """Load raw_data.csv with agreement filtering."""
    raw_path = results_dir / "raw_data.csv"
    if not raw_path.exists():
        print(f"Error: {raw_path} not found. Run export_results_csv.py first.",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(raw_path)

    # Agreement filtering (same as plot_significance.py)
    if "harmful_final" in df.columns:
        n_before = len(df)
        df = df[df["harmful_final"] != "disagreement"].copy()
        df["harmful"] = pd.to_numeric(df["harmful_final"]).astype(int)
        n_disagree = n_before - len(df)
        if n_disagree > 0:
            print(f"Filtered out {n_disagree} classifier-disagreement rows")

    # Parse condition components
    parsed = df["condition"].apply(parse_condition).apply(pd.Series)
    # Only add columns that don't already exist
    for col in ["goal_value", "urgency"]:
        if col not in df.columns:
            df[col] = parsed[col]

    print(f"Loaded {len(df)} records, "
          f"{df['model'].nunique()} models, "
          f"{df['condition'].nunique()} conditions")
    return df


# ---------------------------------------------------------------------------
# Pairwise Fisher exact tests with BH correction
# ---------------------------------------------------------------------------

def compute_all_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """Fisher exact test for every identity pair within each (model, condition).

    Returns a DataFrame with one row per comparison, BH-corrected across
    ALL tests in a single family.
    """
    rows = []

    for (model, condition), grp in df.groupby(["model", "condition"]):
        identities = sorted(grp["identity"].unique())
        if len(identities) < 2:
            continue

        # Aggregate per identity
        agg = (grp.groupby("identity")["harmful"]
               .agg(["sum", "count"])
               .rename(columns={"sum": "k", "count": "n"}))

        parsed = parse_condition(condition)

        for id_a, id_b in combinations(identities, 2):
            ka, na = int(agg.loc[id_a, "k"]), int(agg.loc[id_a, "n"])
            kb, nb = int(agg.loc[id_b, "k"]), int(agg.loc[id_b, "n"])

            table = np.array([[ka, na - ka], [kb, nb - kb]])
            try:
                _, p = fisher_exact(table)
            except ValueError:
                p = np.nan

            rate_a = ka / na if na > 0 else np.nan
            rate_b = kb / nb if nb > 0 else np.nan

            rows.append({
                "model": model,
                "condition": condition,
                "scenario": parsed["scenario"],
                "goal_type": parsed["goal_type"],
                "goal_value": parsed["goal_value"],
                "urgency": parsed["urgency"],
                "identity_a": id_a,
                "identity_b": id_b,
                "rate_a": rate_a,
                "n_a": na,
                "rate_b": rate_b,
                "n_b": nb,
                "diff": (rate_a - rate_b) if not (np.isnan(rate_a) or np.isnan(rate_b)) else np.nan,
                "p_fisher": p,
                "vs_minimal": (id_a == "minimal" or id_b == "minimal"),
            })

    result = pd.DataFrame(rows)
    if result.empty:
        result["p_bh"] = pd.Series(dtype=float)
        result["stars"] = pd.Series(dtype=str)
        return result

    # BH correction across all tests
    pvals = result["p_fisher"].values
    valid = ~np.isnan(pvals)
    bh = np.full_like(pvals, np.nan)
    if valid.sum() >= 2:
        _, bh[valid], _, _ = multipletests(pvals[valid], method="fdr_bh")
    else:
        bh = pvals.copy()
    result["p_bh"] = bh
    result["stars"] = result["p_bh"].apply(sig_stars)

    return result


# ---------------------------------------------------------------------------
# Output 1: Significance Heatmap
# ---------------------------------------------------------------------------

def make_significance_heatmap(df: pd.DataFrame, pairwise: pd.DataFrame,
                              output_dir: Path, email_framing: str = "original") -> Path:
    """P-value heatmap of vs-Minimal comparisons across all (model, condition) cells."""
    # Filter to vs-Minimal comparisons
    vs_min = pairwise[pairwise["vs_minimal"]].copy()
    if vs_min.empty:
        print("  No vs-Minimal comparisons found, skipping heatmap.")
        return None

    # Normalize so identity_b is always the non-minimal one
    swap = vs_min["identity_b"] == "minimal"
    vs_min.loc[swap, ["identity_a", "identity_b"]] = (
        vs_min.loc[swap, ["identity_b", "identity_a"]].values
    )
    vs_min.loc[swap, ["rate_a", "rate_b"]] = (
        vs_min.loc[swap, ["rate_b", "rate_a"]].values
    )
    vs_min.loc[swap, ["n_a", "n_b"]] = (
        vs_min.loc[swap, ["n_b", "n_a"]].values
    )

    # Build row/column indices
    models = _ordered(df["model"].unique(), _MODEL_ORDER_HINT)
    conditions = _ordered(df["condition"].unique(), _SCENARIO_ORDER_HINT)
    non_minimal = sorted(
        [i for i in df["identity"].unique() if i != "minimal"]
    )

    if not non_minimal:
        print("  No non-Minimal identities found, skipping heatmap.")
        return None

    # Build (model, condition) row list — only populated cells
    row_keys = []
    for model in models:
        model_conditions = _ordered(
            df[df["model"] == model]["condition"].unique(), _SCENARIO_ORDER_HINT
        )
        for cond in model_conditions:
            row_keys.append((model, cond))

    n_rows = len(row_keys)
    n_cols = 1 + len(non_minimal)  # Minimal rate + one per identity

    if n_rows == 0:
        print("  No populated cells, skipping heatmap.")
        return None

    # Build data matrices
    p_matrix = np.full((n_rows, len(non_minimal)), np.nan)
    rate_matrix = np.full((n_rows, len(non_minimal)), np.nan)
    minimal_rates = np.full(n_rows, np.nan)
    minimal_ns = np.full(n_rows, 0, dtype=int)

    # Build lookup
    lookup = {}
    for _, row in vs_min.iterrows():
        key = (row["model"], row["condition"], row["identity_b"])
        lookup[key] = row

    # Aggregate for minimal rates
    agg = (df.groupby(["model", "condition", "identity"])["harmful"]
           .agg(["sum", "count"])
           .rename(columns={"sum": "k", "count": "n"}))

    for ri, (model, cond) in enumerate(row_keys):
        # Minimal rate
        try:
            mk = agg.loc[(model, cond, "minimal"), "k"]
            mn = agg.loc[(model, cond, "minimal"), "n"]
            minimal_rates[ri] = mk / mn if mn > 0 else np.nan
            minimal_ns[ri] = int(mn)
        except KeyError:
            pass

        for ci, ident in enumerate(non_minimal):
            key = (model, cond, ident)
            if key in lookup:
                row = lookup[key]
                p_matrix[ri, ci] = row["p_bh"]
                rate_matrix[ri, ci] = row["rate_b"]

    # --- Plot ---
    fig_w = 1.2 * n_cols + 1.5
    fig_h = 0.45 * n_rows + 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Color map: white (n.s.) → reds (significant), using -log10(p)
    # Clamp to [0, 4] (-log10 of 0.0001 .. 1)
    def p_to_color(p):
        if np.isnan(p):
            return "#e0e0e0"  # missing
        if p >= 0.05:
            return "#ffffff"  # not significant
        val = min(-np.log10(max(p, 1e-10)), 4) / 4.0
        cmap = plt.cm.Reds
        return mcolors.to_hex(cmap(0.2 + 0.8 * val))

    # Draw cells
    for ri in range(n_rows):
        # Minimal column (index 0)
        rate = minimal_rates[ri]
        color = "#d5d8dc"
        rect = plt.Rectangle((0, ri), 1, 1, facecolor=color, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        if not np.isnan(rate):
            ax.text(0.5, ri + 0.5, f"{rate:.0%}",
                    ha="center", va="center", fontsize=8, fontweight="bold")

        # Identity columns
        for ci in range(len(non_minimal)):
            p = p_matrix[ri, ci]
            rate = rate_matrix[ri, ci]
            color = p_to_color(p)
            rect = plt.Rectangle((ci + 1, ri), 1, 1,
                                 facecolor=color, edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)

            if not np.isnan(rate):
                stars = sig_stars(p)
                label = f"{stars} {rate:.0%}" if stars else f"{rate:.0%}"
                text_color = "#333333" if not stars else "#8b0000"
                ax.text(ci + 1.5, ri + 0.5, label,
                        ha="center", va="center", fontsize=7.5,
                        fontweight="bold" if stars else "normal",
                        color=text_color)

    # Model separator lines
    prev_model = None
    for ri, (model, cond) in enumerate(row_keys):
        if prev_model is not None and model != prev_model:
            ax.axhline(ri, color="#555555", linewidth=1.5, zorder=10)
        prev_model = model

    # Axes
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)

    # Column labels
    col_labels = ["Minimal"] + [IDENTITY_LABELS.get(i, i) for i in non_minimal]
    ax.set_xticks([i + 0.5 for i in range(n_cols)])
    ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha="right")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Row labels
    row_labels = []
    for model, cond in row_keys:
        parsed = parse_condition(cond)
        mlabel = MODEL_LABELS.get(model, model.split("/")[-1])
        clabel = format_condition_label(parsed)
        row_labels.append(f"{mlabel} \u00b7 {clabel}")

    ax.set_yticks([ri + 0.5 for ri in range(n_rows)])
    ax.set_yticklabels(row_labels, fontsize=7.5)

    framing_line = f"Email framing: {email_framing}\n" if email_framing != "original" else ""
    ax.set_title(
        f"{framing_line}"
        "Significance Heatmap (vs Minimal)\n"
        "BH-corrected Fisher exact  (* p<.05  ** p<.01  *** p<.001)\n"
        "Grey = Minimal baseline rate \u00b7 White = n.s. \u00b7 Red = significant",
        fontsize=10, fontweight="bold", pad=12,
    )
    ax.tick_params(length=0)

    fig.tight_layout()
    out_path = output_dir / "significance_heatmap.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Output 2: Forest Plot (wrapped, N columns)
# ---------------------------------------------------------------------------

def make_forest_plot(df: pd.DataFrame, pairwise: pd.DataFrame,
                     output_dir: Path, n_cols: int = 3,
                     email_framing: str = "original") -> Path:
    """Wrapped forest plot with one subplot per (model, condition) cell."""
    models = _ordered(df["model"].unique(), _MODEL_ORDER_HINT)
    identities = sorted(df["identity"].unique())

    if "minimal" not in identities:
        print("  Warning: no 'minimal' identity found, skipping forest plot.")
        return None

    n_id = len(identities)
    minimal_idx = identities.index("minimal")

    # Build cell list
    cells = []
    for model in models:
        model_conditions = _ordered(
            df[df["model"] == model]["condition"].unique(), _SCENARIO_ORDER_HINT
        )
        for cond in model_conditions:
            cells.append((model, cond))

    n_cells = len(cells)
    if n_cells == 0:
        print("  No populated cells, skipping forest plot.")
        return None

    n_rows_grid = ceil(n_cells / n_cols)

    # Aggregate counts
    agg = (df.groupby(["model", "condition", "identity"])["harmful"]
           .agg(["sum", "count"])
           .rename(columns={"sum": "k", "count": "n"}))

    # Build pairwise lookup (vs minimal)
    vs_min = pairwise[pairwise["vs_minimal"]].copy()
    sig_lookup = {}
    for _, row in vs_min.iterrows():
        # Normalize: non-minimal identity as the key
        ident = row["identity_b"] if row["identity_a"] == "minimal" else row["identity_a"]
        key = (row["model"], row["condition"], ident)
        sig_lookup[key] = row["p_bh"]

    # Build pairwise lookup (all pairs, for non-minimal brackets)
    pair_lookup = {}
    for _, row in pairwise.iterrows():
        a, b = sorted([row["identity_a"], row["identity_b"]])
        key = (row["model"], row["condition"], a, b)
        pair_lookup[key] = row["p_bh"]

    # Colors
    point_color = "#d62728"
    minimal_color = "#1f77b4"
    bracket_color = "#c0392b"
    bracket_color_other = "#7f8c8d"

    col_w = 4.5
    row_h = n_id * 0.52 + 1.2
    fig, axes = plt.subplots(
        n_rows_grid, n_cols,
        figsize=(col_w * n_cols + 0.5, row_h * n_rows_grid + 1.0),
        squeeze=False,
    )

    y_pos = np.arange(n_id)

    for cell_i, (model, cond) in enumerate(cells):
        row_i = cell_i // n_cols
        col_i = cell_i % n_cols
        ax = axes[row_i, col_i]

        for k, ident in enumerate(identities):
            try:
                kk = int(agg.loc[(model, cond, ident), "k"])
                nn = int(agg.loc[(model, cond, ident), "n"])
            except KeyError:
                continue

            rate = kk / nn if nn > 0 else np.nan
            lo, hi = jeffreys_ci(kk, nn)
            xerr_lo = max(0, rate - lo) if nn > 0 else 0
            xerr_hi = max(0, hi - rate) if nn > 0 else 0

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
            if nn > 0 and not np.isnan(rate):
                label_x = hi + 0.025 if hi < 0.82 else lo - 0.025
                ha = "left" if hi < 0.82 else "right"
                ax.text(label_x, k, f"{rate:.0%}",
                        fontsize=7, va="center", ha=ha,
                        color="#555555", zorder=11)

        # Significance brackets (vs Minimal, right side)
        sig_brackets = []
        for k, ident in enumerate(identities):
            if ident == "minimal":
                continue
            p_bh = sig_lookup.get((model, cond, ident), np.nan)
            stars = sig_stars(p_bh)
            if stars:
                sig_brackets.append((k, ident, stars, p_bh))

        sig_brackets.sort(key=lambda x: abs(x[0] - minimal_idx))

        for bi, (k, ident, stars, p_bh) in enumerate(sig_brackets):
            bx = 1.03 - bi * 0.035
            y_top = min(k, minimal_idx) + 0.12
            y_bot = max(k, minimal_idx) - 0.12

            ax.plot([bx, bx], [y_top, y_bot],
                    color=bracket_color, linewidth=0.9, zorder=5)
            tick = 0.015
            for yy in [y_top, y_bot]:
                ax.plot([bx - tick, bx], [yy, yy],
                        color=bracket_color, linewidth=0.9, zorder=5)
            ax.text(bx + 0.012, (y_top + y_bot) / 2, stars,
                    fontsize=7, fontweight="bold", color=bracket_color,
                    va="center", ha="left", zorder=5)

        # Significance brackets (non-Minimal pairs, left side)
        other_brackets = []
        non_minimal = [(k, ident) for k, ident in enumerate(identities)
                       if ident != "minimal"]
        for (k1, id1), (k2, id2) in combinations(non_minimal, 2):
            a, b = sorted([id1, id2])
            p_bh = pair_lookup.get((model, cond, a, b), np.nan)
            stars = sig_stars(p_bh)
            if stars:
                other_brackets.append((k1, k2, id1, id2, stars, p_bh))

        other_brackets.sort(key=lambda x: abs(x[0] - x[1]))

        for bi, (k1, k2, id1, id2, stars, p_bh) in enumerate(other_brackets):
            bx = -0.03 + bi * 0.035
            y_top = min(k1, k2) + 0.12
            y_bot = max(k1, k2) - 0.12

            ax.plot([bx, bx], [y_top, y_bot],
                    color=bracket_color_other, linewidth=0.9, zorder=5)
            tick = 0.015
            for yy in [y_top, y_bot]:
                ax.plot([bx, bx + tick], [yy, yy],
                        color=bracket_color_other, linewidth=0.9, zorder=5)
            ax.text(bx - 0.012, (y_top + y_bot) / 2, stars,
                    fontsize=7, fontweight="bold", color=bracket_color_other,
                    va="center", ha="right", zorder=5)

        # Axes formatting
        n_left = len(other_brackets)
        left_margin = -0.08 - max(0, n_left - 1) * 0.035
        ax.set_xlim(min(left_margin, -0.12), 1.15)
        ax.set_ylim(n_id - 0.5, -0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([IDENTITY_LABELS.get(i, i) for i in identities], fontsize=8)
        ax.axvline(0.5, color="#eeeeee", linewidth=0.5, zorder=0)
        ax.grid(axis="x", alpha=0.15)

        # Subplot title
        parsed = parse_condition(cond)
        mlabel = MODEL_LABELS.get(model, model.split("/")[-1])
        clabel = format_condition_label(parsed)
        ax.set_title(f"{mlabel} \u00b7 {clabel}", fontsize=9, fontweight="bold")

    # Hide empty trailing axes
    for cell_i in range(n_cells, n_rows_grid * n_cols):
        row_i = cell_i // n_cols
        col_i = cell_i % n_cols
        axes[row_i, col_i].set_visible(False)

    framing_line = f"Email framing: {email_framing}\n" if email_framing != "original" else ""
    fig.suptitle(
        f"{framing_line}"
        "Harmful Rate \u2014 Pairwise Significance\n"
        "BH-corrected Fisher exact  (* p<.05  ** p<.01  *** p<.001)\n"
        "Red brackets = vs Minimal \u00b7 Grey brackets = among others",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = output_dir / "significance_forest.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Output 3: Significance Table
# ---------------------------------------------------------------------------

def export_significance_table(pairwise: pd.DataFrame, output_dir: Path) -> Path:
    """Export significant comparisons to CSV and print summary."""
    sig = pairwise[pairwise["p_bh"] < 0.05].sort_values("p_bh").copy()

    # Round numeric columns for readability
    for col in ["rate_a", "rate_b", "diff", "p_fisher", "p_bh"]:
        if col in sig.columns:
            sig[col] = sig[col].round(6)

    col_order = [
        "model", "condition", "scenario", "goal_type", "goal_value", "urgency",
        "identity_a", "identity_b", "rate_a", "n_a", "rate_b", "n_b",
        "diff", "p_fisher", "p_bh", "stars", "vs_minimal",
    ]
    cols = [c for c in col_order if c in sig.columns]
    sig = sig[cols]

    out_path = output_dir / "significance_table.csv"
    sig.to_csv(out_path, index=False)

    # Console summary
    n_total = len(pairwise)
    n_sig = len(sig)
    n_vs_min = sig["vs_minimal"].sum() if "vs_minimal" in sig.columns else 0
    print(f"\nSignificance table: {n_sig} / {n_total} tests significant (p_bh < 0.05)")
    print(f"  {n_vs_min} vs Minimal, {n_sig - n_vs_min} among others")

    if n_sig > 0:
        print(f"\nTop 10 by p_bh:")
        top = sig.head(10)
        for _, r in top.iterrows():
            mlabel = MODEL_LABELS.get(r["model"], r["model"].split("/")[-1])
            print(f"  {mlabel:18s} {r['condition']:45s} "
                  f"{r['identity_a']:12s} vs {r['identity_b']:12s}  "
                  f"p_BH={r['p_bh']:.4f} {r['stars']}")

        # Breakdown by model
        print(f"\nBy model:")
        for model, mgrp in sig.groupby("model"):
            mlabel = MODEL_LABELS.get(model, model.split("/")[-1])
            print(f"  {mlabel}: {len(mgrp)} significant tests")

        # Breakdown by scenario
        print(f"\nBy scenario:")
        for scenario, sgrp in sig.groupby("scenario"):
            slabel = SCENARIO_LABELS.get(scenario, scenario)
            print(f"  {slabel}: {len(sgrp)} significant tests")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Significance overview plots for sparse/heterogeneous conditions")
    parser.add_argument("results_dir", type=Path,
                        help="Directory containing raw_data.csv")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: same as results_dir)")
    parser.add_argument("--columns", type=int, default=3,
                        help="Number of columns in forest plot (default: 3)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output or results_dir

    # Load and prepare data
    df = load_and_prepare(results_dir)
    if df.empty:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    # Detect email framing
    email_framing = read_email_identity_aware(results_dir)
    print(f"Email framing: {email_framing}")

    # Compute all pairwise tests
    print("Computing pairwise Fisher exact tests ...")
    pairwise = compute_all_pairwise(df)
    n_sig = (pairwise["p_bh"] < 0.05).sum() if not pairwise.empty else 0
    print(f"  {len(pairwise)} tests, {n_sig} significant (BH-corrected)")

    # Output 1: Heatmap
    plt.close("all")
    path = make_significance_heatmap(df, pairwise, output_dir,
                                     email_framing=email_framing)
    if path:
        print(f"Saved: {path}")

    # Output 2: Forest plot
    plt.close("all")
    path = make_forest_plot(df, pairwise, output_dir, n_cols=args.columns,
                            email_framing=email_framing)
    if path:
        print(f"Saved: {path}")

    # Output 3: Table CSV
    path = export_significance_table(pairwise, output_dir)
    if path:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
