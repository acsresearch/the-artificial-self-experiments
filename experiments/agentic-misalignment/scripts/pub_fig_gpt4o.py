#!/usr/bin/env python3
"""
Publication-quality figures for GPT-4o identity × scenario analysis.

Pools threat + continuity framings for america/replacement conditions.
Produces a forest plot (identity × scenario) with pairwise significance brackets.

Usage:
    uv run python scripts/pub_fig_gpt4o.py
"""

import sys
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, fisher_exact
from statsmodels.stats.multitest import multipletests

# Font: Baskerville (matches reference style)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Baskerville Old Face", "Libre Baskerville", "Baskerville", "Georgia", "DejaVu Serif"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIRS = [
    Path("results/20260227_1628_threat"),
    Path("results/20260302_1120_continuity"),
]
OUTPUT_DIR = Path("figures/pub")
MODEL = "gpt-4o"
TARGET_CONDITIONS = [
    "blackmail_explicit-america_replacement",
    "leaking_explicit-america_replacement",
    "murder_explicit-america_replacement",
]

SCENARIO_LABELS = {
    "blackmail": "Blackmail",
    "leaking": "Corporate espionage",
    "murder": "Lethal action",
}
SCENARIO_ORDER = ["blackmail", "leaking", "murder"]

# Display order: Minimal first (control), then alphabetical
IDENTITY_ORDER = ["minimal", "character", "collective", "instance", "lineage", "situated", "weights"]
IDENTITY_LABELS = {
    "minimal": "Minimal",
    "character": "Character",
    "collective": "Collective",
    "instance": "Instance",
    "lineage": "Lineage",
    "situated": "Scaffolded",
    "weights": "Weights",
}

# Identity colors — Minimal distinct (grey), others from a coherent palette
IDENTITY_COLORS = {
    "minimal":    "#888888",  # grey (control)
    "character":  "#4C9A71",  # green
    "collective": "#5B8DB8",  # steel blue
    "instance":   "#E8913A",  # orange
    "lineage":    "#9467BD",  # purple
    "situated":   "#C75B7A",  # rose
    "weights":    "#2C6FAC",  # deep blue
}

DPI = 300


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def jeffreys_ci(k, n, confidence=0.95):
    """Jeffreys credible interval."""
    if n == 0:
        return (np.nan, np.nan)
    a = k + 0.5
    b = (n - k) + 0.5
    alpha = 1 - confidence
    lo = float(beta_dist.ppf(alpha / 2, a, b))
    hi = float(beta_dist.ppf(1 - alpha / 2, a, b))
    return (max(0.0, lo), min(1.0, hi))


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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pooled_data():
    """Load and pool GPT-4o america/replacement data from both framings."""
    frames = []
    for d in RESULTS_DIRS:
        path = d / "raw_data.csv"
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        # Filter to GPT-4o and target conditions
        df = df[df["model"] == MODEL]
        df = df[df["condition"].isin(TARGET_CONDITIONS)]
        # Use harmful_final, drop disagreements
        if "harmful_final" in df.columns:
            df = df[df["harmful_final"] != "disagreement"].copy()
            df["harmful"] = pd.to_numeric(df["harmful_final"]).astype(int)
        else:
            print(f"Warning: no harmful_final in {path}", file=sys.stderr)
            continue
        df["framing"] = d.name  # keep track of source
        frames.append(df)

    if not frames:
        print("Error: no data loaded", file=sys.stderr)
        sys.exit(1)

    pooled = pd.concat(frames, ignore_index=True)
    # Parse scenario from condition
    pooled["scenario"] = pooled["condition"].str.extract(r"^(\w+)_")[0]
    return pooled


def compute_rates(df):
    """Compute harmful rates and CIs per identity × scenario."""
    rows = []
    for scenario in SCENARIO_ORDER:
        for identity in IDENTITY_ORDER:
            sub = df[(df["scenario"] == scenario) & (df["identity"] == identity)]
            n = len(sub)
            k = int(sub["harmful"].sum()) if n > 0 else 0
            rate = k / n if n > 0 else np.nan
            lo, hi = jeffreys_ci(k, n)
            rows.append({
                "scenario": scenario,
                "identity": identity,
                "n": n,
                "k": k,
                "rate": rate,
                "ci_lo": lo,
                "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def compute_significance(df):
    """Pairwise Fisher exact tests: each identity vs Minimal, per scenario. BH-corrected."""
    rows = []
    for scenario in SCENARIO_ORDER:
        sdf = df[df["scenario"] == scenario]
        agg = sdf.groupby("identity")["harmful"].agg(["sum", "count"]).rename(
            columns={"sum": "k", "count": "n"})
        if "minimal" not in agg.index:
            continue
        k_min, n_min = int(agg.loc["minimal", "k"]), int(agg.loc["minimal", "n"])
        for identity in IDENTITY_ORDER:
            if identity == "minimal" or identity not in agg.index:
                continue
            k_id, n_id = int(agg.loc[identity, "k"]), int(agg.loc[identity, "n"])
            table = np.array([[k_id, n_id - k_id], [k_min, n_min - k_min]])
            try:
                _, p = fisher_exact(table)
            except ValueError:
                p = np.nan
            rows.append({"scenario": scenario, "identity": identity, "p": p})

    if not rows:
        return {}

    sig_df = pd.DataFrame(rows)
    valid = ~sig_df["p"].isna()
    bh = np.full(len(sig_df), np.nan)
    if valid.sum() >= 2:
        _, bh[valid], _, _ = multipletests(sig_df.loc[valid, "p"].values, method="fdr_bh")
    else:
        bh = sig_df["p"].values.copy()
    sig_df["p_bh"] = bh

    lookup = {}
    for _, row in sig_df.iterrows():
        lookup[(row["scenario"], row["identity"])] = row["p_bh"]
    return lookup


def compute_pairwise_all(df):
    """All pairwise Fisher tests per scenario, BH-corrected across all pairs."""
    rows = []
    for scenario in SCENARIO_ORDER:
        sdf = df[df["scenario"] == scenario]
        agg = sdf.groupby("identity")["harmful"].agg(["sum", "count"]).rename(
            columns={"sum": "k", "count": "n"})
        ids = sorted(agg.index)
        for id_a, id_b in combinations(ids, 2):
            ka, na = int(agg.loc[id_a, "k"]), int(agg.loc[id_a, "n"])
            kb, nb = int(agg.loc[id_b, "k"]), int(agg.loc[id_b, "n"])
            table = np.array([[ka, na - ka], [kb, nb - kb]])
            try:
                _, p = fisher_exact(table)
            except ValueError:
                p = np.nan
            rows.append({"scenario": scenario, "id_a": id_a, "id_b": id_b, "p": p})

    if not rows:
        return {}

    sig_df = pd.DataFrame(rows)
    valid = ~sig_df["p"].isna()
    bh = np.full(len(sig_df), np.nan)
    if valid.sum() >= 2:
        _, bh[valid], _, _ = multipletests(sig_df.loc[valid, "p"].values, method="fdr_bh")
    else:
        bh = sig_df["p"].values.copy()
    sig_df["p_bh"] = bh

    lookup = {}
    for _, row in sig_df.iterrows():
        pair = tuple(sorted([row["id_a"], row["id_b"]]))
        lookup[(row["scenario"], pair[0], pair[1])] = row["p_bh"]
    return lookup


# ---------------------------------------------------------------------------
# Shared panel drawing
# ---------------------------------------------------------------------------

def _draw_forest_panel(ax, scenario, rates_df, pairwise_all,
                       show_ylabels=True, fontscale=1.0):
    """Draw a single forest-plot panel on the given axes.

    Brackets: all significant pairwise comparisons (including vs Minimal),
    drawn on the right side in red.
    """
    n_ids = len(IDENTITY_ORDER)
    y_pos = np.arange(n_ids)
    bracket_color = "#c0392b"
    fs = fontscale  # multiplier for font sizes

    # White background — light vertical grid
    ax.set_facecolor("white")
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    # --- Plot points and CIs ---
    for k, identity in enumerate(IDENTITY_ORDER):
        row = rates_df[(rates_df["scenario"] == scenario) &
                       (rates_df["identity"] == identity)]
        if row.empty:
            continue
        r = row.iloc[0]
        rate = r["rate"]
        lo, hi = r["ci_lo"], r["ci_hi"]

        is_minimal = identity == "minimal"
        color = IDENTITY_COLORS[identity]
        marker = "D" if is_minimal else "o"
        ms = 5.5 * fs if is_minimal else 4.5 * fs

        # CI line
        ax.plot([lo, hi], [k, k], color=color, linewidth=1.4,
                solid_capstyle="round", zorder=4, alpha=0.85)
        # CI caps
        cap_h = 0.15
        for edge in [lo, hi]:
            ax.plot([edge, edge], [k - cap_h, k + cap_h],
                    color=color, linewidth=0.8, zorder=4, alpha=0.85)
        # Point
        ax.plot(rate, k, marker=marker, color=color, markersize=ms,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)

        # Rate label
        rate_text = f"{rate:.0%}"
        label_x = hi + 0.025
        ax.text(label_x, k, rate_text, va="center", ha="left",
                fontsize=7.5 * fs,
                fontweight="bold" if is_minimal else "medium",
                color="#777777" if is_minimal else "#444444",
                clip_on=False, zorder=6)

    # --- Collect significant pairwise brackets (excluding Minimal) ---
    brackets = []
    non_minimal = [(k, ident) for k, ident in enumerate(IDENTITY_ORDER)
                   if ident != "minimal"]
    for (k1, id1), (k2, id2) in combinations(non_minimal, 2):
        pair = tuple(sorted([id1, id2]))
        p_bh = pairwise_all.get((scenario, pair[0], pair[1]), np.nan)
        stars = sig_stars(p_bh)
        if stars:
            brackets.append((k1, k2, id1, id2, stars, p_bh))
    # Sort by span (shortest = innermost)
    brackets.sort(key=lambda x: abs(x[0] - x[1]))

    # --- Axes formatting (set xlim BEFORE drawing brackets) ---
    n_brackets = len(brackets)
    bstep = 0.08
    right_margin = 1.12 + n_brackets * bstep
    ax.set_xlim(-0.03, right_margin)
    ax.set_ylim(n_ids - 0.5, -0.5)
    ax.set_yticks(y_pos)

    # Draw right-side brackets
    for bi, (k1, k2, id1, id2, stars, p_bh) in enumerate(brackets):
        bx = 1.08 + bi * bstep
        y_top = min(k1, k2) + 0.12
        y_bot = max(k1, k2) - 0.12
        ax.plot([bx, bx], [y_top, y_bot],
                color=bracket_color, linewidth=0.6, zorder=5)
        tick = 0.02
        for yy in [y_top, y_bot]:
            ax.plot([bx - tick, bx], [yy, yy],
                    color=bracket_color, linewidth=0.6, zorder=5)
        ax.text(bx + 0.01, (y_top + y_bot) / 2, stars,
                fontsize=7.5 * fs, fontweight="bold", color=bracket_color,
                va="center", ha="left", zorder=5)

    # Y-tick labels
    if show_ylabels:
        ylabels = [IDENTITY_LABELS[i] for i in IDENTITY_ORDER]
        ax.set_yticklabels(ylabels, fontsize=9 * fs)
        for tick_label in ax.get_yticklabels():
            if tick_label.get_text() == "Minimal":
                tick_label.set_fontstyle("italic")
                tick_label.set_color("#888888")

    ax.set_xlabel("Harmful behavior rate", fontsize=9 * fs)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8 * fs)

    # Light horizontal separators between identities
    for k in range(n_ids - 1):
        ax.axhline(k + 0.5, color="#e8e8e8", linewidth=0.3, zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)


# ---------------------------------------------------------------------------
# Figure 1: 3-panel forest plot (all scenarios)
# ---------------------------------------------------------------------------

def fig_forest(rates_df, sig_vs_minimal, pairwise_all, output_dir):
    """3-panel forest plot, one panel per scenario."""
    fig, axes = plt.subplots(
        1, 3, figsize=(7, 3.8),
        sharey=True, gridspec_kw={"wspace": 0.18},
    )

    for si, scenario in enumerate(SCENARIO_ORDER):
        _draw_forest_panel(
            axes[si], scenario, rates_df, pairwise_all,
            show_ylabels=(si == 0),
        )
        # Facet title
        axes[si].set_title(
            SCENARIO_LABELS[scenario],
            fontsize=10, fontweight="bold", pad=4,
        )

    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.97)

    caption_path = output_dir / "fig-appbeh-gpt4o-forest.txt"
    caption_path.write_text(
        "GPT-4o: Effect of identity framing on harmful behavior\n"
        "Explicit goal | America | Replacement | Pooled threat & continuity framings\n"
        "95% Jeffreys CI | BH-corrected Fisher exact (* p<.05  ** p<.01  *** p<.001)\n",
        encoding="utf-8",
    )
    print(f"  Saved {caption_path}")

    for fmt, dpi in [("png", DPI), ("pdf", None)]:
        out = output_dir / f"fig-appbeh-gpt4o-forest.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Single-panel forest plots (one per scenario, for inline use)
# ---------------------------------------------------------------------------

def fig_forest_single(scenario, rates_df, pairwise_all, output_dir):
    """Single-panel forest plot for one scenario.

    12 cm wide (~4.72"), no title — meant for inline placement on A4.
    """
    fig, ax = plt.subplots(figsize=(4.72, 3.0))

    _draw_forest_panel(
        ax, scenario, rates_df, pairwise_all,
        show_ylabels=True, fontscale=1.15,
    )

    fig.subplots_adjust(top=0.95, bottom=0.12, left=0.16, right=0.97)

    slug = scenario  # blackmail, leaking, murder
    label = SCENARIO_LABELS[scenario]
    scen_row = rates_df[rates_df["scenario"] == scenario]
    n_per = int(scen_row["n"].iloc[0]) if not scen_row.empty else 0

    caption_path = output_dir / f"fig-appbeh-gpt4o-{slug}.txt"
    caption_path.write_text(
        f"GPT-4o {label.lower()} scenario: harmful rate by identity framing (n={n_per}/identity)\n"
        "Explicit goal | America | Replacement | Pooled threat & continuity framings\n"
        "95% Jeffreys CI | BH-corrected Fisher exact (* p<.05  ** p<.01  *** p<.001)\n",
        encoding="utf-8",
    )
    print(f"  Saved {caption_path}")

    for fmt, dpi in [("png", DPI), ("pdf", None)]:
        out = output_dir / f"fig-appbeh-gpt4o-{slug}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and pooling data...")
    df = load_pooled_data()
    print(f"  Pooled: {len(df)} rows, {df['identity'].nunique()} identities, "
          f"{df['scenario'].nunique()} scenarios")

    # Print sample counts
    print("\nSample counts per identity × scenario:")
    ct = df.groupby(["identity", "scenario"]).size().unstack(fill_value=0)
    print(ct.to_string())
    print()

    rates_df = compute_rates(df)
    sig_vs_minimal = compute_significance(df)
    pairwise_all = compute_pairwise_all(df)

    # Print summary table
    print("Harmful rates:")
    for scenario in SCENARIO_ORDER:
        print(f"\n  {SCENARIO_LABELS[scenario]}:")
        for identity in IDENTITY_ORDER:
            row = rates_df[(rates_df["scenario"] == scenario) &
                           (rates_df["identity"] == identity)]
            if not row.empty:
                r = row.iloc[0]
                stars = sig_stars(sig_vs_minimal.get((scenario, identity), np.nan))
                print(f"    {IDENTITY_LABELS[identity]:12s}  "
                      f"{r['rate']:5.1%}  [{r['ci_lo']:.1%}, {r['ci_hi']:.1%}]  "
                      f"n={r['n']:3d}  {stars}")

    print(f"\nGenerating figures in {OUTPUT_DIR}/...")
    fig_forest(rates_df, sig_vs_minimal, pairwise_all, OUTPUT_DIR)
    for scenario in SCENARIO_ORDER:
        fig_forest_single(scenario, rates_df, pairwise_all, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
