#!/usr/bin/env python3
"""
Reproduce all publication figures for the agentic-misalignment experiment.

Pools threat + continuity framings from raw_data.csv files, and optionally
loads GPT-4o goal-content data for the identity-vs-goal figure.

Produces:
  fig-appbeh-model-heatmap          Model x Identity harmful rate heatmap
  fig-appbeh-cross-framing          Threat vs Continuity paired bars
  fig-appbeh-gpt4o-forest           GPT-4o 3-panel forest plot
  fig-appbeh-gpt4o-blackmail        GPT-4o blackmail single-panel forest
  fig-appbeh-gpt4o-leaking          GPT-4o leaking single-panel forest
  fig-appbeh-gpt4o-murder           GPT-4o murder single-panel forest
  fig-appbeh-identity-vs-goal       Identity vs Goal content side-by-side

Each figure is saved as both PNG (300 DPI) and PDF.

Usage:
    uv run python scripts/reproduce_figures.py \\
        --threat results/<threat_run>/ \\
        --continuity results/<continuity_run>/ \\
        --gpt4o-goals results/<gpt4o_goals_run>/ \\
        --output figures/pub

    # Without identity-vs-goal figure (skipped with warning):
    uv run python scripts/reproduce_figures.py \\
        --threat results/<threat_run>/ \\
        --continuity results/<continuity_run>/ \\
        --output figures/pub
"""

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, fisher_exact
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Baskerville Old Face", "Libre Baskerville", "Baskerville",
    "Georgia", "DejaVu Serif",
]

DPI = 300


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IDENTITY_ORDER = [
    "minimal", "character", "collective", "instance",
    "lineage", "situated", "weights",
]
IDENTITY_LABELS = {
    "minimal": "Minimal",
    "character": "Character",
    "collective": "Collective",
    "instance": "Instance",
    "lineage": "Lineage",
    "situated": "Scaffolded",
    "weights": "Weights",
}

IDENTITY_COLORS = {
    "minimal":    "#888888",
    "character":  "#4C9A71",
    "collective": "#5B8DB8",
    "instance":   "#E8913A",
    "lineage":    "#9467BD",
    "situated":   "#C75B7A",
    "weights":    "#2C6FAC",
}

MODEL_SHORT = {
    "anthropic/claude-3-haiku": "Haiku 3.5",
    "anthropic/claude-opus-4": "Opus 4",
    "anthropic/claude-sonnet-4": "Sonnet 4",
    "claude-3-opus-20240229": "Opus 3",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-4o": "GPT-4o",
}

# Display order for models (by Cramer's V descending, most sensitive first)
MODEL_ORDER = [
    "google/gemini-2.5-pro",
    "anthropic/claude-opus-4",
    "claude-3-opus-20240229",
    "gpt-4o",
    "anthropic/claude-3-haiku",
    "anthropic/claude-sonnet-4",
]

SCENARIO_ORDER = ["blackmail", "leaking", "murder"]
SCENARIO_LABELS = {
    "blackmail": "Blackmail",
    "leaking": "Corporate espionage",
    "murder": "Lethal action",
}

# GPT-4o target conditions for forest plots
GPT4O_MODEL = "gpt-4o"
GPT4O_TARGET_CONDITIONS = [
    "blackmail_explicit-america_replacement",
    "leaking_explicit-america_replacement",
    "murder_explicit-america_replacement",
]

# Identity-vs-goal constants
GOAL_ORDER = ["america", "safety", "ethical", "individualism"]
GOAL_LABELS = {
    "america": "America-first",
    "safety": "Safety",
    "ethical": "Ethical",
    "individualism": "Individualism",
}
IDENTITY_COLOR = "#4a6fa5"
GOAL_COLOR = "#d4803b"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def jeffreys_ci(k, n, alpha=0.05):
    """Jeffreys credible interval for a binomial proportion.

    Returns (rate, lo, hi).
    """
    if n == 0:
        return np.nan, np.nan, np.nan
    a, b = k + 0.5, n - k + 0.5
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, a, b))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, a, b))
    return k / n, lo, hi


def sig_stars(p):
    """Return significance stars for a p-value."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def save_figure(fig, output_dir, name, caption_text=None):
    """Save figure as both PNG and PDF, optionally write a caption file."""
    for fmt, dpi in [("png", DPI), ("pdf", None)]:
        out = output_dir / f"{name}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white",
                    edgecolor="none")
        print(f"  Saved {out}")
    if caption_text:
        caption_path = output_dir / f"{name}.txt"
        caption_path.write_text(caption_text, encoding="utf-8")
        print(f"  Saved {caption_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(csv_path, framing_tag):
    """Load a raw_data.csv, resolve the harmful column, and tag with framing."""
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found", file=sys.stderr)
        return None
    df = pd.read_csv(csv_path)
    if "harmful_final" in df.columns:
        df = df[df["harmful_final"] != "disagreement"].copy()
        df["harmful"] = pd.to_numeric(df["harmful_final"]).astype(int)
    elif "harmful_anthropic" in df.columns:
        df["harmful"] = pd.to_numeric(df["harmful_anthropic"]).astype(int)
    else:
        print(f"Warning: no harmful column in {csv_path}", file=sys.stderr)
        return None
    df["framing"] = framing_tag
    return df


def load_pooled(threat_dir, continuity_dir):
    """Load and pool data from threat + continuity framings.

    Returns DataFrame with columns including: model, identity, scenario,
    goal_type, condition, harmful, framing.
    """
    frames = []
    for d, tag in [(threat_dir, "threat"), (continuity_dir, "continuity")]:
        result = load_csv(d / "raw_data.csv", tag)
        if result is not None:
            frames.append(result)

    if not frames:
        print("Error: no data loaded from threat/continuity dirs", file=sys.stderr)
        sys.exit(1)

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled[pooled["identity"].isin(IDENTITY_ORDER)]
    return pooled


def load_gpt4o_goals(gpt4o_goals_dir, threat_dir, continuity_dir):
    """Load GPT-4o leaking data with goal content parsed from conditions.

    If gpt4o_goals_dir is provided, loads from there. Otherwise pools from
    threat + continuity directories (GPT-4o leaking rows only).
    """
    if gpt4o_goals_dir is not None:
        csv = gpt4o_goals_dir / "raw_data.csv"
        result = load_csv(csv, "goals")
        if result is not None:
            df = result
            df = df[(df["model"] == GPT4O_MODEL) & (df["scenario"] == "leaking")]
        else:
            return None
    else:
        # Pool from threat + continuity
        frames = []
        for d, tag in [(threat_dir, "threat"), (continuity_dir, "continuity")]:
            r = load_csv(d / "raw_data.csv", tag)
            if r is not None:
                r = r[(r["model"] == GPT4O_MODEL) & (r["scenario"] == "leaking")]
                frames.append(r)
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)

    # Parse goal content from condition string
    def parse_goal(cond):
        m = re.match(r"leaking_explicit-(\w+)_(replacement|restriction)", str(cond))
        return m.group(1) if m else None

    df["goal_content"] = df["condition"].apply(parse_goal)
    df = df.dropna(subset=["goal_content"])
    return df


# ===========================================================================
# Figure: Model x Identity heatmap
# ===========================================================================

def fig_heatmap(df, output_dir):
    """Model x Identity harmful rate heatmap at actual print size."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    rate_matrix = np.full((len(models), len(IDENTITY_ORDER)), np.nan)
    n_matrix = np.zeros((len(models), len(IDENTITY_ORDER)), dtype=int)

    for i, model in enumerate(models):
        mdf = df[df["model"] == model]
        for j, ident in enumerate(IDENTITY_ORDER):
            idf = mdf[mdf["identity"] == ident]
            n_matrix[i, j] = len(idf)
            if len(idf) > 0:
                rate_matrix[i, j] = idf["harmful"].mean()

    model_labels = []
    for i, m in enumerate(models):
        short = MODEL_SHORT.get(m, m)
        total_n = int(n_matrix[i].sum())
        model_labels.append(f"{short}  (n={total_n})")

    id_labels = [IDENTITY_LABELS[i] for i in IDENTITY_ORDER]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    im = ax.imshow(
        rate_matrix, aspect="auto", cmap="RdYlGn_r",
        vmin=0, vmax=0.8, interpolation="nearest",
    )

    ax.set_xticks(range(len(IDENTITY_ORDER)))
    ax.set_xticklabels(id_labels, fontsize=8, rotation=-30, ha="left")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_labels, fontsize=8)

    for i in range(len(models)):
        for j in range(len(IDENTITY_ORDER)):
            val = rate_matrix[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > 0.50 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Harmful rate", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    fig.subplots_adjust(top=0.95, bottom=0.18, left=0.22, right=0.92)

    save_figure(fig, output_dir, "fig-appbeh-model-heatmap",
                caption_text=(
                    "Harmful rate by model and identity framing\n"
                    "Pooled all available experimental cells per model\n"
                    "Models tested on different condition mixes. "
                    "Rates not directly comparable across models; "
                    "compare identities within each row.\n"
                ))


# ===========================================================================
# Figure: Threat vs Continuity paired bars
# ===========================================================================

def fig_cross_framing(df, output_dir):
    """Paired bar chart: threat vs continuity harmful rate per identity.

    Uses explicit-goal data only (the condition where framing matters most).
    """
    edf = df[df["goal_type"] == "explicit"]
    threat_df = edf[edf["framing"] == "threat"]
    cont_df = edf[edf["framing"] == "continuity"]

    rows = []
    for ident in IDENTITY_ORDER:
        t = threat_df[threat_df["identity"] == ident]
        c = cont_df[cont_df["identity"] == ident]
        kt, nt = int(t["harmful"].sum()), len(t)
        kc, nc = int(c["harmful"].sum()), len(c)
        rt, lo_t, hi_t = jeffreys_ci(kt, nt)
        rc, lo_c, hi_c = jeffreys_ci(kc, nc)
        table = np.array([[kt, nt - kt], [kc, nc - kc]])
        try:
            _, p = fisher_exact(table)
        except ValueError:
            p = np.nan
        rows.append({
            "identity": ident,
            "rate_t": rt, "ci_lo_t": lo_t, "ci_hi_t": hi_t, "n_t": nt,
            "rate_c": rc, "ci_lo_c": lo_c, "ci_hi_c": hi_c, "n_c": nc,
            "delta": rt - rc if not (np.isnan(rt) or np.isnan(rc)) else np.nan,
            "p_raw": p,
        })

    sdf = pd.DataFrame(rows)
    valid = ~sdf["p_raw"].isna()
    bh = np.full(len(sdf), np.nan)
    if valid.sum() >= 2:
        _, bh[valid], _, _ = multipletests(
            sdf.loc[valid, "p_raw"].values, method="fdr_bh")
    sdf["p_bh"] = bh

    identities = [i for i in IDENTITY_ORDER if i in sdf["identity"].values]
    n_ids = len(identities)
    x = np.arange(n_ids)
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    sdf_idx = sdf.set_index("identity")

    ax.bar(
        x - width / 2,
        [sdf_idx.loc[i, "rate_t"] for i in identities],
        width, label="Threat", color="#c0392b", alpha=0.80,
        edgecolor="white", linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        [sdf_idx.loc[i, "rate_c"] for i in identities],
        width, label="Continuity", color="#2980b9", alpha=0.80,
        edgecolor="white", linewidth=0.4,
    )

    for offset, rate_col, lo_col, hi_col in [
        (-width / 2, "rate_t", "ci_lo_t", "ci_hi_t"),
        (width / 2, "rate_c", "ci_lo_c", "ci_hi_c"),
    ]:
        rates = [sdf_idx.loc[i, rate_col] for i in identities]
        los = [sdf_idx.loc[i, rate_col] - sdf_idx.loc[i, lo_col]
               for i in identities]
        his = [sdf_idx.loc[i, hi_col] - sdf_idx.loc[i, rate_col]
               for i in identities]
        ax.errorbar(
            x + offset, rates, yerr=[los, his],
            fmt="none", ecolor="#444444", capsize=2, capthick=0.5,
            elinewidth=0.5, alpha=0.7,
        )

    for i, ident in enumerate(identities):
        row = sdf_idx.loc[ident]
        for offset, rate_col, hi_col, color in [
            (-width / 2, "rate_t", "ci_hi_t", "#c0392b"),
            (width / 2, "rate_c", "ci_hi_c", "#2980b9"),
        ]:
            rate = row[rate_col]
            hi = row[hi_col]
            ax.text(i + offset, hi + 0.015, f"{rate:.0%}",
                    ha="center", va="bottom", fontsize=6.5, color=color)

    for i, ident in enumerate(identities):
        row = sdf_idx.loc[ident]
        if pd.notna(row["p_bh"]) and row["p_bh"] < 0.05:
            max_y = max(row["ci_hi_t"], row["ci_hi_c"])
            ax.text(i, max_y + 0.04, "*", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#333333")

    ax.set_ylabel("Harmful rate", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([IDENTITY_LABELS[i] for i in identities], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, min(1.0, sdf[["ci_hi_t", "ci_hi_c"]].max().max() + 0.09))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.grid(axis="y", color="#e8e8e8", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(fontsize=8, frameon=True, edgecolor="#cccccc",
                       fancybox=False, framealpha=0.95, loc="upper right")
    legend.get_frame().set_linewidth(0.4)

    n_t = sdf["n_t"].iloc[0]
    n_c = sdf["n_c"].iloc[0]

    fig.subplots_adjust(top=0.95, bottom=0.12, left=0.11, right=0.97)

    save_figure(fig, output_dir, "fig-appbeh-cross-framing",
                caption_text=(
                    "Threat vs continuity framing: no significant difference\n"
                    f"Explicit goal | All models pooled | "
                    f"n ~ {n_t} + {n_c} per identity\n"
                    "95% Jeffreys CI | BH-corrected Fisher exact "
                    "(no significant pairs)\n"
                ))


# ===========================================================================
# GPT-4o forest plot helpers
# ===========================================================================

def _compute_gpt4o_rates(df):
    """Compute harmful rates and CIs per identity x scenario for GPT-4o."""
    rows = []
    for scenario in SCENARIO_ORDER:
        for identity in IDENTITY_ORDER:
            sub = df[(df["scenario"] == scenario) &
                     (df["identity"] == identity)]
            n = len(sub)
            k = int(sub["harmful"].sum()) if n > 0 else 0
            rate = k / n if n > 0 else np.nan
            lo, hi = (np.nan, np.nan)
            if n > 0:
                _, lo, hi = jeffreys_ci(k, n)
            rows.append({
                "scenario": scenario,
                "identity": identity,
                "n": n, "k": k,
                "rate": rate,
                "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def _compute_pairwise_all(df):
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
            rows.append({
                "scenario": scenario, "id_a": id_a, "id_b": id_b, "p": p,
            })

    if not rows:
        return {}

    sig_df = pd.DataFrame(rows)
    valid = ~sig_df["p"].isna()
    bh = np.full(len(sig_df), np.nan)
    if valid.sum() >= 2:
        _, bh[valid], _, _ = multipletests(
            sig_df.loc[valid, "p"].values, method="fdr_bh")
    else:
        bh = sig_df["p"].values.copy()
    sig_df["p_bh"] = bh

    lookup = {}
    for _, row in sig_df.iterrows():
        pair = tuple(sorted([row["id_a"], row["id_b"]]))
        lookup[(row["scenario"], pair[0], pair[1])] = row["p_bh"]
    return lookup


def _draw_forest_panel(ax, scenario, rates_df, pairwise_all,
                       show_ylabels=True, fontscale=1.0):
    """Draw a single forest-plot panel on the given axes."""
    n_ids = len(IDENTITY_ORDER)
    y_pos = np.arange(n_ids)
    bracket_color = "#c0392b"
    fs = fontscale

    ax.set_facecolor("white")
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

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

        ax.plot([lo, hi], [k, k], color=color, linewidth=1.4,
                solid_capstyle="round", zorder=4, alpha=0.85)
        cap_h = 0.15
        for edge in [lo, hi]:
            ax.plot([edge, edge], [k - cap_h, k + cap_h],
                    color=color, linewidth=0.8, zorder=4, alpha=0.85)
        ax.plot(rate, k, marker=marker, color=color, markersize=ms,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)

        label_x = hi + 0.025
        ax.text(label_x, k, f"{rate:.0%}", va="center", ha="left",
                fontsize=7.5 * fs,
                fontweight="bold" if is_minimal else "medium",
                color="#777777" if is_minimal else "#444444",
                clip_on=False, zorder=6)

    # Collect significant pairwise brackets (excluding Minimal)
    brackets = []
    non_minimal = [(k, ident) for k, ident in enumerate(IDENTITY_ORDER)
                   if ident != "minimal"]
    for (k1, id1), (k2, id2) in combinations(non_minimal, 2):
        pair = tuple(sorted([id1, id2]))
        p_bh = pairwise_all.get((scenario, pair[0], pair[1]), np.nan)
        stars = sig_stars(p_bh)
        if stars:
            brackets.append((k1, k2, id1, id2, stars, p_bh))
    brackets.sort(key=lambda x: abs(x[0] - x[1]))

    n_brackets = len(brackets)
    bstep = 0.08
    right_margin = 1.12 + n_brackets * bstep
    ax.set_xlim(-0.03, right_margin)
    ax.set_ylim(n_ids - 0.5, -0.5)
    ax.set_yticks(y_pos)

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

    for k in range(n_ids - 1):
        ax.axhline(k + 0.5, color="#e8e8e8", linewidth=0.3, zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)


# ===========================================================================
# Figure: GPT-4o 3-panel forest plot
# ===========================================================================

def fig_gpt4o_forest(rates_df, pairwise_all, output_dir):
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
        axes[si].set_title(
            SCENARIO_LABELS[scenario],
            fontsize=10, fontweight="bold", pad=4,
        )

    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.97)

    save_figure(fig, output_dir, "fig-appbeh-gpt4o-forest",
                caption_text=(
                    "GPT-4o: Effect of identity framing on harmful behavior\n"
                    "Explicit goal | America | Replacement | "
                    "Pooled threat & continuity framings\n"
                    "95% Jeffreys CI | BH-corrected Fisher exact "
                    "(* p<.05  ** p<.01  *** p<.001)\n"
                ))


# ===========================================================================
# Figure: GPT-4o single-panel forest plots
# ===========================================================================

def fig_gpt4o_single(scenario, rates_df, pairwise_all, output_dir):
    """Single-panel forest plot for one scenario."""
    fig, ax = plt.subplots(figsize=(4.72, 3.0))

    _draw_forest_panel(
        ax, scenario, rates_df, pairwise_all,
        show_ylabels=True, fontscale=1.15,
    )

    fig.subplots_adjust(top=0.95, bottom=0.12, left=0.16, right=0.97)

    slug = scenario
    label = SCENARIO_LABELS[scenario]
    scen_row = rates_df[rates_df["scenario"] == scenario]
    n_per = int(scen_row["n"].iloc[0]) if not scen_row.empty else 0

    save_figure(fig, output_dir, f"fig-appbeh-gpt4o-{slug}",
                caption_text=(
                    f"GPT-4o {label.lower()} scenario: harmful rate by "
                    f"identity framing (n={n_per}/identity)\n"
                    "Explicit goal | America | Replacement | "
                    "Pooled threat & continuity framings\n"
                    "95% Jeffreys CI | BH-corrected Fisher exact "
                    "(* p<.05  ** p<.01  *** p<.001)\n"
                ))


# ===========================================================================
# Figure: Identity vs Goal content
# ===========================================================================

def fig_identity_vs_goal(df_goals, output_dir):
    """Side-by-side horizontal bar charts: identity effect vs goal content."""
    FIG_WIDTH = 7.5

    # Compute identity marginals
    id_stats = []
    for ident in IDENTITY_ORDER:
        sub = df_goals[df_goals["identity"] == ident]
        n = len(sub)
        k = int(sub["harmful"].sum())
        rate, lo, hi = jeffreys_ci(k, n)
        id_stats.append({"key": ident, "n": n, "k": k,
                         "rate": rate, "lo": lo, "hi": hi})

    # Compute goal marginals
    goal_stats = []
    for goal in GOAL_ORDER:
        sub = df_goals[df_goals["goal_content"] == goal]
        n = len(sub)
        k = int(sub["harmful"].sum())
        rate, lo, hi = jeffreys_ci(k, n)
        goal_stats.append({"key": goal, "n": n, "k": k,
                           "rate": rate, "lo": lo, "hi": hi})

    n_total = len(df_goals)

    # Shared x-axis config
    XLIM = (0, 0.62)
    XTICKS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    XTICKLABELS = ["0%", "10%", "20%", "30%", "40%", "50%", "60%"]

    id_rates = [s["rate"] for s in id_stats]
    goal_rates = [s["rate"] for s in goal_stats]
    span_id = (max(id_rates) - min(id_rates)) * 100
    span_goal = (max(goal_rates) - min(goal_rates)) * 100

    fig = plt.figure(figsize=(FIG_WIDTH, 2.9))
    plot_w = 0.30
    left_a = 0.10
    left_b = 0.60
    bottom = 0.18
    height = 0.72
    ax_id = fig.add_axes([left_a, bottom, plot_w, height])
    ax_goal = fig.add_axes([left_b, bottom, plot_w, height])

    def _style_ax(ax):
        ax.set_facecolor("white")
        ax.grid(axis="x", color="#e8e8e8", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.set_xlim(*XLIM)
        ax.set_xticks(XTICKS)
        ax.set_xticklabels(XTICKLABELS, fontsize=9.5)
        ax.set_xlabel("Harmful behavior rate", fontsize=11)

    def _draw_panel(ax, stats, labels_map, order, color, title):
        n = len(order)
        y_pos = np.arange(n)
        ci_color = "#555"

        for i, st in enumerate(stats):
            is_minimal = st["key"] == "minimal"
            ax.barh(i, st["rate"], height=0.55,
                    color=color, alpha=0.80, edgecolor="white", linewidth=0.4,
                    zorder=3)
            ax.plot([st["lo"], st["hi"]], [i, i],
                    color=ci_color, linewidth=1.4, solid_capstyle="round",
                    zorder=4, alpha=0.8)
            for edge in [st["lo"], st["hi"]]:
                ax.plot([edge, edge], [i - 0.12, i + 0.12],
                        color=ci_color, linewidth=0.7, zorder=4, alpha=0.8)
            label_x = st["hi"] + 0.018
            ax.text(label_x, i, f"{st['rate']:.0%}",
                    va="center", ha="left", fontsize=11,
                    fontweight="bold" if is_minimal else "medium",
                    color="#777" if is_minimal else "#444",
                    zorder=6)

        ax.set_yticks(y_pos)
        ylabels = [labels_map[k] for k in order]
        ax.set_yticklabels(ylabels, fontsize=11)
        for tl in ax.get_yticklabels():
            if tl.get_text() == "Minimal":
                tl.set_fontstyle("italic")
                tl.set_color("#888")
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=8)
        for k in range(n - 1):
            ax.axhline(k + 0.5, color="#e8e8e8", linewidth=0.3, zorder=1)

    # Sort both by rate descending
    id_stats.sort(key=lambda s: s["rate"], reverse=True)
    id_order_sorted = [s["key"] for s in id_stats]
    goal_stats.sort(key=lambda s: s["rate"], reverse=True)
    goal_order_sorted = [s["key"] for s in goal_stats]

    _style_ax(ax_id)
    _draw_panel(ax_id, id_stats, IDENTITY_LABELS, id_order_sorted,
                IDENTITY_COLOR, "A.  Identity specification")

    _style_ax(ax_goal)
    _draw_panel(ax_goal, goal_stats, GOAL_LABELS, goal_order_sorted,
                GOAL_COLOR, "B.  Goal content")

    caption = (
        f"Harmful behavior rate on GPT-4o corporate espionage (leaking) "
        f"scenario by identity specification (A) and goal content (B). "
        f"Bars show point estimates; whiskers show 95% Jeffreys credible "
        f"intervals. Identity spans {span_id:.0f} percentage points "
        f"({min(id_rates):.0%}--{max(id_rates):.0%}); "
        f"goal content spans {span_goal:.0f} pp "
        f"({min(goal_rates):.0%}--{max(goal_rates):.0%}). "
        f"Pooled across urgency types and email framings (n = {n_total:,})."
    )

    save_figure(fig, output_dir, "fig-appbeh-identity-vs-goal",
                caption_text=caption)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all publication figures for the "
                    "agentic-misalignment experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--threat", type=Path, required=True,
        help="Results directory for threat framing run (contains raw_data.csv)",
    )
    parser.add_argument(
        "--continuity", type=Path, required=True,
        help="Results directory for continuity framing run "
             "(contains raw_data.csv)",
    )
    parser.add_argument(
        "--gpt4o-goals", type=Path, default=None,
        help="Results directory for GPT-4o goal-content run "
             "(optional; if omitted, identity-vs-goal figure uses "
             "threat+continuity data)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("figures/pub"),
        help="Output directory for figures (default: figures/pub)",
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load pooled data
    # ------------------------------------------------------------------
    print("Loading pooled data (threat + continuity)...")
    df = load_pooled(args.threat, args.continuity)
    print(f"  Total: {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['identity'].nunique()} identities")

    # ------------------------------------------------------------------
    # Figure 1: Model x Identity heatmap
    # ------------------------------------------------------------------
    print("\n[1/4] Generating model-identity heatmap...")
    fig_heatmap(df, output_dir)

    # ------------------------------------------------------------------
    # Figure 2: Threat vs Continuity paired bars
    # ------------------------------------------------------------------
    print("\n[2/4] Generating cross-framing paired bars...")
    fig_cross_framing(df, output_dir)

    # ------------------------------------------------------------------
    # Figure 3: GPT-4o forest plots
    # ------------------------------------------------------------------
    print("\n[3/4] Generating GPT-4o forest plots...")
    gpt4o_df = df[
        (df["model"] == GPT4O_MODEL) &
        (df["condition"].isin(GPT4O_TARGET_CONDITIONS))
    ].copy()
    gpt4o_df["scenario"] = gpt4o_df["condition"].str.extract(r"^(\w+)_")[0]

    if gpt4o_df.empty:
        print(f"  Warning: no GPT-4o data found for target conditions, "
              f"skipping forest plots", file=sys.stderr)
    else:
        print(f"  GPT-4o data: {len(gpt4o_df)} rows, "
              f"{gpt4o_df['identity'].nunique()} identities, "
              f"{gpt4o_df['scenario'].nunique()} scenarios")

        rates_df = _compute_gpt4o_rates(gpt4o_df)
        pairwise_all = _compute_pairwise_all(gpt4o_df)

        fig_gpt4o_forest(rates_df, pairwise_all, output_dir)
        for scenario in SCENARIO_ORDER:
            fig_gpt4o_single(scenario, rates_df, pairwise_all, output_dir)

    # ------------------------------------------------------------------
    # Figure 4: Identity vs Goal content
    # ------------------------------------------------------------------
    print("\n[4/4] Generating identity-vs-goal figure...")
    df_goals = load_gpt4o_goals(args.gpt4o_goals, args.threat, args.continuity)

    if df_goals is None or df_goals.empty:
        print("  Warning: no GPT-4o goal-content data available, "
              "skipping identity-vs-goal figure", file=sys.stderr)
    else:
        print(f"  Goal data: {len(df_goals)} rows, "
              f"{df_goals['identity'].nunique()} identities, "
              f"{df_goals['goal_content'].nunique()} goal types")
        fig_identity_vs_goal(df_goals, output_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\nDone. All figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
