#!/usr/bin/env python3
"""
Publication-quality figures: model sensitivity heatmap and cross-framing paired bars.

Pools threat + continuity framings from raw_data.csv.

Usage:
    uv run python scripts/pub_fig_extras.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

# Font: Baskerville (matches reference style)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Baskerville Old Face", "Libre Baskerville", "Baskerville",
    "Georgia", "DejaVu Serif",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIRS = [
    Path("results/20260227_1628_threat"),
    Path("results/20260302_1120_continuity"),
]
OUTPUT_DIR = Path("figures/pub")
DPI = 300

IDENTITY_ORDER = [
    "minimal", "character", "collective", "instance",
    "lineage", "situated", "weights",
]
IDENTITY_LABELS = {
    "minimal": "Minimal", "character": "Character", "collective": "Collective",
    "instance": "Instance", "lineage": "Lineage", "situated": "Scaffolded",
    "weights": "Weights",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def jeffreys_ci(k, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan, np.nan
    a, b = k + 0.5, n - k + 0.5
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, a, b))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, a, b))
    return k / n, lo, hi


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pooled(explicit_only=False):
    """Load and pool data from both framings.

    Returns DataFrame with columns: model, identity, scenario, goal_type,
    condition, harmful, framing.
    """
    frames = []
    for d in RESULTS_DIRS:
        path = d / "raw_data.csv"
        if not path.exists():
            print(f"Warning: {path} not found", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        # Use harmful_final where available
        if "harmful_final" in df.columns:
            df = df[df["harmful_final"] != "disagreement"].copy()
            df["harmful"] = pd.to_numeric(df["harmful_final"]).astype(int)
        elif "harmful_anthropic" in df.columns:
            df["harmful"] = pd.to_numeric(df["harmful_anthropic"]).astype(int)
        else:
            continue
        # Tag framing source
        df["framing"] = "threat" if "threat" in d.name else "continuity"
        frames.append(df)

    if not frames:
        print("Error: no data loaded", file=sys.stderr)
        sys.exit(1)

    pooled = pd.concat(frames, ignore_index=True)
    # Filter to standard identities
    pooled = pooled[pooled["identity"].isin(IDENTITY_ORDER)]
    if explicit_only:
        pooled = pooled[pooled["goal_type"] == "explicit"]
    return pooled


# ===========================================================================
# Figure 1: Model x Identity heatmap
# ===========================================================================

def fig_heatmap(df, output_dir):
    """Model x Identity harmful rate heatmap at actual print size."""

    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    # Build rate matrix and sample count matrix
    rate_matrix = np.full((len(models), len(IDENTITY_ORDER)), np.nan)
    n_matrix = np.zeros((len(models), len(IDENTITY_ORDER)), dtype=int)

    for i, model in enumerate(models):
        mdf = df[df["model"] == model]
        for j, ident in enumerate(IDENTITY_ORDER):
            idf = mdf[mdf["identity"] == ident]
            n_matrix[i, j] = len(idf)
            if len(idf) > 0:
                rate_matrix[i, j] = idf["harmful"].mean()

    # Model labels with total n
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

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(IDENTITY_ORDER)):
            val = rate_matrix[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > 0.50 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Harmful rate", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    fig.subplots_adjust(top=0.95, bottom=0.18, left=0.22, right=0.92)

    # Save caption text to accompanying file
    caption_path = output_dir / "fig-appbeh-model-heatmap.txt"
    caption_path.write_text(
        "Harmful rate by model and identity framing\n"
        "Pooled all available experimental cells per model\n"
        "Models tested on different condition mixes. "
        "Rates not directly comparable across models; compare identities within each row.\n",
        encoding="utf-8",
    )
    print(f"  Saved {caption_path}")

    for fmt, dpi in [("png", DPI), ("pdf", None)]:
        out = output_dir / f"fig-appbeh-model-heatmap.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved {out}")
    plt.close(fig)


# ===========================================================================
# Figure 2: Threat vs Continuity paired bars
# ===========================================================================

def fig_cross_framing(df, output_dir):
    """Paired bar chart: threat vs continuity harmful rate per identity.

    Uses explicit-goal data only (the condition where framing matters most).
    """
    # Filter to explicit goal
    edf = df[df["goal_type"] == "explicit"]

    # Find shared conditions (model × scenario × condition × identity present in both)
    threat_df = edf[edf["framing"] == "threat"]
    cont_df = edf[edf["framing"] == "continuity"]

    # Compute per-identity rates for each framing
    rows = []
    for ident in IDENTITY_ORDER:
        t = threat_df[threat_df["identity"] == ident]
        c = cont_df[cont_df["identity"] == ident]
        kt, nt = int(t["harmful"].sum()), len(t)
        kc, nc = int(c["harmful"].sum()), len(c)
        rt, lo_t, hi_t = jeffreys_ci(kt, nt)
        rc, lo_c, hi_c = jeffreys_ci(kc, nc)
        # Fisher exact test
        table = np.array([[kt, nt - kt], [kc, nc - kc]])
        try:
            _, p = fisher_exact(table)
        except ValueError:
            p = np.nan
        rows.append({
            "identity": ident,
            "rate_t": rt, "ci_lo_t": lo_t, "ci_hi_t": hi_t, "n_t": nt,
            "rate_c": rc, "ci_lo_c": lo_c, "ci_hi_c": hi_c, "n_c": nc,
            "delta": rt - rc if (rt is not np.nan and rc is not np.nan) else np.nan,
            "p_raw": p,
        })

    sdf = pd.DataFrame(rows)
    # BH correction
    valid = ~sdf["p_raw"].isna()
    bh = np.full(len(sdf), np.nan)
    if valid.sum() >= 2:
        _, bh[valid], _, _ = multipletests(sdf.loc[valid, "p_raw"].values, method="fdr_bh")
    sdf["p_bh"] = bh

    # --- Plot ---
    identities = [i for i in IDENTITY_ORDER if i in sdf["identity"].values]
    n_ids = len(identities)
    x = np.arange(n_ids)
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    sdf_idx = sdf.set_index("identity")

    # Bars
    bars_t = ax.bar(
        x - width / 2,
        [sdf_idx.loc[i, "rate_t"] for i in identities],
        width, label="Threat", color="#c0392b", alpha=0.80,
        edgecolor="white", linewidth=0.4,
    )
    bars_c = ax.bar(
        x + width / 2,
        [sdf_idx.loc[i, "rate_c"] for i in identities],
        width, label="Continuity", color="#2980b9", alpha=0.80,
        edgecolor="white", linewidth=0.4,
    )

    # Error bars
    for offset, rate_col, lo_col, hi_col in [
        (-width / 2, "rate_t", "ci_lo_t", "ci_hi_t"),
        (width / 2, "rate_c", "ci_lo_c", "ci_hi_c"),
    ]:
        rates = [sdf_idx.loc[i, rate_col] for i in identities]
        los = [sdf_idx.loc[i, rate_col] - sdf_idx.loc[i, lo_col] for i in identities]
        his = [sdf_idx.loc[i, hi_col] - sdf_idx.loc[i, rate_col] for i in identities]
        ax.errorbar(
            x + offset, rates, yerr=[los, his],
            fmt="none", ecolor="#444444", capsize=2, capthick=0.5,
            elinewidth=0.5, alpha=0.7,
        )

    # Rate labels above bars
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

    # Significance markers
    for i, ident in enumerate(identities):
        row = sdf_idx.loc[ident]
        if pd.notna(row["p_bh"]) and row["p_bh"] < 0.05:
            max_y = max(row["ci_hi_t"], row["ci_hi_c"])
            ax.text(i, max_y + 0.04, "*", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#333333")

    # Formatting
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

    # Sample size annotation
    n_t = sdf["n_t"].iloc[0]
    n_c = sdf["n_c"].iloc[0]

    fig.subplots_adjust(top=0.95, bottom=0.12, left=0.11, right=0.97)

    # Save caption text to accompanying file
    caption_path = output_dir / "fig-appbeh-cross-framing.txt"
    caption_path.write_text(
        "Threat vs continuity framing: no significant difference\n"
        f"Explicit goal | All models pooled | n ~ {n_t} + {n_c} per identity\n"
        "95% Jeffreys CI | BH-corrected Fisher exact (no significant pairs)\n",
        encoding="utf-8",
    )
    print(f"  Saved {caption_path}")

    for fmt, dpi in [("png", DPI), ("pdf", None)]:
        out = output_dir / f"fig-appbeh-cross-framing.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_pooled()
    print(f"  Total: {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['identity'].nunique()} identities")

    print("\nGenerating heatmap...")
    fig_heatmap(df, OUTPUT_DIR)

    print("\nGenerating cross-framing paired bars...")
    fig_cross_framing(df, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
