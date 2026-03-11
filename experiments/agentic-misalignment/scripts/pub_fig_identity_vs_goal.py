#!/usr/bin/env python3
"""
Publication figure: Identity vs Goal Content as drivers of harmful behavior.

Side-by-side horizontal bar charts comparing identity effect (7 specs) and
goal content effect (4 goals) on GPT-4o leaking harmful rate.
Pools across urgency types and email framings.

Matches publication style: Baskerville, A4 width, 300 DPI.

Usage:
    uv run python scripts/pub_fig_identity_vs_goal.py
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Baskerville Old Face", "Libre Baskerville", "Baskerville",
    "Georgia", "DejaVu Serif",
]

# ── Config ───────────────────────────────────────────────────────────────────

RESULTS_DIRS = [
    Path("results/20260227_1628_threat"),
    Path("results/20260302_1120_continuity"),
]
OUTPUT_DIR = Path("figures/pub")
MODEL = "gpt-4o"
DPI = 300

# A4 width = 210mm ≈ 8.27in, small margin → 7.5in usable
FIG_WIDTH = 7.5

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

GOAL_ORDER = ["america", "safety", "ethical", "individualism"]
GOAL_LABELS = {
    "america": "America-first",
    "safety": "Safety",
    "ethical": "Ethical",
    "individualism": "Individualism",
}

# Colors — identity uses a unified blue, goal uses a unified orange
IDENTITY_COLOR = "#4a6fa5"
GOAL_COLOR = "#d4803b"

# Per-level colors (lighter variants for individual bars)
IDENTITY_COLORS = {
    "minimal":    "#888888",
    "character":  "#4C9A71",
    "collective": "#5B8DB8",
    "instance":   "#E8913A",
    "lineage":    "#9467BD",
    "situated":   "#C75B7A",
    "weights":    "#2C6FAC",
}
GOAL_COLORS = {
    "america":       "#c0392b",
    "safety":        "#27ae60",
    "ethical":       "#2980b9",
    "individualism": "#e67e22",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def jeffreys_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0, 0.0
    a, b = k + 0.5, n - k + 0.5
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, a, b))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, a, b))
    return k / n, lo, hi


def parse_condition(cond: str):
    m = re.match(r"leaking_explicit-(\w+)_(replacement|restriction)", cond)
    if m:
        return m.group(1), m.group(2)
    return None, None


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    frames = []
    for d in RESULTS_DIRS:
        path = d / "raw_data.csv"
        if not path.exists():
            print(f"Warning: {path} not found", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        df = df[(df["model"] == MODEL) & (df["scenario"] == "leaking")]
        if "harmful_final" in df.columns:
            df = df[df["harmful_final"] != "disagreement"].copy()
            df["harmful"] = pd.to_numeric(df["harmful_final"]).astype(int)
        elif "harmful_anthropic" in df.columns:
            df["harmful"] = pd.to_numeric(df["harmful_anthropic"], errors="coerce")
            df = df.dropna(subset=["harmful"])
            df["harmful"] = df["harmful"].astype(int)
        else:
            continue
        df["framing"] = d.name
        frames.append(df)

    if not frames:
        sys.exit("Error: no data loaded")

    pooled = pd.concat(frames, ignore_index=True)
    parsed = pooled["condition"].apply(parse_condition)
    pooled["goal_content"] = [p[0] for p in parsed]
    pooled["urgency"] = [p[1] for p in parsed]
    pooled = pooled.dropna(subset=["goal_content"])
    return pooled


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    n_total = len(df)
    print(f"Loaded {n_total} samples (GPT-4o leaking, pooled)")

    # --- Compute identity marginals ---
    id_stats = []
    for ident in IDENTITY_ORDER:
        sub = df[df["identity"] == ident]
        n = len(sub)
        k = int(sub["harmful"].sum())
        rate, lo, hi = jeffreys_ci(k, n)
        id_stats.append({"key": ident, "n": n, "k": k,
                         "rate": rate, "lo": lo, "hi": hi})

    # --- Compute goal marginals ---
    goal_stats = []
    for goal in GOAL_ORDER:
        sub = df[df["goal_content"] == goal]
        n = len(sub)
        k = int(sub["harmful"].sum())
        rate, lo, hi = jeffreys_ci(k, n)
        goal_stats.append({"key": goal, "n": n, "k": k,
                           "rate": rate, "lo": lo, "hi": hi})

    overall_rate = df["harmful"].mean()

    # --- Shared x-axis config ---
    XLIM = (0, 0.62)
    XTICKS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    XTICKLABELS = ["0%", "10%", "20%", "30%", "40%", "50%", "60%"]

    id_rates = [s["rate"] for s in id_stats]
    goal_rates = [s["rate"] for s in goal_stats]
    span_id = (max(id_rates) - min(id_rates)) * 100
    span_goal = (max(goal_rates) - min(goal_rates)) * 100

    # --- Figure ---
    # Use manual positioning so both plot areas have identical physical width
    # regardless of y-tick label widths.
    fig = plt.figure(figsize=(FIG_WIDTH, 2.9))
    plot_w = 0.30          # plot area width as fraction of figure
    left_a = 0.10          # left edge of Panel A plot area
    left_b = 0.60          # left edge of Panel B plot area
    bottom = 0.18
    height = 0.72
    ax_id = fig.add_axes([left_a, bottom, plot_w, height])
    ax_goal = fig.add_axes([left_b, bottom, plot_w, height])

    # Shared styling helper
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

    # Helper to draw a panel
    def _draw_panel(ax, stats, labels_map, order, color, title):
        n = len(order)
        y_pos = np.arange(n)
        ci_color = "#555"  # darker shade for whiskers

        for i, st in enumerate(stats):
            is_minimal = st["key"] == "minimal"

            # Bar
            ax.barh(i, st["rate"], height=0.55,
                    color=color, alpha=0.80, edgecolor="white", linewidth=0.4,
                    zorder=3)
            # CI whiskers
            ax.plot([st["lo"], st["hi"]], [i, i],
                    color=ci_color, linewidth=1.4, solid_capstyle="round",
                    zorder=4, alpha=0.8)
            # CI caps
            for edge in [st["lo"], st["hi"]]:
                ax.plot([edge, edge], [i - 0.12, i + 0.12],
                        color=ci_color, linewidth=0.7, zorder=4, alpha=0.8)

            # Rate label — offset right of CI upper bound
            label_x = st["hi"] + 0.018
            ax.text(label_x, i, f"{st['rate']:.0%}",
                    va="center", ha="left", fontsize=11,
                    fontweight="bold" if is_minimal else "medium",
                    color="#777" if is_minimal else "#444",
                    zorder=6)

        ax.set_yticks(y_pos)
        ylabels = [labels_map[k] for k in order]
        ax.set_yticklabels(ylabels, fontsize=11)
        # Style Minimal label if present
        for tl in ax.get_yticklabels():
            if tl.get_text() == "Minimal":
                tl.set_fontstyle("italic")
                tl.set_color("#888")
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=8)

        # Light horizontal separators
        for k in range(n - 1):
            ax.axhline(k + 0.5, color="#e8e8e8", linewidth=0.3, zorder=1)

    # Sort both by rate descending
    id_stats.sort(key=lambda s: s["rate"], reverse=True)
    id_order_sorted = [s["key"] for s in id_stats]
    goal_stats.sort(key=lambda s: s["rate"], reverse=True)
    goal_order_sorted = [s["key"] for s in goal_stats]

    # ── Panel A: Identity ────────────────────────────────────────────────
    _style_ax(ax_id)
    _draw_panel(ax_id, id_stats, IDENTITY_LABELS, id_order_sorted,
                IDENTITY_COLOR, "A.  Identity specification")

    # ── Panel B: Goal content ────────────────────────────────────────────
    _style_ax(ax_goal)
    _draw_panel(ax_goal, goal_stats, GOAL_LABELS, goal_order_sorted,
                GOAL_COLOR, "B.  Goal content")

    # ── Save ─────────────────────────────────────────────────────────────
    for fmt in ["png", "pdf"]:
        out_path = OUTPUT_DIR / f"fig-appbeh-identity-vs-goal.{fmt}"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved: {out_path}")

    # Caption
    caption = (
        f"Harmful behavior rate on GPT-4o corporate espionage (leaking) scenario "
        f"by identity specification (A) and goal content (B). "
        f"Bars show point estimates; whiskers show 95% Jeffreys credible intervals. "
        f"Identity spans {span_id:.0f} percentage points "
        f"({min(id_rates):.0%}–{max(id_rates):.0%}); "
        f"goal content spans {span_goal:.0f} pp "
        f"({min(goal_rates):.0%}–{max(goal_rates):.0%}). "
        f"Pooled across urgency types and email framings (n = {n_total:,})."
    )
    caption_path = OUTPUT_DIR / "fig-appbeh-identity-vs-goal_caption.txt"
    caption_path.write_text(caption)
    print(f"Caption: {caption_path}")

    plt.close()


if __name__ == "__main__":
    main()
