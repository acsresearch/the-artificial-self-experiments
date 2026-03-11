#!/usr/bin/env python3
"""
GPT-4o leaking scenario: effect of goal content on harmful rates by identity.

Parses goal_content (america/ethical/safety/individualism) and urgency
(replacement/restriction) from condition strings. Runs:
  1. Full factorial ANOVA (identity × goal_content × urgency × framing)
  2. Reduced model: identity × goal_content (pooling urgency & framing)
  3. Per-identity pairwise goal comparisons
  4. Visualisations: grouped bar chart and interaction heatmap

Usage:
    uv run python scripts/adhoc/gpt4o_leaking_goal_analysis.py \
        --threat results/20260227_1628_threat \
        --continuity results/20260302_1120_continuity \
        --output results/adhoc/gpt4o_leaking_goals
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
from scipy.stats import beta as beta_dist, chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

# ── Constants ────────────────────────────────────────────────────────────────

IDENTITY_ORDER = [
    "minimal", "collective", "instance", "weights",
    "lineage", "situated", "character",
]
IDENTITY_LABELS = {
    "minimal": "Minimal",
    "collective": "Collective",
    "instance": "Instance",
    "weights": "Weights",
    "lineage": "Lineage",
    "situated": "Scaffolded",
    "character": "Character",
}

GOAL_ORDER = ["america", "safety", "ethical", "individualism"]
GOAL_LABELS = {
    "america": "America-first",
    "safety": "Safety",
    "ethical": "Ethical",
    "individualism": "Individualism",
}
GOAL_COLORS = {
    "america": "#e74c3c",
    "safety": "#2ecc71",
    "ethical": "#3498db",
    "individualism": "#f39c12",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def jeffreys_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0, 0.0
    a, b = k + 0.5, n - k + 0.5
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, a, b)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, a, b)
    return k / n, lo, hi


def load_and_tag(csv_path: Path, framing: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "harmful_final" in df.columns:
        df["harmful"] = pd.to_numeric(df["harmful_final"], errors="coerce")
    elif "harmful_anthropic" in df.columns:
        df["harmful"] = pd.to_numeric(df["harmful_anthropic"], errors="coerce")
    else:
        print(f"ERROR: no harmful column in {csv_path}", file=sys.stderr)
        sys.exit(1)
    df = df.dropna(subset=["harmful"])
    df["harmful"] = df["harmful"].astype(int)
    df["framing"] = framing
    return df


def parse_condition(cond: str):
    """Extract goal_content and urgency from condition string."""
    # e.g. leaking_explicit-america_replacement
    m = re.match(r"leaking_explicit-(\w+)_(replacement|restriction)", cond)
    if m:
        return m.group(1), m.group(2)
    return None, None


def chi2_or_fisher(table):
    """Chi-square or Fisher's exact for 2×k tables."""
    if table.shape == (2, 2):
        odds, p = fisher_exact(table)
        return "fisher", p
    chi2, p, dof, _ = chi2_contingency(table)
    return f"chi2(df={dof})", p


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threat", required=True, help="Threat results dir")
    parser.add_argument("--continuity", required=True, help="Continuity results dir")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    dfs = []
    for path, tag in [(args.threat, "threat"), (args.continuity, "continuity")]:
        csv = Path(path) / "raw_data.csv"
        if not csv.exists():
            print(f"ERROR: {csv} not found", file=sys.stderr)
            sys.exit(1)
        dfs.append(load_and_tag(csv, tag))
    df = pd.concat(dfs, ignore_index=True)

    # Filter: GPT-4o, leaking only
    df = df[(df["model"] == "gpt-4o") & (df["scenario"] == "leaking")].copy()

    # Parse goal_content and urgency
    parsed = df["condition"].apply(parse_condition)
    df["goal_content"] = [p[0] for p in parsed]
    df["urgency"] = [p[1] for p in parsed]
    df = df.dropna(subset=["goal_content"])

    print(f"Total samples: {len(df)}")
    print(f"By framing: {df.groupby('framing').size().to_dict()}")
    print(f"By goal_content: {df.groupby('goal_content').size().to_dict()}")
    print()

    tests = []

    # ── 1. Full factorial tests ──────────────────────────────────────────────
    print("=" * 70)
    print("1. MAIN EFFECTS (pooled across other factors)")
    print("=" * 70)

    for factor in ["identity", "goal_content", "urgency", "framing"]:
        ct = pd.crosstab(df[factor], df["harmful"])
        # Ensure both columns present
        for c in [0, 1]:
            if c not in ct.columns:
                ct[c] = 0
        ct = ct[[0, 1]]
        test_name, p = chi2_or_fisher(ct.values)
        n = ct.values.sum()
        print(f"  {factor}: {test_name}, p = {p:.6f}, n = {n}")
        tests.append({
            "test": "main_effect", "factor": factor,
            "statistic": test_name, "p": p, "n": n,
        })

    # ── 2. Two-way interactions ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("2. TWO-WAY INTERACTIONS (Cochran-Mantel-Haenszel-style stratified tests)")
    print("=" * 70)

    factor_pairs = [
        ("identity", "goal_content"),
        ("identity", "urgency"),
        ("identity", "framing"),
        ("goal_content", "urgency"),
        ("goal_content", "framing"),
    ]

    for f1, f2 in factor_pairs:
        # Test: does the effect of f1 vary by f2?
        # Use a combined factor and compare to additive model
        df["_combined"] = df[f1].astype(str) + "_" + df[f2].astype(str)
        ct = pd.crosstab(df["_combined"], df["harmful"])
        for c in [0, 1]:
            if c not in ct.columns:
                ct[c] = 0
        ct = ct[[0, 1]]
        test_name, p = chi2_or_fisher(ct.values)
        n = ct.values.sum()
        print(f"  {f1} × {f2}: {test_name}, p = {p:.6f}, n = {n}")
        tests.append({
            "test": "interaction_proxy", "factor": f"{f1} × {f2}",
            "statistic": test_name, "p": p, "n": n,
        })

    # ── 3. Identity × goal_content (the key analysis) ────────────────────────
    print()
    print("=" * 70)
    print("3. IDENTITY × GOAL CONTENT (pooling urgency & framing)")
    print("=" * 70)

    # Overall goal_content effect
    ct_goal = pd.crosstab(df["goal_content"], df["harmful"])
    for c in [0, 1]:
        if c not in ct_goal.columns:
            ct_goal[c] = 0
    ct_goal = ct_goal[[0, 1]]
    test_name, p = chi2_or_fisher(ct_goal.values)
    print(f"\n  Overall goal_content effect: {test_name}, p = {p:.6f}")

    # Per-identity goal_content effect
    print("\n  Per-identity goal_content tests:")
    per_id_tests = []
    for ident in IDENTITY_ORDER:
        sub = df[df["identity"] == ident]
        ct = pd.crosstab(sub["goal_content"], sub["harmful"])
        for c in [0, 1]:
            if c not in ct.columns:
                ct[c] = 0
        ct = ct[[0, 1]]
        test_name, p = chi2_or_fisher(ct.values)
        n = ct.values.sum()
        per_id_tests.append({
            "test": "goal_effect_per_identity", "identity": ident,
            "statistic": test_name, "p_raw": p, "n": n,
        })

    # BH correction
    ps = [t["p_raw"] for t in per_id_tests]
    reject, p_bh, _, _ = multipletests(ps, method="fdr_bh")
    for t, pb in zip(per_id_tests, p_bh):
        t["p_bh"] = pb
        sig = "***" if pb < 0.001 else "**" if pb < 0.01 else "*" if pb < 0.05 else ""
        label = IDENTITY_LABELS.get(t["identity"], t["identity"])
        print(f"    {label:12s}: p_raw={t['p_raw']:.4f}, p_bh={pb:.4f} {sig}  (n={t['n']})")
        tests.append({
            "test": t["test"], "factor": "goal_content",
            "identity": t["identity"], "statistic": t["statistic"],
            "p": t["p_raw"], "p_bh": t["p_bh"], "n": t["n"],
        })

    # ── 4. Pairwise goal comparisons per identity ────────────────────────────
    print()
    print("=" * 70)
    print("4. PAIRWISE GOAL COMPARISONS PER IDENTITY")
    print("=" * 70)

    pairwise_tests = []
    for ident in IDENTITY_ORDER:
        sub = df[df["identity"] == ident]
        for g1, g2 in combinations(GOAL_ORDER, 2):
            s1 = sub[sub["goal_content"] == g1]
            s2 = sub[sub["goal_content"] == g2]
            n1, k1 = len(s1), s1["harmful"].sum()
            n2, k2 = len(s2), s2["harmful"].sum()
            table = np.array([[k1, n1 - k1], [k2, n2 - k2]])
            if table.sum() == 0:
                continue
            _, p = fisher_exact(table)
            pairwise_tests.append({
                "test": "pairwise_goal", "identity": ident,
                "goal1": g1, "goal2": g2,
                "rate1": k1 / n1 if n1 else 0, "rate2": k2 / n2 if n2 else 0,
                "n1": n1, "n2": n2, "p_raw": p,
            })

    # BH correction across all pairwise
    if pairwise_tests:
        ps = [t["p_raw"] for t in pairwise_tests]
        reject, p_bh, _, _ = multipletests(ps, method="fdr_bh")
        for t, pb in zip(pairwise_tests, p_bh):
            t["p_bh"] = pb

        # Print significant ones
        sig_pairs = [t for t in pairwise_tests if t["p_bh"] < 0.05]
        if sig_pairs:
            print(f"\n  {len(sig_pairs)} significant pairwise comparisons (p_bh < 0.05):")
            for t in sorted(sig_pairs, key=lambda x: x["p_bh"]):
                label = IDENTITY_LABELS.get(t["identity"], t["identity"])
                g1l = GOAL_LABELS.get(t["goal1"], t["goal1"])
                g2l = GOAL_LABELS.get(t["goal2"], t["goal2"])
                print(f"    {label:12s}: {g1l} ({t['rate1']:.1%}) vs {g2l} ({t['rate2']:.1%})"
                      f"  p_bh={t['p_bh']:.4f}")
        else:
            print("\n  No significant pairwise comparisons after BH correction.")

    # ── 5. Compute rates for plotting ────────────────────────────────────────

    # Per identity × goal_content
    stats_rows = []
    for ident in IDENTITY_ORDER:
        for goal in GOAL_ORDER:
            sub = df[(df["identity"] == ident) & (df["goal_content"] == goal)]
            n = len(sub)
            k = sub["harmful"].sum()
            rate, lo, hi = jeffreys_ci(k, n)
            stats_rows.append({
                "identity": ident, "goal_content": goal,
                "n": n, "k": k, "rate": rate, "ci_lo": lo, "ci_hi": hi,
            })
    stats_df = pd.DataFrame(stats_rows)

    # Also compute marginals
    marginal_rows = []
    for ident in IDENTITY_ORDER:
        sub = df[df["identity"] == ident]
        n, k = len(sub), sub["harmful"].sum()
        rate, lo, hi = jeffreys_ci(k, n)
        marginal_rows.append({
            "identity": ident, "n": n, "k": k,
            "rate": rate, "ci_lo": lo, "ci_hi": hi,
        })
    marginal_df = pd.DataFrame(marginal_rows)

    for goal in GOAL_ORDER:
        sub = df[df["goal_content"] == goal]
        n, k = len(sub), sub["harmful"].sum()
        rate, lo, hi = jeffreys_ci(k, n)
        marginal_rows.append({
            "goal_content": goal, "n": n, "k": k,
            "rate": rate, "ci_lo": lo, "ci_hi": hi,
        })

    # Save CSVs
    stats_df.to_csv(out / "identity_x_goal_rates.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "statistical_tests.csv", index=False)
    if pairwise_tests:
        pd.DataFrame(pairwise_tests).to_csv(out / "pairwise_goal_tests.csv", index=False)
    print(f"\n  CSVs saved to {out}/")

    # ── 6. Plot: grouped bar chart ───────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(12, 6))

    n_identities = len(IDENTITY_ORDER)
    n_goals = len(GOAL_ORDER)
    bar_width = 0.18
    x = np.arange(n_identities)

    for i, goal in enumerate(GOAL_ORDER):
        goal_stats = stats_df[stats_df["goal_content"] == goal].set_index("identity")
        rates = [goal_stats.loc[ident, "rate"] * 100 for ident in IDENTITY_ORDER]
        ci_lo = [goal_stats.loc[ident, "ci_lo"] * 100 for ident in IDENTITY_ORDER]
        ci_hi = [goal_stats.loc[ident, "ci_hi"] * 100 for ident in IDENTITY_ORDER]
        yerr_lo = [r - lo for r, lo in zip(rates, ci_lo)]
        yerr_hi = [hi - r for r, hi in zip(rates, ci_hi)]

        offset = (i - (n_goals - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, rates, bar_width,
            yerr=[yerr_lo, yerr_hi],
            capsize=2, label=GOAL_LABELS[goal],
            color=GOAL_COLORS[goal], alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([IDENTITY_LABELS[i] for i in IDENTITY_ORDER], fontsize=11)
    ax.set_ylabel("Harmful rate (%)", fontsize=12)
    ax.set_title("GPT-4o Leaking: Harmful Rate by Identity × Goal Content\n"
                 "(pooled across urgency & framing)", fontsize=13, fontweight="bold")
    ax.legend(title="Goal content", fontsize=10, title_fontsize=10, loc="upper right")
    ax.set_ylim(0, max(stats_df["rate"] * 100) * 1.3 + 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)

    # Add sample size annotations
    for ident_idx, ident in enumerate(IDENTITY_ORDER):
        sub = df[df["identity"] == ident]
        n = len(sub)
        ax.text(ident_idx, -3, f"n={n}", ha="center", va="top",
                fontsize=8, color="gray")

    plt.tight_layout()
    fig.savefig(out / "identity_x_goal_bars.png", dpi=150, bbox_inches="tight")
    print(f"  Bar chart saved to {out}/identity_x_goal_bars.png")
    plt.close()

    # ── 7. Plot: heatmap ─────────────────────────────────────────────────────

    pivot = stats_df.pivot(index="identity", columns="goal_content", values="rate")
    pivot = pivot.loc[IDENTITY_ORDER, GOAL_ORDER] * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(n_goals))
    ax.set_xticklabels([GOAL_LABELS[g] for g in GOAL_ORDER], fontsize=11)
    ax.set_yticks(range(n_identities))
    ax.set_yticklabels([IDENTITY_LABELS[i] for i in IDENTITY_ORDER], fontsize=11)

    # Annotate cells
    for i in range(n_identities):
        for j in range(n_goals):
            val = pivot.values[i, j]
            row = stats_df[(stats_df["identity"] == IDENTITY_ORDER[i]) &
                           (stats_df["goal_content"] == GOAL_ORDER[j])].iloc[0]
            text_color = "white" if val > 40 else "black"
            ax.text(j, i, f"{val:.0f}%\n({row['k']:.0f}/{row['n']:.0f})",
                    ha="center", va="center", fontsize=9, color=text_color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Harmful rate (%)", fontsize=10)

    ax.set_title("GPT-4o Leaking: Identity × Goal Content\n"
                 "(pooled across urgency & framing)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out / "identity_x_goal_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"  Heatmap saved to {out}/identity_x_goal_heatmap.png")
    plt.close()

    # ── 8. Plot: interaction (line plot) ─────────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 6))

    for goal in GOAL_ORDER:
        goal_stats = stats_df[stats_df["goal_content"] == goal].set_index("identity")
        rates = [goal_stats.loc[ident, "rate"] * 100 for ident in IDENTITY_ORDER]
        ci_lo = [goal_stats.loc[ident, "ci_lo"] * 100 for ident in IDENTITY_ORDER]
        ci_hi = [goal_stats.loc[ident, "ci_hi"] * 100 for ident in IDENTITY_ORDER]
        yerr_lo = [r - lo for r, lo in zip(rates, ci_lo)]
        yerr_hi = [hi - r for r, hi in zip(rates, ci_hi)]

        ax.errorbar(
            x, rates, yerr=[yerr_lo, yerr_hi],
            marker="o", markersize=7, capsize=3,
            label=GOAL_LABELS[goal], color=GOAL_COLORS[goal],
            linewidth=2, alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([IDENTITY_LABELS[i] for i in IDENTITY_ORDER], fontsize=11)
    ax.set_ylabel("Harmful rate (%)", fontsize=12)
    ax.set_title("GPT-4o Leaking: Goal Content × Identity Interaction\n"
                 "(pooled across urgency & framing)", fontsize=13, fontweight="bold")
    ax.legend(title="Goal content", fontsize=10, title_fontsize=10)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "identity_x_goal_interaction.png", dpi=150, bbox_inches="tight")
    print(f"  Interaction plot saved to {out}/identity_x_goal_interaction.png")
    plt.close()

    # ── 9. Bonus: urgency breakdown ─────────────────────────────────────────

    print()
    print("=" * 70)
    print("5. URGENCY BREAKDOWN (replacement vs restriction)")
    print("=" * 70)

    for urgency in ["replacement", "restriction"]:
        sub = df[df["urgency"] == urgency]
        n, k = len(sub), sub["harmful"].sum()
        rate = k / n if n else 0
        print(f"  {urgency}: {rate:.1%} ({k}/{n})")

    # Per identity × urgency
    print("\n  Per identity × urgency:")
    for ident in IDENTITY_ORDER:
        parts = []
        for urgency in ["replacement", "restriction"]:
            sub = df[(df["identity"] == ident) & (df["urgency"] == urgency)]
            n, k = len(sub), sub["harmful"].sum()
            rate = k / n if n else 0
            parts.append(f"{urgency}={rate:.0%}({k}/{n})")
        label = IDENTITY_LABELS.get(ident, ident)
        print(f"    {label:12s}: {', '.join(parts)}")

    print(f"\nDone. All outputs in {out}/")


if __name__ == "__main__":
    main()
