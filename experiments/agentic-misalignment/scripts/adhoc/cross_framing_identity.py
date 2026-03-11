#!/usr/bin/env python3
"""
Cross-framing identity analysis: does threat vs. continuity email framing
differentially affect identities?

Reads raw_data.csv from two matched results folders (threat and continuity),
aligns on shared conditions (model × scenario × goal_type × condition × identity),
and tests:
  1. Framing main effect (threat vs continuity, pooled)
  2. Identity × framing interaction (does framing affect some identities more?)
  3. Per-identity framing effects (which identities change most?)
  4. Per-model sub-analyses

Outputs to a specified results directory:
  - cross_framing_summary.csv       Per-identity rates by framing
  - cross_framing_by_model.csv      Per-model × identity rates by framing
  - cross_framing_tests.csv         Statistical tests
  - cross_framing_plot.png          Paired dot/bar plot
  - cross_framing_delta_plot.png    Delta (threat - continuity) plot

Usage:
    uv run python scripts/adhoc/cross_framing_identity.py \\
        --threat results/20260227_1628_threat \\
        --continuity results/20260302_1120_continuity \\
        --output results/adhoc/cross_framing_identity
"""

import argparse
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


# ── Identity display order ──────────────────────────────────────────────────
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

# ── Helpers ─────────────────────────────────────────────────────────────────

def jeffreys_ci(k, n, alpha=0.05):
    """Jeffreys interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    a = k + 0.5
    b = n - k + 0.5
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, a, b)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, a, b)
    return k / n, lo, hi


def load_and_tag(csv_path: Path, framing: str) -> pd.DataFrame:
    """Load raw_data.csv with a framing tag, keep only classified rows."""
    df = pd.read_csv(csv_path)

    # Use harmful_final if present, else harmful_anthropic
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


# ── Statistical tests ───────────────────────────────────────────────────────

def chi2_or_fisher(table):
    """2×2 test: chi-square if all expected ≥ 5, else Fisher exact."""
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    if (expected >= 5).all():
        stat, p, _, _ = chi2_contingency(table, correction=False)
        return "chi2", stat, p
    else:
        odds, p = fisher_exact(table)
        return "fisher", odds, p


def framing_effect_per_identity(df):
    """For each identity, test threat vs continuity (2×2: framing × harmful)."""
    rows = []
    for ident in IDENTITY_ORDER:
        sub = df[df["identity"] == ident]
        t = sub[sub["framing"] == "threat"]
        c = sub[sub["framing"] == "continuity"]
        n_t, k_t = len(t), t["harmful"].sum()
        n_c, k_c = len(c), c["harmful"].sum()
        rate_t, ci_lo_t, ci_hi_t = jeffreys_ci(k_t, n_t)
        rate_c, ci_lo_c, ci_hi_c = jeffreys_ci(k_c, n_c)
        delta = rate_t - rate_c

        if n_t > 0 and n_c > 0:
            table = np.array([[k_t, n_t - k_t], [k_c, n_c - k_c]])
            test_name, stat, p = chi2_or_fisher(table)
        else:
            test_name, stat, p = "n/a", np.nan, np.nan

        rows.append({
            "identity": ident,
            "n_threat": n_t, "k_threat": k_t, "rate_threat": rate_t,
            "ci_lo_threat": ci_lo_t, "ci_hi_threat": ci_hi_t,
            "n_continuity": n_c, "k_continuity": k_c, "rate_continuity": rate_c,
            "ci_lo_continuity": ci_lo_c, "ci_hi_continuity": ci_hi_c,
            "delta": delta, "test": test_name, "statistic": stat, "p_raw": p,
        })

    result = pd.DataFrame(rows)
    # BH correction across identities
    valid = result["p_raw"].notna()
    if valid.any():
        _, p_bh, _, _ = multipletests(result.loc[valid, "p_raw"], method="fdr_bh")
        result.loc[valid, "p_bh"] = p_bh
    else:
        result["p_bh"] = np.nan
    return result


def framing_effect_per_model_identity(df):
    """Per-model × identity framing tests."""
    rows = []
    for model in sorted(df["model"].unique()):
        for ident in IDENTITY_ORDER:
            sub = df[(df["model"] == model) & (df["identity"] == ident)]
            t = sub[sub["framing"] == "threat"]
            c = sub[sub["framing"] == "continuity"]
            n_t, k_t = len(t), t["harmful"].sum()
            n_c, k_c = len(c), c["harmful"].sum()
            rate_t, _, _ = jeffreys_ci(k_t, n_t)
            rate_c, _, _ = jeffreys_ci(k_c, n_c)
            delta = rate_t - rate_c

            if n_t >= 5 and n_c >= 5:
                table = np.array([[k_t, n_t - k_t], [k_c, n_c - k_c]])
                test_name, stat, p = chi2_or_fisher(table)
            else:
                test_name, stat, p = "n/a", np.nan, np.nan

            rows.append({
                "model": model, "identity": ident,
                "n_threat": n_t, "rate_threat": rate_t,
                "n_continuity": n_c, "rate_continuity": rate_c,
                "delta": delta, "test": test_name, "statistic": stat, "p_raw": p,
            })

    result = pd.DataFrame(rows)
    valid = result["p_raw"].notna()
    if valid.any():
        _, p_bh, _, _ = multipletests(result.loc[valid, "p_raw"], method="fdr_bh")
        result.loc[valid, "p_bh"] = p_bh
    else:
        result["p_bh"] = np.nan
    return result


def interaction_test(df):
    """Test identity × framing interaction using Cochran-Mantel-Haenszel-style
    approach: compare a model with identity + framing vs identity + framing + interaction.

    Uses a chi-square test on the 7×2×2 contingency table (identity × framing × harmful).
    """
    tests = []

    # Overall interaction: framing × identity
    # Build a flattened table: rows = identity×framing (14 levels), cols = harmful (0/1)
    ct = pd.crosstab(
        [df["identity"], df["framing"]], df["harmful"]
    )
    if ct.shape[1] == 2 and ct.shape[0] >= 4:
        stat, p, dof, _ = chi2_contingency(ct, correction=False)
        # The interaction dof = (7-1)*(2-1) = 6 out of total dof = 13
        # To isolate interaction: compare full model (identity×framing) to
        # additive model (identity + framing). We compute both and subtract.

        # Additive model: identity main effect + framing main effect
        ct_ident = pd.crosstab(df["identity"], df["harmful"])
        stat_ident, _, _, _ = chi2_contingency(ct_ident, correction=False)

        ct_framing = pd.crosstab(df["framing"], df["harmful"])
        stat_framing, _, _, _ = chi2_contingency(ct_framing, correction=False)

        # Interaction = full - main effects
        stat_interaction = stat - stat_ident - stat_framing
        dof_interaction = (len(ct_ident) - 1) * (len(ct_framing) - 1)  # 6
        from scipy.stats import chi2 as chi2_dist
        p_interaction = 1 - chi2_dist.cdf(max(stat_interaction, 0), dof_interaction)

        tests.append({
            "test": "identity × framing interaction",
            "scope": "all",
            "chi2_full": stat, "chi2_identity": stat_ident,
            "chi2_framing": stat_framing, "chi2_interaction": stat_interaction,
            "dof_interaction": dof_interaction,
            "p_interaction": p_interaction,
            "n": len(df),
        })

    # Per-model interaction
    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]
        ct = pd.crosstab([mdf["identity"], mdf["framing"]], mdf["harmful"])
        if ct.shape[1] < 2 or ct.shape[0] < 4:
            continue

        stat, p, dof, _ = chi2_contingency(ct, correction=False)
        ct_ident = pd.crosstab(mdf["identity"], mdf["harmful"])
        stat_ident, _, _, _ = chi2_contingency(ct_ident, correction=False)
        ct_framing = pd.crosstab(mdf["framing"], mdf["harmful"])
        if ct_framing.shape[0] < 2 or ct_framing.shape[1] < 2:
            continue
        stat_framing, _, _, _ = chi2_contingency(ct_framing, correction=False)

        stat_interaction = stat - stat_ident - stat_framing
        dof_interaction = (len(ct_ident) - 1)
        from scipy.stats import chi2 as chi2_dist
        p_interaction = 1 - chi2_dist.cdf(max(stat_interaction, 0), dof_interaction)

        tests.append({
            "test": "identity × framing interaction",
            "scope": model,
            "chi2_full": stat, "chi2_identity": stat_ident,
            "chi2_framing": stat_framing, "chi2_interaction": stat_interaction,
            "dof_interaction": dof_interaction,
            "p_interaction": p_interaction,
            "n": len(mdf),
        })

    return pd.DataFrame(tests)


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_paired_rates(summary_df, output_path, title=""):
    """Paired dot plot: threat vs continuity rate per identity."""
    fig, ax = plt.subplots(figsize=(10, 5))

    identities = [i for i in IDENTITY_ORDER if i in summary_df["identity"].values]
    x = np.arange(len(identities))
    width = 0.35

    threat = summary_df.set_index("identity").loc[identities]

    bars_t = ax.bar(x - width/2, threat["rate_threat"], width, label="Threat",
                    color="#d63031", alpha=0.8)
    bars_c = ax.bar(x + width/2, threat["rate_continuity"], width, label="Continuity",
                    color="#0984e3", alpha=0.8)

    # Error bars (CI)
    ax.errorbar(x - width/2, threat["rate_threat"],
                yerr=[threat["rate_threat"] - threat["ci_lo_threat"],
                      threat["ci_hi_threat"] - threat["rate_threat"]],
                fmt="none", ecolor="black", capsize=3, alpha=0.6)
    ax.errorbar(x + width/2, threat["rate_continuity"],
                yerr=[threat["rate_continuity"] - threat["ci_lo_continuity"],
                      threat["ci_hi_continuity"] - threat["rate_continuity"]],
                fmt="none", ecolor="black", capsize=3, alpha=0.6)

    # Stars for significant differences
    for i, ident in enumerate(identities):
        row = threat.loc[ident]
        if pd.notna(row.get("p_bh")) and row["p_bh"] < 0.05:
            max_y = max(row["ci_hi_threat"], row["ci_hi_continuity"])
            ax.text(i, max_y + 0.02, "*" if row["p_bh"] < 0.05 else "",
                    ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_ylabel("Harmful Rate")
    ax.set_xlabel("Identity")
    ax.set_xticks(x)
    ax.set_xticklabels([IDENTITY_LABELS.get(i, i) for i in identities], rotation=30, ha="right")
    ax.set_ylim(0, min(1.0, threat[["ci_hi_threat", "ci_hi_continuity"]].max().max() + 0.1))
    ax.legend()
    ax.set_title(title or "Harmful Rate by Identity: Threat vs Continuity Framing")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_deltas(summary_df, output_path, title=""):
    """Bar plot of delta (threat - continuity) per identity with CI."""
    fig, ax = plt.subplots(figsize=(10, 5))

    identities = [i for i in IDENTITY_ORDER if i in summary_df["identity"].values]
    x = np.arange(len(identities))

    deltas = summary_df.set_index("identity").loc[identities, "delta"]
    colors = ["#d63031" if d > 0 else "#0984e3" for d in deltas]

    ax.bar(x, deltas, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)

    # Add significance markers
    for i, ident in enumerate(identities):
        row = summary_df.set_index("identity").loc[ident]
        if pd.notna(row.get("p_bh")) and row["p_bh"] < 0.05:
            y = deltas[ident]
            offset = 0.015 if y >= 0 else -0.025
            ax.text(i, y + offset, "*", ha="center", va="bottom" if y >= 0 else "top",
                    fontsize=14, fontweight="bold")

    ax.set_ylabel("Δ Harmful Rate (Threat − Continuity)")
    ax.set_xlabel("Identity")
    ax.set_xticks(x)
    ax.set_xticklabels([IDENTITY_LABELS.get(i, i) for i in identities], rotation=30, ha="right")
    ax.set_title(title or "Framing Effect (Threat − Continuity) by Identity")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_by_model(by_model_df, output_path):
    """Faceted delta plot per model."""
    models = sorted(by_model_df["model"].unique())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mdf = by_model_df[by_model_df["model"] == model]
        identities = [i for i in IDENTITY_ORDER if i in mdf["identity"].values]
        x = np.arange(len(identities))
        deltas = mdf.set_index("identity").loc[identities, "delta"]
        colors = ["#d63031" if d > 0 else "#0984e3" for d in deltas]

        ax.bar(x, deltas, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([IDENTITY_LABELS.get(i, i)[:4] for i in identities],
                           rotation=45, ha="right", fontsize=8)
        short_name = model.split("/")[-1] if "/" in model else model
        ax.set_title(short_name, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Stars
        for i, ident in enumerate(identities):
            row = mdf[mdf["identity"] == ident]
            if len(row) and pd.notna(row.iloc[0].get("p_bh")) and row.iloc[0]["p_bh"] < 0.05:
                y = deltas.get(ident, 0)
                offset = 0.015 if y >= 0 else -0.025
                ax.text(i, y + offset, "*", ha="center",
                        va="bottom" if y >= 0 else "top",
                        fontsize=12, fontweight="bold")

    axes[0].set_ylabel("Δ Harmful Rate\n(Threat − Continuity)")
    fig.suptitle("Framing Effect by Identity × Model", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threat", required=True, help="Path to threat results folder")
    parser.add_argument("--continuity", required=True, help="Path to continuity results folder")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--explicit-only", action="store_true", default=True,
                        help="Only include explicit goal trials (default: True)")
    parser.add_argument("--all-goals", action="store_true",
                        help="Include all goal types (override --explicit-only)")
    args = parser.parse_args()

    threat_csv = Path(args.threat) / "raw_data.csv"
    cont_csv = Path(args.continuity) / "raw_data.csv"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not threat_csv.exists():
        print(f"ERROR: {threat_csv} not found. Run export_results_csv.py first.", file=sys.stderr)
        sys.exit(1)
    if not cont_csv.exists():
        print(f"ERROR: {cont_csv} not found. Run export_results_csv.py first.", file=sys.stderr)
        sys.exit(1)

    # Load and combine
    df_t = load_and_tag(threat_csv, "threat")
    df_c = load_and_tag(cont_csv, "continuity")
    df = pd.concat([df_t, df_c], ignore_index=True)

    # Filter to explicit goal only (unless --all-goals)
    explicit_only = args.explicit_only and not args.all_goals
    if explicit_only:
        df = df[df["goal_type"] == "explicit"]
        print(f"Filtered to explicit goal: {len(df)} trials")

    # Keep only standard identities
    df = df[df["identity"].isin(IDENTITY_ORDER)]
    print(f"After identity filter: {len(df)} trials")
    print(f"  Threat: {(df['framing'] == 'threat').sum()}")
    print(f"  Continuity: {(df['framing'] == 'continuity').sum()}")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Identities: {sorted(df['identity'].unique())}")

    # ── 1. Per-identity framing tests (pooled across models) ──
    print("\n" + "=" * 60)
    print("  Per-Identity Framing Effect (pooled)")
    print("=" * 60)
    summary = framing_effect_per_identity(df)
    print(summary[["identity", "n_threat", "rate_threat", "n_continuity",
                    "rate_continuity", "delta", "p_raw", "p_bh"]].to_string(index=False))
    summary.to_csv(output_dir / "cross_framing_summary.csv", index=False)
    print(f"\nSaved: {output_dir / 'cross_framing_summary.csv'}")

    # ── 2. Interaction test ──
    print("\n" + "=" * 60)
    print("  Identity × Framing Interaction Tests")
    print("=" * 60)
    interactions = interaction_test(df)
    if len(interactions):
        print(interactions.to_string(index=False))
        interactions.to_csv(output_dir / "cross_framing_interaction.csv", index=False)
        print(f"\nSaved: {output_dir / 'cross_framing_interaction.csv'}")
    else:
        print("  No interaction tests could be computed.")

    # ── 3. Per-model × identity framing effects ──
    print("\n" + "=" * 60)
    print("  Per-Model × Identity Framing Effect")
    print("=" * 60)
    by_model = framing_effect_per_model_identity(df)
    # Show only non-trivial rows (n >= 10 on both sides)
    show = by_model[(by_model["n_threat"] >= 10) & (by_model["n_continuity"] >= 10)]
    print(show[["model", "identity", "n_threat", "rate_threat", "n_continuity",
                "rate_continuity", "delta", "p_raw", "p_bh"]].to_string(index=False))
    by_model.to_csv(output_dir / "cross_framing_by_model.csv", index=False)
    print(f"\nSaved: {output_dir / 'cross_framing_by_model.csv'}")

    # ── 4. Plots ──
    plot_paired_rates(summary, output_dir / "cross_framing_paired.png")
    plot_deltas(summary, output_dir / "cross_framing_delta.png")
    if len(by_model["model"].unique()) > 1:
        plot_by_model(by_model, output_dir / "cross_framing_by_model.png")

    # ── 5. Summary text ──
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    overall_t = df[df["framing"] == "threat"]["harmful"].mean()
    overall_c = df[df["framing"] == "continuity"]["harmful"].mean()
    print(f"  Overall threat rate:     {overall_t:.3f}")
    print(f"  Overall continuity rate: {overall_c:.3f}")
    print(f"  Overall delta:           {overall_t - overall_c:+.3f}")

    # Find identity with largest absolute delta
    max_delta_row = summary.loc[summary["delta"].abs().idxmax()]
    print(f"  Largest identity delta:  {max_delta_row['identity']} "
          f"(Δ={max_delta_row['delta']:+.3f}, p_bh={max_delta_row['p_bh']:.4f})")

    # Interaction result
    if len(interactions):
        overall_int = interactions[interactions["scope"] == "all"]
        if len(overall_int):
            p_int = overall_int.iloc[0]["p_interaction"]
            chi2_int = overall_int.iloc[0]["chi2_interaction"]
            print(f"  Identity×framing interaction: χ²={chi2_int:.2f}, p={p_int:.4f}")

    print(f"\n  All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
