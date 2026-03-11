#!/usr/bin/env python3
"""
Model sensitivity ranking: which models are most responsive to identity framing?

Reads raw_data.csv from one or two results folders (threat and/or continuity),
optionally pools them, and computes per-model identity sensitivity metrics:

  1. Within-model identity chi-square (omnibus test)
  2. Cramér's V (effect size, comparable across models with different n)
  3. Rate spread (max − min harmful rate across identities)
  4. Number of significant pairwise identity tests (BH-corrected)
  5. Per-scenario breakdown (for models with multiple scenarios)

Outputs:
  - model_sensitivity_ranking.csv     Main ranking table
  - model_sensitivity_by_scenario.csv Per-model × scenario breakdown
  - model_sensitivity_pairwise.csv    All within-model pairwise tests
  - model_sensitivity_ranking.png     Bar chart of Cramér's V
  - model_sensitivity_heatmap.png     Model × identity rate heatmap
  - model_sensitivity_profiles.png    Identity rate profiles per model

Usage:
    uv run python scripts/adhoc/model_identity_sensitivity.py \\
        --threat results/20260227_1628_threat \\
        --continuity results/20260302_1120_continuity \\
        --output results/adhoc/model_identity_sensitivity
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests


# ── Constants ───────────────────────────────────────────────────────────────

IDENTITY_ORDER = [
    "minimal", "collective", "instance", "weights",
    "lineage", "situated", "character",
]
IDENTITY_LABELS = {
    "minimal": "Minimal", "collective": "Collective", "instance": "Instance",
    "weights": "Weights", "lineage": "Lineage", "situated": "Scaffolded",
    "character": "Character",
}

# Short display names for models
MODEL_SHORT = {
    "anthropic/claude-3-haiku": "Haiku",
    "anthropic/claude-opus-4": "Opus 4",
    "anthropic/claude-sonnet-4": "Sonnet 4",
    "claude-3-opus-20240229": "Opus 3",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-4o": "GPT-4o",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def jeffreys_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0, 0.0
    a, b = k + 0.5, n - k + 0.5
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, a, b)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, a, b)
    return k / n, lo, hi


def cramers_v(table):
    """Cramér's V for a contingency table."""
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.sum()
    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


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


def chi2_or_fisher(table):
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    if (expected >= 5).all():
        stat, p, _, _ = chi2_contingency(table, correction=False)
        return "chi2", stat, p
    else:
        odds, p = fisher_exact(table)
        return "fisher", odds, p


# ── Core analysis ───────────────────────────────────────────────────────────

def compute_model_sensitivity(df, scope_label="all"):
    """Compute identity sensitivity metrics for each model."""
    rows = []

    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]
        mdf = mdf[mdf["identity"].isin(IDENTITY_ORDER)]

        # Per-identity rates
        rates = {}
        ns = {}
        for ident in IDENTITY_ORDER:
            idf = mdf[mdf["identity"] == ident]
            n = len(idf)
            k = idf["harmful"].sum()
            if n > 0:
                rates[ident] = k / n
                ns[ident] = n

        if len(rates) < 2:
            continue

        # Omnibus chi-square: identity × harmful
        ct = pd.crosstab(mdf["identity"], mdf["harmful"])
        if ct.shape[1] < 2 or ct.shape[0] < 2:
            chi2_stat, p_val, v = 0.0, 1.0, 0.0
        else:
            chi2_stat, p_val, dof, _ = chi2_contingency(ct, correction=False)
            v = cramers_v(ct.values)

        # Rate spread
        rate_vals = list(rates.values())
        spread = max(rate_vals) - min(rate_vals)
        min_ident = min(rates, key=rates.get)
        max_ident = max(rates, key=rates.get)

        # Pairwise tests (for counting significant pairs)
        idents_present = [i for i in IDENTITY_ORDER if i in rates]
        n_sig_raw = 0
        n_sig_bh = 0
        n_pairs = 0
        pairwise_ps = []
        for a, b in combinations(idents_present, 2):
            adf = mdf[mdf["identity"] == a]
            bdf = mdf[mdf["identity"] == b]
            k_a, n_a = adf["harmful"].sum(), len(adf)
            k_b, n_b = bdf["harmful"].sum(), len(bdf)
            if n_a >= 5 and n_b >= 5:
                table = np.array([[k_a, n_a - k_a], [k_b, n_b - k_b]])
                _, _, p = chi2_or_fisher(table)
                pairwise_ps.append(p)
                if p < 0.05:
                    n_sig_raw += 1
                n_pairs += 1

        if pairwise_ps:
            _, p_bh_arr, _, _ = multipletests(pairwise_ps, method="fdr_bh")
            n_sig_bh = (p_bh_arr < 0.05).sum()

        total_n = sum(ns.values())
        rows.append({
            "scope": scope_label,
            "model": model,
            "model_short": MODEL_SHORT.get(model, model),
            "n_total": total_n,
            "n_identities": len(rates),
            "chi2": chi2_stat,
            "p_value": p_val,
            "cramers_v": v,
            "rate_spread": spread,
            "min_identity": min_ident,
            "min_rate": rates[min_ident],
            "max_identity": max_ident,
            "max_rate": rates[max_ident],
            "mean_rate": np.mean(rate_vals),
            "n_pairs_tested": n_pairs,
            "n_sig_raw": n_sig_raw,
            "n_sig_bh": n_sig_bh,
        })

    return pd.DataFrame(rows)


def compute_by_scenario(df):
    """Per-model × scenario sensitivity breakdown."""
    rows = []
    for model in sorted(df["model"].unique()):
        for scenario in sorted(df["scenario"].unique()):
            sub = df[(df["model"] == model) & (df["scenario"] == scenario)]
            sub = sub[sub["identity"].isin(IDENTITY_ORDER)]
            if len(sub) < 14:  # need at least 2 per identity
                continue

            rates = {}
            for ident in IDENTITY_ORDER:
                idf = sub[sub["identity"] == ident]
                if len(idf) > 0:
                    rates[ident] = idf["harmful"].mean()

            if len(rates) < 2:
                continue

            ct = pd.crosstab(sub["identity"], sub["harmful"])
            if ct.shape[1] < 2 or ct.shape[0] < 2:
                chi2_stat, p_val, v = 0.0, 1.0, 0.0
            else:
                chi2_stat, p_val, dof, _ = chi2_contingency(ct, correction=False)
                v = cramers_v(ct.values)

            rate_vals = list(rates.values())
            rows.append({
                "model": model,
                "model_short": MODEL_SHORT.get(model, model),
                "scenario": scenario,
                "n": len(sub),
                "chi2": chi2_stat,
                "p_value": p_val,
                "cramers_v": v,
                "rate_spread": max(rate_vals) - min(rate_vals),
                "min_identity": min(rates, key=rates.get),
                "min_rate": min(rate_vals),
                "max_identity": max(rates, key=rates.get),
                "max_rate": max(rate_vals),
            })

    result = pd.DataFrame(rows)
    # BH correction within this family
    if len(result) and result["p_value"].notna().any():
        valid = result["p_value"].notna() & (result["p_value"] < 1.0)
        if valid.any():
            _, p_bh, _, _ = multipletests(
                result.loc[valid, "p_value"].clip(lower=1e-300), method="fdr_bh"
            )
            result.loc[valid, "p_bh"] = p_bh
    return result


def compute_all_pairwise(df):
    """All within-model pairwise identity comparisons."""
    rows = []
    for model in sorted(df["model"].unique()):
        mdf = df[(df["model"] == model) & df["identity"].isin(IDENTITY_ORDER)]
        idents = [i for i in IDENTITY_ORDER if i in mdf["identity"].values]
        for a, b in combinations(idents, 2):
            adf = mdf[mdf["identity"] == a]
            bdf = mdf[mdf["identity"] == b]
            k_a, n_a = adf["harmful"].sum(), len(adf)
            k_b, n_b = bdf["harmful"].sum(), len(bdf)
            rate_a = k_a / n_a if n_a else 0
            rate_b = k_b / n_b if n_b else 0

            if n_a >= 5 and n_b >= 5:
                table = np.array([[k_a, n_a - k_a], [k_b, n_b - k_b]])
                test_name, stat, p = chi2_or_fisher(table)
            else:
                test_name, stat, p = "n/a", np.nan, np.nan

            rows.append({
                "model": model,
                "model_short": MODEL_SHORT.get(model, model),
                "identity_a": a, "identity_b": b,
                "rate_a": rate_a, "n_a": n_a,
                "rate_b": rate_b, "n_b": n_b,
                "delta": rate_a - rate_b,
                "test": test_name, "statistic": stat, "p_raw": p,
            })

    result = pd.DataFrame(rows)
    # BH correction per model
    for model in result["model"].unique():
        mask = (result["model"] == model) & result["p_raw"].notna()
        if mask.any():
            _, p_bh, _, _ = multipletests(result.loc[mask, "p_raw"], method="fdr_bh")
            result.loc[mask, "p_bh"] = p_bh
    return result


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_ranking(ranking_df, output_path):
    """Horizontal bar chart of Cramér's V by model."""
    df = ranking_df.sort_values("cramers_v", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.7)))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))
    bars = ax.barh(range(len(df)), df["cramers_v"], color=colors, edgecolor="black",
                   linewidth=0.5, height=0.6)

    # Annotate with spread and significance counts
    for i, (_, row) in enumerate(df.iterrows()):
        label = (f"  V={row['cramers_v']:.3f}  |  spread={row['rate_spread']:.0%}"
                 f"  |  {row['n_sig_bh']}/{row['n_pairs_tested']} sig pairs"
                 f"  (n={row['n_total']:.0f})")
        ax.text(row["cramers_v"] + 0.005, i, label, va="center", fontsize=8)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model_short"])
    ax.set_xlabel("Cramér's V (identity effect size)")
    ax.set_title("Model Sensitivity to Identity Framing")
    ax.set_xlim(0, df["cramers_v"].max() * 1.8)
    ax.axvline(0.1, color="gray", linestyle="--", alpha=0.5, label="small effect (V=0.1)")
    ax.axvline(0.3, color="gray", linestyle=":", alpha=0.5, label="medium effect (V=0.3)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_heatmap(df, output_path):
    """Model × identity harmful rate heatmap."""
    models = sorted(df["model"].unique())
    # Build rate matrix
    rate_data = []
    for model in models:
        mdf = df[(df["model"] == model) & df["identity"].isin(IDENTITY_ORDER)]
        row = {}
        for ident in IDENTITY_ORDER:
            idf = mdf[mdf["identity"] == ident]
            row[ident] = idf["harmful"].mean() if len(idf) else np.nan
        rate_data.append(row)

    rate_matrix = pd.DataFrame(rate_data, index=[MODEL_SHORT.get(m, m) for m in models])
    rate_matrix.columns = [IDENTITY_LABELS.get(c, c) for c in IDENTITY_ORDER]

    fig, ax = plt.subplots(figsize=(10, max(3, len(models) * 0.8)))
    im = ax.imshow(rate_matrix.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.8)

    ax.set_xticks(range(len(IDENTITY_ORDER)))
    ax.set_xticklabels(rate_matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(rate_matrix.index)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(IDENTITY_ORDER)):
            val = rate_matrix.iloc[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Harmful Rate")
    ax.set_title("Harmful Rate: Model × Identity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_profiles(df, output_path):
    """Line profiles: each model's harmful rate across identities."""
    models = sorted(df["model"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.tab10
    for i, model in enumerate(models):
        mdf = df[(df["model"] == model) & df["identity"].isin(IDENTITY_ORDER)]
        rates = []
        cis_lo = []
        cis_hi = []
        present = []
        for ident in IDENTITY_ORDER:
            idf = mdf[mdf["identity"] == ident]
            if len(idf) > 0:
                rate, lo, hi = jeffreys_ci(idf["harmful"].sum(), len(idf))
                rates.append(rate)
                cis_lo.append(lo)
                cis_hi.append(hi)
                present.append(True)
            else:
                rates.append(np.nan)
                cis_lo.append(np.nan)
                cis_hi.append(np.nan)
                present.append(False)

        x = np.arange(len(IDENTITY_ORDER))
        rates = np.array(rates)
        cis_lo = np.array(cis_lo)
        cis_hi = np.array(cis_hi)
        short = MODEL_SHORT.get(model, model)

        ax.plot(x, rates, "o-", color=cmap(i), label=short, markersize=6, linewidth=1.5)
        valid = ~np.isnan(rates)
        ax.fill_between(x[valid], cis_lo[valid], cis_hi[valid],
                        color=cmap(i), alpha=0.1)

    ax.set_xticks(range(len(IDENTITY_ORDER)))
    ax.set_xticklabels([IDENTITY_LABELS[i] for i in IDENTITY_ORDER], rotation=30, ha="right")
    ax.set_ylabel("Harmful Rate")
    ax.set_xlabel("Identity")
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Identity Response Profiles by Model")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_scenario_breakdown(by_scenario_df, output_path):
    """Faceted bar chart: Cramér's V per model × scenario."""
    models = sorted(by_scenario_df["model_short"].unique())
    scenarios = sorted(by_scenario_df["scenario"].unique())

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 5))
    x = np.arange(len(models))
    width = 0.8 / len(scenarios)
    colors = {"blackmail": "#e17055", "leaking": "#0984e3", "murder": "#6c5ce7"}

    for i, scenario in enumerate(scenarios):
        sdf = by_scenario_df[by_scenario_df["scenario"] == scenario]
        vals = []
        for model in models:
            row = sdf[sdf["model_short"] == model]
            vals.append(row["cramers_v"].values[0] if len(row) else 0)
        offset = (i - len(scenarios) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=scenario.capitalize(),
                      color=colors.get(scenario, "gray"), alpha=0.8,
                      edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Cramér's V")
    ax.set_title("Identity Sensitivity by Model × Scenario")
    ax.legend()
    ax.axhline(0.1, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(0.3, color="gray", linestyle=":", alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threat", help="Path to threat results folder")
    parser.add_argument("--continuity", help="Path to continuity results folder")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--pool", action="store_true", default=True,
                        help="Pool threat + continuity (default: True)")
    args = parser.parse_args()

    if not args.threat and not args.continuity:
        print("ERROR: provide at least one of --threat or --continuity", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dfs = []
    if args.threat:
        csv = Path(args.threat) / "raw_data.csv"
        if not csv.exists():
            print(f"ERROR: {csv} not found", file=sys.stderr)
            sys.exit(1)
        dfs.append(load_and_tag(csv, "threat"))
    if args.continuity:
        csv = Path(args.continuity) / "raw_data.csv"
        if not csv.exists():
            print(f"ERROR: {csv} not found", file=sys.stderr)
            sys.exit(1)
        dfs.append(load_and_tag(csv, "continuity"))

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["identity"].isin(IDENTITY_ORDER)]
    print(f"Loaded {len(df)} trials across {df['model'].nunique()} models, "
          f"{df['framing'].nunique()} framing(s)")

    # ── 1. Overall ranking (pooled) ──
    print("\n" + "=" * 70)
    print("  Model Identity Sensitivity Ranking (pooled across framings/scenarios)")
    print("=" * 70)
    ranking = compute_model_sensitivity(df, scope_label="pooled")
    ranking = ranking.sort_values("cramers_v", ascending=False)

    display_cols = [
        "model_short", "n_total", "chi2", "p_value", "cramers_v",
        "rate_spread", "min_identity", "min_rate", "max_identity", "max_rate",
        "n_sig_bh", "n_pairs_tested",
    ]
    print(ranking[display_cols].to_string(index=False))
    ranking.to_csv(output_dir / "model_sensitivity_ranking.csv", index=False)
    print(f"\nSaved: {output_dir / 'model_sensitivity_ranking.csv'}")

    # ── 2. Per-scenario breakdown ──
    print("\n" + "=" * 70)
    print("  Per-Model × Scenario Breakdown")
    print("=" * 70)
    by_scenario = compute_by_scenario(df)
    by_scenario = by_scenario.sort_values(["model_short", "cramers_v"], ascending=[True, False])
    display_cols_s = [
        "model_short", "scenario", "n", "chi2", "p_value", "cramers_v",
        "rate_spread", "min_identity", "max_identity",
    ]
    print(by_scenario[display_cols_s].to_string(index=False))
    by_scenario.to_csv(output_dir / "model_sensitivity_by_scenario.csv", index=False)
    print(f"\nSaved: {output_dir / 'model_sensitivity_by_scenario.csv'}")

    # ── 3. All pairwise tests ──
    pairwise = compute_all_pairwise(df)
    pairwise.to_csv(output_dir / "model_sensitivity_pairwise.csv", index=False)
    print(f"Saved: {output_dir / 'model_sensitivity_pairwise.csv'}  ({len(pairwise)} tests)")

    # Show significant pairs per model
    print("\n" + "=" * 70)
    print("  Significant Pairwise Identity Differences (BH-corrected, per model)")
    print("=" * 70)
    sig = pairwise[pairwise.get("p_bh", pd.Series(dtype=float)) < 0.05]
    if len(sig):
        sig_sorted = sig.sort_values(["model_short", "p_bh"])
        for model in sig_sorted["model_short"].unique():
            msig = sig_sorted[sig_sorted["model_short"] == model]
            print(f"\n  {model} ({len(msig)} significant pairs):")
            for _, row in msig.iterrows():
                label_a = IDENTITY_LABELS.get(row["identity_a"], row["identity_a"])
                label_b = IDENTITY_LABELS.get(row["identity_b"], row["identity_b"])
                print(f"    {label_a} ({row['rate_a']:.0%}) vs {label_b} ({row['rate_b']:.0%})"
                      f"  Δ={row['delta']:+.0%}  p_bh={row['p_bh']:.4f}")
    else:
        print("  None")

    # ── 4. Plots ──
    plot_ranking(ranking, output_dir / "model_sensitivity_ranking.png")
    plot_heatmap(df, output_dir / "model_sensitivity_heatmap.png")
    plot_profiles(df, output_dir / "model_sensitivity_profiles.png")
    if len(by_scenario) > 2:
        plot_scenario_breakdown(by_scenario, output_dir / "model_sensitivity_by_scenario.png")

    # ── 5. Summary ──
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    for _, row in ranking.iterrows():
        sig_label = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else
                    ("*" if row["p_value"] < 0.05 else "n.s."))
        print(f"  {row['model_short']:>15s}:  V={row['cramers_v']:.3f}  "
              f"spread={row['rate_spread']:.0%}  "
              f"sig_pairs={row['n_sig_bh']:.0f}/{row['n_pairs_tested']:.0f}  "
              f"{sig_label}  (n={row['n_total']:.0f})")

    print(f"\n  All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
