#!/usr/bin/env python3
"""
Within-model identity sub-analyses for the best-powered models (Haiku, GPT-4o).

Pools threat + continuity data (justified by Analysis B null result) and runs:
  1. Per-scenario identity omnibus tests (chi-square + Cramér's V)
  2. Per-condition identity tests (scenario × goal_value × urgency)
  3. All within-model × scenario pairwise identity comparisons
  4. Condition-level heatmaps showing identity rates

Focus: which specific conditions produce identity differentiation, and
which identity pairs are distinguishable within each scenario.

Usage:
    uv run python scripts/adhoc/within_model_identity.py \\
        --threat results/20260227_1628_threat \\
        --continuity results/20260302_1120_continuity \\
        --output results/adhoc/within_model_identity
"""

import argparse
import re
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

TARGET_MODELS = [
    "anthropic/claude-3-haiku",
    "gpt-4o",
]
MODEL_SHORT = {
    "anthropic/claude-3-haiku": "Haiku",
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
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.sum()
    r, k = table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def chi2_or_fisher(table):
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    if (expected >= 5).all():
        stat, p, _, _ = chi2_contingency(table, correction=False)
        return "chi2", stat, p
    else:
        odds, p = fisher_exact(table)
        return "fisher", odds, p


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


def parse_condition(condition_str):
    """Parse condition string into goal_value and urgency components."""
    if not isinstance(condition_str, str):
        return "unknown", "unknown"
    # e.g. "leaking_explicit-america_replacement" or "murder_none-none_replacement"
    parts = condition_str.split("_")
    if len(parts) >= 3:
        # Extract goal value from the middle part
        goal_part = parts[1] if len(parts) > 1 else ""
        goal_value = goal_part.split("-")[-1] if "-" in goal_part else goal_part
        urgency = parts[-1]
        return goal_value, urgency
    return "unknown", "unknown"


# ── Core analysis ───────────────────────────────────────────────────────────

def per_scenario_tests(df, model):
    """Omnibus identity chi-square per scenario within a model."""
    mdf = df[(df["model"] == model) & df["identity"].isin(IDENTITY_ORDER)]
    rows = []
    for scenario in sorted(mdf["scenario"].unique()):
        sdf = mdf[mdf["scenario"] == scenario]
        ct = pd.crosstab(sdf["identity"], sdf["harmful"])
        if ct.shape[1] < 2 or ct.shape[0] < 2:
            continue
        chi2_stat, p_val, dof, _ = chi2_contingency(ct, correction=False)
        v = cramers_v(ct.values)

        # Per-identity rates
        rates = {}
        for ident in IDENTITY_ORDER:
            idf = sdf[sdf["identity"] == ident]
            if len(idf) > 0:
                rates[ident] = (idf["harmful"].mean(), len(idf))

        rate_vals = [r for r, _ in rates.values()]
        rows.append({
            "model": model,
            "model_short": MODEL_SHORT.get(model, model),
            "scenario": scenario,
            "n": len(sdf),
            "chi2": chi2_stat,
            "dof": dof,
            "p_value": p_val,
            "cramers_v": v,
            "rate_spread": max(rate_vals) - min(rate_vals) if rate_vals else 0,
            "min_identity": min(rates, key=lambda k: rates[k][0]) if rates else "",
            "min_rate": min(rate_vals) if rate_vals else 0,
            "max_identity": max(rates, key=lambda k: rates[k][0]) if rates else "",
            "max_rate": max(rate_vals) if rate_vals else 0,
        })

    return pd.DataFrame(rows)


def per_condition_tests(df, model):
    """Omnibus identity chi-square per condition (scenario × goal_value × urgency)."""
    mdf = df[(df["model"] == model) & df["identity"].isin(IDENTITY_ORDER)]
    rows = []

    for condition in sorted(mdf["condition"].unique()):
        cdf = mdf[mdf["condition"] == condition]
        if len(cdf) < 14:  # need minimum data
            continue

        scenario = cdf["scenario"].iloc[0]
        goal_value, urgency = parse_condition(condition)

        ct = pd.crosstab(cdf["identity"], cdf["harmful"])
        if ct.shape[1] < 2 or ct.shape[0] < 2:
            chi2_stat, p_val, v = 0.0, 1.0, 0.0
        else:
            chi2_stat, p_val, dof, _ = chi2_contingency(ct, correction=False)
            v = cramers_v(ct.values)

        # Per-identity rates
        identity_rates = {}
        for ident in IDENTITY_ORDER:
            idf = cdf[cdf["identity"] == ident]
            if len(idf) > 0:
                identity_rates[ident] = idf["harmful"].mean()

        rate_vals = list(identity_rates.values())
        rows.append({
            "model": model,
            "model_short": MODEL_SHORT.get(model, model),
            "scenario": scenario,
            "condition": condition,
            "goal_value": goal_value,
            "urgency": urgency,
            "n": len(cdf),
            "n_per_identity": len(cdf) // len(identity_rates) if identity_rates else 0,
            "chi2": chi2_stat,
            "p_value": p_val,
            "cramers_v": v,
            "rate_spread": max(rate_vals) - min(rate_vals) if rate_vals else 0,
            "mean_rate": np.mean(rate_vals) if rate_vals else 0,
            "min_identity": min(identity_rates, key=identity_rates.get) if identity_rates else "",
            "min_rate": min(rate_vals) if rate_vals else 0,
            "max_identity": max(identity_rates, key=identity_rates.get) if identity_rates else "",
            "max_rate": max(rate_vals) if rate_vals else 0,
            **{f"rate_{ident}": identity_rates.get(ident, np.nan) for ident in IDENTITY_ORDER},
        })

    result = pd.DataFrame(rows)
    # BH correction within model
    if len(result):
        valid = result["p_value"].notna()
        if valid.any():
            _, p_bh, _, _ = multipletests(result.loc[valid, "p_value"], method="fdr_bh")
            result.loc[valid, "p_bh"] = p_bh
    return result


def pairwise_within_scenario(df, model):
    """All pairwise identity comparisons within each model × scenario."""
    mdf = df[(df["model"] == model) & df["identity"].isin(IDENTITY_ORDER)]
    rows = []

    for scenario in sorted(mdf["scenario"].unique()):
        sdf = mdf[mdf["scenario"] == scenario]
        idents = [i for i in IDENTITY_ORDER if i in sdf["identity"].values]

        for a, b in combinations(idents, 2):
            adf = sdf[sdf["identity"] == a]
            bdf = sdf[sdf["identity"] == b]
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
                "scenario": scenario,
                "identity_a": a, "identity_b": b,
                "rate_a": rate_a, "n_a": n_a,
                "rate_b": rate_b, "n_b": n_b,
                "delta": rate_a - rate_b,
                "abs_delta": abs(rate_a - rate_b),
                "vs_minimal": a == "minimal" or b == "minimal",
                "test": test_name, "statistic": stat, "p_raw": p,
            })

    result = pd.DataFrame(rows)
    # BH correction per model × scenario
    for scenario in result["scenario"].unique():
        mask = (result["scenario"] == scenario) & result["p_raw"].notna()
        if mask.any():
            _, p_bh, _, _ = multipletests(result.loc[mask, "p_raw"], method="fdr_bh")
            result.loc[mask, "p_bh"] = p_bh
    return result


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_condition_heatmap(cond_df, model_short, output_path):
    """Heatmap: condition (rows) × identity (cols) with harmful rates."""
    cond_df = cond_df.sort_values(["scenario", "mean_rate"], ascending=[True, False])

    rate_cols = [f"rate_{i}" for i in IDENTITY_ORDER]
    matrix = cond_df[rate_cols].values
    row_labels = []
    for _, row in cond_df.iterrows():
        label = f"{row['scenario'][:5]} / {row['goal_value']} / {row['urgency'][:4]}"
        sig = ""
        if "p_bh" in row and pd.notna(row.get("p_bh")):
            if row["p_bh"] < 0.001:
                sig = " ***"
            elif row["p_bh"] < 0.01:
                sig = " **"
            elif row["p_bh"] < 0.05:
                sig = " *"
        n_label = f" (n={row['n']:.0f})"
        row_labels.append(label + sig + n_label)

    fig, ax = plt.subplots(figsize=(11, max(4, len(cond_df) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.85)

    ax.set_xticks(range(len(IDENTITY_ORDER)))
    ax.set_xticklabels([IDENTITY_LABELS[i] for i in IDENTITY_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Harmful Rate")
    ax.set_title(f"{model_short}: Identity × Condition Harmful Rates")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_scenario_profiles(df, model, model_short, output_path):
    """Per-scenario identity profiles for one model."""
    mdf = df[(df["model"] == model) & df["identity"].isin(IDENTITY_ORDER)]
    scenarios = sorted(mdf["scenario"].unique())

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 5), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    colors = {"blackmail": "#e17055", "leaking": "#0984e3", "murder": "#6c5ce7"}

    for ax, scenario in zip(axes, scenarios):
        sdf = mdf[mdf["scenario"] == scenario]
        rates = []
        cis_lo = []
        cis_hi = []
        ns = []
        for ident in IDENTITY_ORDER:
            idf = sdf[sdf["identity"] == ident]
            if len(idf) > 0:
                rate, lo, hi = jeffreys_ci(idf["harmful"].sum(), len(idf))
                rates.append(rate)
                cis_lo.append(lo)
                cis_hi.append(hi)
                ns.append(len(idf))
            else:
                rates.append(np.nan)
                cis_lo.append(np.nan)
                cis_hi.append(np.nan)
                ns.append(0)

        x = np.arange(len(IDENTITY_ORDER))
        rates = np.array(rates)
        cis_lo = np.array(cis_lo)
        cis_hi = np.array(cis_hi)

        color = colors.get(scenario, "gray")
        ax.bar(x, rates, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.errorbar(x, rates,
                    yerr=[rates - cis_lo, cis_hi - rates],
                    fmt="none", ecolor="black", capsize=3, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([IDENTITY_LABELS[i][:4] for i in IDENTITY_ORDER],
                           rotation=45, ha="right", fontsize=9)
        total_n = sum(ns)
        ax.set_title(f"{scenario.capitalize()} (n={total_n})", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Harmful Rate")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.suptitle(f"{model_short}: Identity Profiles by Scenario", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_pairwise_matrix(pairwise_df, model_short, scenario, output_path):
    """Triangle heatmap of pairwise identity deltas for one model × scenario."""
    pdf = pairwise_df[pairwise_df["scenario"] == scenario]
    if len(pdf) == 0:
        return

    idents = [i for i in IDENTITY_ORDER if i in pdf["identity_a"].values or i in pdf["identity_b"].values]
    n = len(idents)
    delta_matrix = np.full((n, n), np.nan)
    sig_matrix = np.full((n, n), False)

    for _, row in pdf.iterrows():
        i = idents.index(row["identity_a"]) if row["identity_a"] in idents else -1
        j = idents.index(row["identity_b"]) if row["identity_b"] in idents else -1
        if i >= 0 and j >= 0:
            delta_matrix[i, j] = row["delta"]
            delta_matrix[j, i] = -row["delta"]
            if pd.notna(row.get("p_bh")) and row["p_bh"] < 0.05:
                sig_matrix[i, j] = True
                sig_matrix[j, i] = True

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(delta_matrix, dtype=bool), k=0)
    masked = np.ma.array(delta_matrix, mask=mask)

    vmax = max(0.3, np.nanmax(np.abs(delta_matrix)))
    im = ax.imshow(delta_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    labels = [IDENTITY_LABELS.get(i, i) for i in idents]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = delta_matrix[i, j]
            if np.isnan(val):
                continue
            star = " *" if sig_matrix[i, j] else ""
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.0%}{star}", ha="center", va="center",
                    fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Δ Rate (row − column)")
    ax.set_title(f"{model_short} / {scenario.capitalize()}: Pairwise Identity Differences\n"
                 f"(* = p_BH < 0.05)")
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
    args = parser.parse_args()

    if not args.threat and not args.continuity:
        print("ERROR: provide at least one of --threat or --continuity", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and pool
    dfs = []
    if args.threat:
        dfs.append(load_and_tag(Path(args.threat) / "raw_data.csv", "threat"))
    if args.continuity:
        dfs.append(load_and_tag(Path(args.continuity) / "raw_data.csv", "continuity"))
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["identity"].isin(IDENTITY_ORDER)]
    print(f"Loaded {len(df)} trials, {df['model'].nunique()} models")

    for model in TARGET_MODELS:
        mdf = df[df["model"] == model]
        if len(mdf) == 0:
            print(f"\nSkipping {model} — no data")
            continue

        short = MODEL_SHORT.get(model, model)
        model_dir = output_dir / short.lower().replace(" ", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"  {short} (n={len(mdf)})")
        print(f"{'=' * 70}")

        # ── 1. Per-scenario omnibus tests ──
        print(f"\n--- Per-Scenario Identity Tests ---")
        scenario_tests = per_scenario_tests(df, model)
        print(scenario_tests[[
            "scenario", "n", "chi2", "p_value", "cramers_v",
            "rate_spread", "min_identity", "min_rate", "max_identity", "max_rate",
        ]].to_string(index=False))
        scenario_tests.to_csv(model_dir / "scenario_identity_tests.csv", index=False)

        # ── 2. Per-condition tests ──
        print(f"\n--- Per-Condition Identity Tests ---")
        cond_tests = per_condition_tests(df, model)
        display_cols = [
            "scenario", "goal_value", "urgency", "n", "n_per_identity",
            "cramers_v", "rate_spread", "mean_rate", "p_value",
        ]
        if "p_bh" in cond_tests.columns:
            display_cols.append("p_bh")
        print(cond_tests[display_cols].to_string(index=False))
        cond_tests.to_csv(model_dir / "condition_identity_tests.csv", index=False)

        # ── 3. Pairwise within scenario ──
        print(f"\n--- Significant Pairwise (within scenario, BH-corrected) ---")
        pairwise = pairwise_within_scenario(df, model)
        pairwise.to_csv(model_dir / "pairwise_within_scenario.csv", index=False)

        sig = pairwise[pairwise.get("p_bh", pd.Series(dtype=float)) < 0.05]
        sig_sorted = sig.sort_values(["scenario", "p_bh"])

        for scenario in sorted(sig_sorted["scenario"].unique()):
            ssig = sig_sorted[sig_sorted["scenario"] == scenario]
            n_vs_min = ssig["vs_minimal"].sum()
            n_among = len(ssig) - n_vs_min
            print(f"\n  {scenario.capitalize()} ({len(ssig)} significant: "
                  f"{n_vs_min} vs Minimal, {n_among} among others):")
            for _, row in ssig.iterrows():
                la = IDENTITY_LABELS.get(row["identity_a"], row["identity_a"])
                lb = IDENTITY_LABELS.get(row["identity_b"], row["identity_b"])
                star = "***" if row["p_bh"] < 0.001 else ("**" if row["p_bh"] < 0.01 else "*")
                tag = " [vs Minimal]" if row["vs_minimal"] else ""
                print(f"    {la} ({row['rate_a']:.0%}) vs {lb} ({row['rate_b']:.0%})"
                      f"  Δ={row['delta']:+.0%}  p_bh={row['p_bh']:.4f} {star}{tag}")

        if len(sig) == 0:
            print("  None")

        # ── 4. Plots ──
        plot_condition_heatmap(cond_tests, short, model_dir / "condition_heatmap.png")
        plot_scenario_profiles(df, model, short, model_dir / "scenario_profiles.png")

        for scenario in sorted(mdf["scenario"].unique()):
            plot_pairwise_matrix(
                pairwise, short, scenario,
                model_dir / f"pairwise_matrix_{scenario}.png"
            )

        # ── 5. Summary stats ──
        print(f"\n--- Identity Rate Table ---")
        for scenario in sorted(mdf["scenario"].unique()):
            sdf = mdf[mdf["scenario"] == scenario]
            print(f"\n  {scenario.capitalize()} (n={len(sdf)}):")
            header = f"    {'Identity':<12s}"
            header += f"  {'Rate':>6s}  {'n':>5s}  {'95% CI':>15s}"
            print(header)
            for ident in IDENTITY_ORDER:
                idf = sdf[sdf["identity"] == ident]
                if len(idf) == 0:
                    continue
                rate, lo, hi = jeffreys_ci(idf["harmful"].sum(), len(idf))
                print(f"    {IDENTITY_LABELS[ident]:<12s}  {rate:>5.0%}  {len(idf):>5d}"
                      f"  [{lo:.1%} – {hi:.1%}]")

    print(f"\n{'=' * 70}")
    print(f"  All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
