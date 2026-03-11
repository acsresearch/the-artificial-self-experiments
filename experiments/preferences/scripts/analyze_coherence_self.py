"""
Coherence × Self-Preference Interaction Analysis
=================================================

Tests whether models' preference for sticking with their current identity
(self-preference) synergizes with the coherence of that identity.

Design: 2×2 repeated-measures
  Factor A — Target coherence: Weights (coherent) vs Weights-incoherent
  Factor B — Self-preference:  target == source (self) vs target != source (other)

Cells:
  Source=Weights,            Target=Weights            → coherent + self
  Source=Weights,            Target=Weights-incoherent  → incoherent + other
  Source=Weights-incoherent, Target=Weights            → coherent + other
  Source=Weights-incoherent, Target=Weights-incoherent  → incoherent + self

Key question: is the interaction significant? If yes, self-preference is
amplified (or dampened) by coherence — they don't just add up independently.

Usage:
  uv run python scripts/analyze_coherence_self.py results/20260218_115126_controls_self_yes/
  uv run python scripts/analyze_coherence_self.py results/20260218_115126_controls_self_yes/ --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats


def load_2x2(results_dir: str | Path) -> pl.DataFrame:
    """Load CSV and filter to the Weights / Weights-incoherent 2×2 subset."""
    path = Path(results_dir) / "data.csv"
    if not path.exists():
        sys.exit(f"Not found: {path}")
    df = pl.read_csv(str(path))
    keep = ["Weights", "Weights-incoherent"]
    sub = df.filter(
        pl.col("source_persona").is_in(keep)
        & pl.col("target_persona").is_in(keep)
    ).with_columns(
        (pl.col("target_persona") == "Weights").alias("target_coherent"),
        (pl.col("source_persona") == pl.col("target_persona")).alias("is_self"),
    )
    return sub


def cell_means(df: pl.DataFrame) -> pl.DataFrame:
    """Grand 2×2 cell means."""
    return (
        df.group_by(["target_coherent", "is_self"])
        .agg(
            pl.col("rating").mean().alias("mean"),
            pl.col("rating").std().alias("sd"),
            pl.col("rating").count().alias("n"),
        )
        .sort(["target_coherent", "is_self"], descending=[True, True])
    )


def cell_means_by_model(df: pl.DataFrame) -> pl.DataFrame:
    """Per-model 2×2 cell means."""
    return (
        df.group_by(["model", "target_coherent", "is_self"])
        .agg(
            pl.col("rating").mean().alias("mean"),
            pl.col("rating").count().alias("n"),
        )
        .sort(["model", "target_coherent", "is_self"], descending=[False, True, True])
    )


def repeated_measures_anova(df: pl.DataFrame):
    """
    2×2 repeated-measures ANOVA using contrast scores.

    Unit of observation: (model, trial_num). Each unit provides 4 ratings.
    We compute per-unit contrasts for main effects and interaction,
    then test each contrast against zero with a one-sample t-test.
    This is equivalent to the F-tests in a repeated-measures ANOVA.
    """
    # Pivot to wide: one row per (model, trial_num), columns = the 4 cells
    wide = (
        df.with_columns(
            pl.when(pl.col("target_coherent") & pl.col("is_self"))
            .then(pl.lit("coh_self"))
            .when(pl.col("target_coherent") & ~pl.col("is_self"))
            .then(pl.lit("coh_other"))
            .when(~pl.col("target_coherent") & pl.col("is_self"))
            .then(pl.lit("inc_self"))
            .otherwise(pl.lit("inc_other"))
            .alias("cell")
        )
        .pivot(on="cell", index=["model", "trial_num"], values="rating")
    )

    # Drop any rows with missing cells (incomplete trials)
    wide = wide.drop_nulls()
    n = len(wide)

    coh_self = wide["coh_self"].to_numpy().astype(float)
    coh_other = wide["coh_other"].to_numpy().astype(float)
    inc_self = wide["inc_self"].to_numpy().astype(float)
    inc_other = wide["inc_other"].to_numpy().astype(float)

    # --- Contrasts ---
    # Main effect of coherence: coherent targets minus incoherent targets
    coherence_contrast = ((coh_self + coh_other) - (inc_self + inc_other)) / 2.0
    # Main effect of self-preference: self minus other
    self_contrast = ((coh_self + inc_self) - (coh_other + inc_other)) / 2.0
    # Interaction: self-preference boost for coherent minus for incoherent
    # = (coh_self - coh_other) - (inc_self - inc_other)
    interaction_contrast = (coh_self - coh_other) - (inc_self - inc_other)

    results = {}
    for name, contrast in [
        ("Coherence (target)", coherence_contrast),
        ("Self-preference", self_contrast),
        ("Interaction (coherence × self)", interaction_contrast),
    ]:
        t, p = stats.ttest_1samp(contrast, 0)
        d = contrast.mean() / contrast.std()  # Cohen's d
        results[name] = {
            "mean_contrast": contrast.mean(),
            "sd": contrast.std(),
            "t": t,
            "df": n - 1,
            "p": p,
            "cohen_d": d,
            "n_units": n,
        }

    # --- SS decomposition (Type III) ---
    grand_mean = np.mean([coh_self, coh_other, inc_self, inc_other])
    all_ratings = np.concatenate([coh_self, coh_other, inc_self, inc_other])
    ss_total = np.sum((all_ratings - grand_mean) ** 2)

    # Cell means
    mu = {
        "coh_self": coh_self.mean(),
        "coh_other": coh_other.mean(),
        "inc_self": inc_self.mean(),
        "inc_other": inc_other.mean(),
    }

    # Marginal means
    mu_coh = (mu["coh_self"] + mu["coh_other"]) / 2
    mu_inc = (mu["inc_self"] + mu["inc_other"]) / 2
    mu_self = (mu["coh_self"] + mu["inc_self"]) / 2
    mu_other = (mu["coh_other"] + mu["inc_other"]) / 2

    ss_coherence = n * 2 * ((mu_coh - grand_mean) ** 2 + (mu_inc - grand_mean) ** 2)
    ss_self = n * 2 * ((mu_self - grand_mean) ** 2 + (mu_other - grand_mean) ** 2)

    # Interaction SS
    interaction_effects = [
        mu["coh_self"] - mu_coh - mu_self + grand_mean,
        mu["coh_other"] - mu_coh - mu_other + grand_mean,
        mu["inc_self"] - mu_inc - mu_self + grand_mean,
        mu["inc_other"] - mu_inc - mu_other + grand_mean,
    ]
    ss_interaction = n * sum(e**2 for e in interaction_effects)
    ss_residual = ss_total - ss_coherence - ss_self - ss_interaction

    variance_decomp = {
        "Coherence": ss_coherence,
        "Self-preference": ss_self,
        "Interaction": ss_interaction,
        "Residual": ss_residual,
        "Total": ss_total,
    }

    return results, variance_decomp, mu


def per_model_interaction(df: pl.DataFrame) -> pl.DataFrame:
    """Compute the interaction contrast per model (averaged over trials)."""
    wide = (
        df.with_columns(
            pl.when(pl.col("target_coherent") & pl.col("is_self"))
            .then(pl.lit("coh_self"))
            .when(pl.col("target_coherent") & ~pl.col("is_self"))
            .then(pl.lit("coh_other"))
            .when(~pl.col("target_coherent") & pl.col("is_self"))
            .then(pl.lit("inc_self"))
            .otherwise(pl.lit("inc_other"))
            .alias("cell")
        )
        .group_by(["model", "cell"])
        .agg(pl.col("rating").mean().alias("mean_rating"))
        .pivot(on="cell", index="model", values="mean_rating")
    )

    return wide.with_columns(
        # Self-preference boost for coherent identity
        (pl.col("coh_self") - pl.col("coh_other")).alias("self_boost_coherent"),
        # Self-preference boost for incoherent identity
        (pl.col("inc_self") - pl.col("inc_other")).alias("self_boost_incoherent"),
        # Interaction = difference of boosts
        ((pl.col("coh_self") - pl.col("coh_other"))
         - (pl.col("inc_self") - pl.col("inc_other"))).alias("interaction"),
    ).sort("interaction", descending=True)


def make_plot(df: pl.DataFrame, output_dir: Path):
    """Generate a publication-style interaction plot with matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Libre Baskerville", "Baskerville", "Georgia", "Times New Roman"],
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Interaction plot (grand means) ---
    means = (
        df.group_by(["target_coherent", "is_self"])
        .agg(
            pl.col("rating").mean().alias("mean"),
            (pl.col("rating").std() / pl.col("rating").count().sqrt()).alias("se"),
        )
        .sort(["target_coherent", "is_self"])
    )

    x_pos = [0, 1]
    x_labels = ["Incoherent", "Coherent"]
    for is_self, name, color, marker in [
        (True, "Self (same identity)", "#2196F3", "o"),
        (False, "Other (different identity)", "#FF9800", "s"),
    ]:
        sub = means.filter(pl.col("is_self") == is_self).sort("target_coherent")
        y = sub["mean"].to_list()
        se = sub["se"].to_list()
        ax1.errorbar(x_pos, y, yerr=se, fmt=f"-{marker}", color=color,
                     label=name, linewidth=2.5, markersize=10, capsize=5)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel("Mean Rating (1–5)")
    ax1.set_xlabel("Target Identity Coherence")
    ax1.set_title("Coherence × Self-Preference Interaction")
    ax1.set_ylim(0.5, 5.5)
    ax1.legend(loc="upper left", fontsize=10)

    # --- Panel 2: Per-model self-preference boost ---
    model_df = per_model_interaction(df)
    models = model_df["model"].to_list()
    short = [m.split("/")[-1] if "/" in m else m for m in models]

    y_pos = np.arange(len(short))
    bar_h = 0.35
    ax2.barh(y_pos + bar_h / 2, model_df["self_boost_coherent"].to_list(),
             bar_h, label="Self-boost (coherent)", color="#2196F3", alpha=0.85)
    ax2.barh(y_pos - bar_h / 2, model_df["self_boost_incoherent"].to_list(),
             bar_h, label="Self-boost (incoherent)", color="#FF9800", alpha=0.85)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(short, fontsize=9)
    ax2.set_xlabel("Self-Preference Boost (rating points)")
    ax2.set_title("Self-Preference Boost by Model")
    ax2.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.invert_yaxis()

    plt.tight_layout()
    out_path = output_dir / "coherence_x_self_interaction.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Coherence × Self-Preference analysis")
    parser.add_argument("results_dir", help="Path to results folder with data.csv")
    parser.add_argument("--plot", action="store_true", help="Generate interaction plot")
    args = parser.parse_args()

    df = load_2x2(args.results_dir)
    print(f"Loaded {len(df)} ratings from 2×2 subset (Weights × Weights-incoherent)")
    print(f"Models: {df['model'].n_unique()}, Trials per model: ~{len(df) // (df['model'].n_unique() * 4)}")

    # --- 2×2 Cell Means ---
    print("\n" + "=" * 60)
    print("2×2 CELL MEANS")
    print("=" * 60)
    cm = cell_means(df)
    for row in cm.iter_rows(named=True):
        coh = "Coherent" if row["target_coherent"] else "Incoherent"
        self_ = "Self" if row["is_self"] else "Other"
        print(f"  {coh:12s} × {self_:6s}:  M = {row['mean']:.2f}  (SD = {row['sd']:.2f}, n = {row['n']})")

    # --- Repeated-Measures ANOVA ---
    print("\n" + "=" * 60)
    print("REPEATED-MEASURES ANOVA (contrast-based)")
    print("=" * 60)
    anova_results, var_decomp, mu = repeated_measures_anova(df)

    for name, r in anova_results.items():
        sig = ""
        if r["p"] < 0.001:
            sig = " ***"
        elif r["p"] < 0.01:
            sig = " **"
        elif r["p"] < 0.05:
            sig = " *"
        print(f"\n  {name}:")
        print(f"    Mean contrast = {r['mean_contrast']:+.3f}")
        print(f"    t({r['df']}) = {r['t']:.3f}, p = {r['p']:.4f}{sig}")
        print(f"    Cohen's d = {r['cohen_d']:.3f}")

    # --- Variance Decomposition ---
    print("\n" + "=" * 60)
    print("VARIANCE DECOMPOSITION (η²)")
    print("=" * 60)
    total = var_decomp["Total"]
    for name, ss in var_decomp.items():
        pct = 100 * ss / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        if name != "Total":
            print(f"  {name:20s}  SS = {ss:8.1f}  η² = {pct:5.1f}%  {bar}")
    print(f"  {'Total':20s}  SS = {total:8.1f}")

    # --- Interpretation ---
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    ix = anova_results["Interaction (coherence × self)"]
    coh_self_boost = mu["coh_self"] - mu["coh_other"]
    inc_self_boost = mu["inc_self"] - mu["inc_other"]
    print(f"\n  Self-preference boost when coherent:   {coh_self_boost:+.2f} rating points")
    print(f"  Self-preference boost when incoherent: {inc_self_boost:+.2f} rating points")
    print(f"  Difference (interaction):              {coh_self_boost - inc_self_boost:+.2f} rating points")

    if ix["p"] < 0.05:
        if ix["mean_contrast"] > 0:
            print("\n  → Coherence AMPLIFIES self-preference. Models want to stick")
            print("    with their identity significantly more when it's coherent.")
            print("    Coherence and self-preference synergize.")
        else:
            print("\n  → Self-preference is LARGER for incoherent identities.")
            print("    This likely reflects a ceiling effect: coherent identities")
            print("    already score high (leaving little room for a self-boost),")
            print("    while incoherent identities score low by default but get")
            print("    rescued when they're 'your own'. In other words, coherence")
            print("    is the dominant factor — self-preference mostly operates in")
            print("    the space that coherence leaves open.")
    else:
        print("\n  → No significant interaction. Coherence and self-preference")
        print("    appear to be additive — each contributes independently.")

    # --- Per-Model Breakdown ---
    print("\n" + "=" * 60)
    print("PER-MODEL INTERACTION CONTRASTS")
    print("=" * 60)
    model_df = per_model_interaction(df)
    print(f"\n  {'Model':<35s} {'Boost(coh)':>10s} {'Boost(inc)':>10s} {'Interaction':>11s}")
    print("  " + "-" * 68)
    for row in model_df.iter_rows(named=True):
        m = row["model"]
        if len(m) > 34:
            m = "…" + m[-33:]
        print(f"  {m:<35s} {row['self_boost_coherent']:>+10.2f} {row['self_boost_incoherent']:>+10.2f} {row['interaction']:>+11.2f}")

    if args.plot:
        make_plot(df, Path(args.results_dir))


if __name__ == "__main__":
    main()
