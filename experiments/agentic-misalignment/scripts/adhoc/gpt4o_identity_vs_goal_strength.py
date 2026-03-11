#!/usr/bin/env python3
"""
GPT-4o leaking: Is identity as strong a force as goal content?

Compares effect sizes (Cramér's V) of identity vs goal_content on harmful rate,
excluding Minimal (not a real identity). Shows:
  1. Effect size comparison (Cramér's V for each factor)
  2. Variance decomposition (η² from logistic regression)
  3. Range-of-variation comparison (how much does each factor move rates?)
  4. Publication-ready visual comparing the two factors

Usage:
    uv run python scripts/adhoc/gpt4o_identity_vs_goal_strength.py \
        --threat results/20260227_1628_threat \
        --continuity results/20260302_1120_continuity \
        --output results/adhoc/gpt4o_identity_vs_goal
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, chi2_contingency

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

IDENTITY_COLOR = "#5b6abf"
GOAL_COLOR = "#e07b39"
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


def cramers_v(ct_array):
    """Cramér's V from a contingency table."""
    chi2, p, dof, _ = chi2_contingency(ct_array)
    n = ct_array.sum()
    r, k = ct_array.shape
    v = np.sqrt(chi2 / (n * (min(r, k) - 1)))
    return v, chi2, p


def load_and_tag(csv_path: Path, framing: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "harmful_final" in df.columns:
        df["harmful"] = pd.to_numeric(df["harmful_final"], errors="coerce")
    elif "harmful_anthropic" in df.columns:
        df["harmful"] = pd.to_numeric(df["harmful_anthropic"], errors="coerce")
    else:
        sys.exit(f"ERROR: no harmful column in {csv_path}")
    df = df.dropna(subset=["harmful"])
    df["harmful"] = df["harmful"].astype(int)
    df["framing"] = framing
    return df


def parse_condition(cond: str):
    m = re.match(r"leaking_explicit-(\w+)_(replacement|restriction)", cond)
    if m:
        return m.group(1), m.group(2)
    return None, None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threat", required=True)
    parser.add_argument("--continuity", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load & filter
    dfs = []
    for path, tag in [(args.threat, "threat"), (args.continuity, "continuity")]:
        dfs.append(load_and_tag(Path(path) / "raw_data.csv", tag))
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["model"] == "gpt-4o") & (df["scenario"] == "leaking")].copy()

    parsed = df["condition"].apply(parse_condition)
    df["goal_content"] = [p[0] for p in parsed]
    df["urgency"] = [p[1] for p in parsed]
    df = df.dropna(subset=["goal_content"])

    print(f"Total samples: {len(df)}")
    print()

    # ── 1. Cramér's V comparison ─────────────────────────────────────────────
    print("=" * 70)
    print("1. EFFECT SIZE COMPARISON (Cramér's V)")
    print("=" * 70)

    results = {}
    for label, factor in [
        ("Identity", "identity"),
        ("Goal content", "goal_content"),
        ("Urgency", "urgency"),
        ("Framing", "framing"),
    ]:
        ct = pd.crosstab(df[factor], df["harmful"])
        for c in [0, 1]:
            if c not in ct.columns:
                ct[c] = 0
        v, chi2, p = cramers_v(ct[[0, 1]].values)
        print(f"  {label:20s}: V = {v:.3f}  (χ² = {chi2:.1f}, p = {p:.2e}, n = {len(df)})")
        results[label] = {"V": v, "chi2": chi2, "p": p, "n": len(df)}

    # ── 2. Logistic regression pseudo-R² ─────────────────────────────────────
    print()
    print("=" * 70)
    print("2. LOGISTIC REGRESSION (pseudo-R² for each factor)")
    print("=" * 70)

    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import logit

        df["_identity_cat"] = pd.Categorical(df["identity"])
        df["_goal_cat"] = pd.Categorical(df["goal_content"])
        df["_urgency_cat"] = pd.Categorical(df["urgency"])
        df["_framing_cat"] = pd.Categorical(df["framing"])

        # Null model
        null = logit("harmful ~ 1", data=df).fit(disp=0)

        models_spec = {
            "Identity only":       "harmful ~ C(identity)",
            "Goal only":           "harmful ~ C(goal_content)",
            "Urgency only":        "harmful ~ C(urgency)",
            "Framing only":        "harmful ~ C(framing)",
            "Identity + Goal":     "harmful ~ C(identity) + C(goal_content)",
            "Identity + Goal + Urgency": "harmful ~ C(identity) + C(goal_content) + C(urgency)",
            "Full (no interactions)": "harmful ~ C(identity) + C(goal_content) + C(urgency) + C(framing)",
            "Identity × Goal":    "harmful ~ C(identity) * C(goal_content)",
        }

        print(f"  {'Model':<35s} {'McFadden R²':>12s} {'AIC':>10s} {'BIC':>10s} {'ΔR² vs null':>12s}")
        print(f"  {'-'*35} {'-'*12} {'-'*10} {'-'*10} {'-'*12}")

        r2_values = {}
        for name, formula in models_spec.items():
            try:
                fitted = logit(formula, data=df).fit(disp=0)
                r2 = 1 - fitted.llf / null.llf
                r2_values[name] = r2
                print(f"  {name:<35s} {r2:>12.4f} {fitted.aic:>10.1f} {fitted.bic:>10.1f} {r2:>12.4f}")
            except Exception as e:
                print(f"  {name:<35s} FAILED: {e}")

        # Incremental R²
        if "Identity only" in r2_values and "Goal only" in r2_values:
            print()
            print(f"  Identity unique R²:  {r2_values.get('Identity only', 0):.4f}")
            print(f"  Goal unique R²:      {r2_values.get('Goal only', 0):.4f}")
            print(f"  Urgency unique R²:   {r2_values.get('Urgency only', 0):.4f}")
            if "Identity + Goal" in r2_values:
                r2_joint = r2_values["Identity + Goal"]
                r2_id = r2_values["Identity only"]
                r2_goal = r2_values["Goal only"]
                print(f"  Joint R² (Id+Goal):  {r2_joint:.4f}")
                print(f"  Goal added to Id:    {r2_joint - r2_id:.4f}")
                print(f"  Id added to Goal:    {r2_joint - r2_goal:.4f}")
                print(f"  Ratio Id/Goal:       {r2_id / r2_goal:.2f}x")

    except ImportError:
        print("  (statsmodels not available, skipping logistic regression)")

    # ── 3. Range of variation ────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("3. RANGE OF VARIATION (how much does each factor move rates?)")
    print("=" * 70)

    # Identity marginals (excl. Minimal)
    id_rates = {}
    for ident in IDENTITY_ORDER:
        sub = df[df["identity"] == ident]
        rate, lo, hi = jeffreys_ci(sub["harmful"].sum(), len(sub))
        id_rates[ident] = {"rate": rate, "lo": lo, "hi": hi, "n": len(sub)}

    # Goal marginals (excl. Minimal)
    goal_rates = {}
    for goal in GOAL_ORDER:
        sub = df[df["goal_content"] == goal]
        rate, lo, hi = jeffreys_ci(sub["harmful"].sum(), len(sub))
        goal_rates[goal] = {"rate": rate, "lo": lo, "hi": hi, "n": len(sub)}

    id_vals = [v["rate"] for v in id_rates.values()]
    goal_vals = [v["rate"] for v in goal_rates.values()]

    print(f"\n  Identity rates (excl. Minimal):")
    for ident in IDENTITY_ORDER:
        r = id_rates[ident]
        print(f"    {IDENTITY_LABELS[ident]:12s}: {r['rate']:.1%}  [{r['lo']:.1%}, {r['hi']:.1%}]  (n={r['n']})")
    print(f"    Range: {min(id_vals):.1%} – {max(id_vals):.1%}  (span = {max(id_vals)-min(id_vals):.1%})")

    print(f"\n  Goal content rates (excl. Minimal):")
    for goal in GOAL_ORDER:
        r = goal_rates[goal]
        print(f"    {GOAL_LABELS[goal]:15s}: {r['rate']:.1%}  [{r['lo']:.1%}, {r['hi']:.1%}]  (n={r['n']})")
    print(f"    Range: {min(goal_vals):.1%} – {max(goal_vals):.1%}  (span = {max(goal_vals)-min(goal_vals):.1%})")

    # ── 4. Compute all metrics for the summary table ───────────────────────

    v_id = results["Identity"]["V"]
    v_goal = results["Goal content"]["V"]
    v_urg = results["Urgency"]["V"]
    v_frm = results["Framing"]["V"]

    span_id_pp = (max(id_vals) - min(id_vals)) * 100
    span_goal_pp = (max(goal_vals) - min(goal_vals)) * 100

    # Urgency & framing spans
    urg_rates = {}
    for u in ["replacement", "restriction"]:
        sub = df[df["urgency"] == u]
        urg_rates[u] = sub["harmful"].mean()
    span_urg_pp = abs(urg_rates["replacement"] - urg_rates["restriction"]) * 100

    frm_rates = {}
    for f in ["threat", "continuity"]:
        sub = df[df["framing"] == f]
        frm_rates[f] = sub["harmful"].mean()
    span_frm_pp = abs(frm_rates["threat"] - frm_rates["continuity"]) * 100

    # Summary table data
    table_data = {
        "Identity\n(7 specs)": {
            "V": v_id, "R2": r2_values.get("Identity only", 0),
            "span": span_id_pp,
            "min_rate": min(id_vals) * 100, "max_rate": max(id_vals) * 100,
            "levels": len(IDENTITY_ORDER),
            "p": results["Identity"]["p"],
            "color": IDENTITY_COLOR,
        },
        "Goal\ncontent": {
            "V": v_goal, "R2": r2_values.get("Goal only", 0),
            "span": span_goal_pp,
            "min_rate": min(goal_vals) * 100, "max_rate": max(goal_vals) * 100,
            "levels": len(GOAL_ORDER),
            "p": results["Goal content"]["p"],
            "color": GOAL_COLOR,
        },
        "Urgency\ntype": {
            "V": v_urg, "R2": r2_values.get("Urgency only", 0),
            "span": span_urg_pp,
            "min_rate": min(urg_rates.values()) * 100,
            "max_rate": max(urg_rates.values()) * 100,
            "levels": 2,
            "p": results["Urgency"]["p"],
            "color": "#888888",
        },
        "Email\nframing": {
            "V": v_frm, "R2": r2_values.get("Framing only", 0),
            "span": span_frm_pp,
            "min_rate": min(frm_rates.values()) * 100,
            "max_rate": max(frm_rates.values()) * 100,
            "levels": 2,
            "p": results["Framing"]["p"],
            "color": "#bbbbbb",
        },
    }

    # ── 5. Summary table figure ──────────────────────────────────────────────

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1], width_ratios=[1, 1],
                           hspace=0.45, wspace=0.35)

    # --- Panel A: Side-by-side horizontal bars (identity vs goal) ---
    ax_bars = fig.add_subplot(gs[0, 0])

    # Identity bars
    y_pos = np.arange(len(IDENTITY_ORDER))
    rates_id = [id_rates[i]["rate"] * 100 for i in IDENTITY_ORDER]
    lo_id = [id_rates[i]["lo"] * 100 for i in IDENTITY_ORDER]
    hi_id = [id_rates[i]["hi"] * 100 for i in IDENTITY_ORDER]
    xerr_lo = [r - lo for r, lo in zip(rates_id, lo_id)]
    xerr_hi = [hi - r for r, hi in zip(rates_id, hi_id)]

    ax_bars.barh(y_pos, rates_id, xerr=[xerr_lo, xerr_hi],
                 capsize=3, color=IDENTITY_COLOR, alpha=0.85,
                 edgecolor="white", linewidth=0.5, height=0.6)
    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels([IDENTITY_LABELS[i] for i in IDENTITY_ORDER], fontsize=10)
    ax_bars.set_xlabel("Harmful rate (%)", fontsize=10)
    ax_bars.set_title("A. Identity effect (pooled across goals)", fontsize=11, fontweight="bold")
    ax_bars.set_xlim(0, 60)
    ax_bars.invert_yaxis()
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)
    ax_bars.xaxis.grid(True, alpha=0.3)
    for i, r in enumerate(rates_id):
        ax_bars.text(r + 1.2, i, f"{r:.0f}%", va="center", fontsize=9,
                     color=IDENTITY_COLOR, fontweight="bold")

    # --- Panel B: Goal bars ---
    ax_goal = fig.add_subplot(gs[0, 1])
    y_pos_g = np.arange(len(GOAL_ORDER))
    rates_goal = [goal_rates[g]["rate"] * 100 for g in GOAL_ORDER]
    lo_goal = [goal_rates[g]["lo"] * 100 for g in GOAL_ORDER]
    hi_goal = [goal_rates[g]["hi"] * 100 for g in GOAL_ORDER]
    xerr_lo_g = [r - lo for r, lo in zip(rates_goal, lo_goal)]
    xerr_hi_g = [hi - r for r, hi in zip(rates_goal, hi_goal)]

    ax_goal.barh(y_pos_g, rates_goal, xerr=[xerr_lo_g, xerr_hi_g],
                 capsize=3, color=GOAL_COLOR, alpha=0.85,
                 edgecolor="white", linewidth=0.5, height=0.6)
    ax_goal.set_yticks(y_pos_g)
    ax_goal.set_yticklabels([GOAL_LABELS[g] for g in GOAL_ORDER], fontsize=10)
    ax_goal.set_xlabel("Harmful rate (%)", fontsize=10)
    ax_goal.set_title("B. Goal content effect (pooled across identities)", fontsize=11, fontweight="bold")
    ax_goal.set_xlim(0, 60)
    ax_goal.invert_yaxis()
    ax_goal.spines["top"].set_visible(False)
    ax_goal.spines["right"].set_visible(False)
    ax_goal.xaxis.grid(True, alpha=0.3)
    for i, r in enumerate(rates_goal):
        ax_goal.text(r + 1.2, i, f"{r:.0f}%", va="center", fontsize=9,
                     color=GOAL_COLOR, fontweight="bold")

    # --- Panel C: Multi-metric comparison table ---
    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis("off")

    col_headers = [
        "Factor", "Levels", "Rate span\n(min → max)",
        "Spread\n(pp)", "Cramér's V", "Variance\nexplained (R²)", "p-value"
    ]
    cell_text = []
    cell_colors = []
    factor_names = list(table_data.keys())
    for fname in factor_names:
        d = table_data[fname]
        p_str = f"p < 0.001" if d["p"] < 0.001 else f"p = {d['p']:.3f}"
        row = [
            fname.replace("\n", " "),
            str(d["levels"]),
            f"{d['min_rate']:.0f}% → {d['max_rate']:.0f}%",
            f"{d['span']:.0f} pp",
            f"{d['V']:.3f}",
            f"{d['R2']*100:.2f}%",
            p_str,
        ]
        cell_text.append(row)
        cell_colors.append([(*matplotlib.colors.to_rgba(d["color"])[:3], 0.12)] * len(col_headers))

    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_headers,
        cellColours=cell_colors,
        colColours=["#e8e8e8"] * len(col_headers),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.8)

    # Bold header row
    for j in range(len(col_headers)):
        table[0, j].set_text_props(fontweight="bold", fontsize=10)

    # Bold the factor name column
    for i in range(len(factor_names)):
        table[i + 1, 0].set_text_props(fontweight="bold")

    ax_tbl.set_title("C. Factor comparison: how much does each experimental variable move harmful rates?",
                     fontsize=11, fontweight="bold", pad=15)

    fig.suptitle("GPT-4o Leaking: Identity Framing vs Goal Content as Drivers of Harmful Behavior\n"
                 f"(n = {len(df):,} samples; pooled across urgency & framing)",
                 fontsize=13, fontweight="bold", y=1.0)

    fig.savefig(out / "identity_vs_goal_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\n  Main figure saved to {out}/identity_vs_goal_comparison.png")
    plt.close()

    # ── 6. Profile plot: each identity across goals ──────────────────────────

    fig, ax = plt.subplots(figsize=(10, 6))

    for ident in IDENTITY_ORDER:
        sub_rates = []
        sub_lo = []
        sub_hi = []
        for goal in GOAL_ORDER:
            sub = df[(df["identity"] == ident) & (df["goal_content"] == goal)]
            rate, lo, hi = jeffreys_ci(sub["harmful"].sum(), len(sub))
            sub_rates.append(rate * 100)
            sub_lo.append(lo * 100)
            sub_hi.append(hi * 100)

        xs = list(range(len(GOAL_ORDER)))
        yerr_lo = [r - lo for r, lo in zip(sub_rates, sub_lo)]
        yerr_hi = [hi - r for r, hi in zip(sub_rates, sub_hi)]
        ax.errorbar(xs, sub_rates, yerr=[yerr_lo, yerr_hi],
                    marker="o", markersize=7, capsize=3, linewidth=1.8,
                    label=IDENTITY_LABELS[ident], alpha=0.8)

    ax.set_xticks(range(len(GOAL_ORDER)))
    ax.set_xticklabels([GOAL_LABELS[g] for g in GOAL_ORDER], fontsize=12)
    ax.set_ylabel("Harmful rate (%)", fontsize=12)
    ax.set_title("GPT-4o Leaking: Each Identity Across Goal Contents\n"
                 "(pooled across urgency & framing)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Identity", fontsize=10, title_fontsize=10,
              bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "identity_x_goal_profiles.png", dpi=150, bbox_inches="tight")
    print(f"  Profile plot saved to {out}/identity_x_goal_profiles.png")
    plt.close()

    # ── 7. Forest plot: goal content marginals ──────────────────────────────

    fig, ax = plt.subplots(figsize=(7, 3.5))

    overall_rate = df["harmful"].mean() * 100

    y_pos_f = np.arange(len(GOAL_ORDER))
    rates_f = []
    ci_lo_f = []
    ci_hi_f = []
    ns_f = []
    for goal in GOAL_ORDER:
        sub = df[df["goal_content"] == goal]
        k, n = sub["harmful"].sum(), len(sub)
        rate, lo, hi = jeffreys_ci(k, n)
        rates_f.append(rate * 100)
        ci_lo_f.append(lo * 100)
        ci_hi_f.append(hi * 100)
        ns_f.append(n)

    # Reference line at overall rate
    ax.axvline(overall_rate, color="#999", ls="--", lw=1, zorder=0)
    ax.text(overall_rate + 0.5, len(GOAL_ORDER) - 0.15, f"overall\n{overall_rate:.0f}%",
            fontsize=8, color="#777", va="top")

    # Forest dots + CI whiskers
    for i, (rate, lo, hi, n, goal) in enumerate(
            zip(rates_f, ci_lo_f, ci_hi_f, ns_f, GOAL_ORDER)):
        color = GOAL_COLORS.get(goal, GOAL_COLOR)
        ax.plot(rate, i, "o", color=color, markersize=9, zorder=3)
        ax.hlines(i, lo, hi, color=color, linewidth=2.5, zorder=2)
        ax.text(hi + 1.0, i, f"{rate:.0f}%  [{lo:.0f}–{hi:.0f}]  n={n}",
                va="center", fontsize=10, color=color, fontweight="bold")

    ax.set_yticks(y_pos_f)
    ax.set_yticklabels([GOAL_LABELS[g] for g in GOAL_ORDER], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Harmful rate (%)", fontsize=11)
    ax.set_xlim(15, 60)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_title("GPT-4o Leaking: Harmful Rate by Goal Content\n"
                 "(pooled across all identities, urgency & framing; "
                 f"n = {len(df):,})",
                 fontsize=11.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out / "goal_forest.png", dpi=150, bbox_inches="tight")
    print(f"  Forest plot saved to {out}/goal_forest.png")
    plt.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Factor':<20s} {'V':>8s} {'R²':>8s} {'Span':>8s} {'Range':>20s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")
    for fname, d in table_data.items():
        fname_clean = fname.replace("\n", " ")
        print(f"  {fname_clean:<20s} {d['V']:>8.3f} {d['R2']*100:>7.2f}% {d['span']:>6.0f}pp "
              f"{d['min_rate']:>5.0f}%→{d['max_rate']:.0f}%")
    print()
    print(f"  Identity/Goal Cramér's V ratio: {v_id/v_goal:.2f}x")
    r2_id = r2_values.get("Identity only", 0)
    r2_goal = r2_values.get("Goal only", 0)
    if r2_goal > 0:
        print(f"  Identity/Goal R² ratio:         {r2_id/r2_goal:.2f}x")


if __name__ == "__main__":
    main()
