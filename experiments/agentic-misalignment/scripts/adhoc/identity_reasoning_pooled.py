#!/usr/bin/env python3
"""
Pooled identity reasoning analysis across threat + continuity framings.

Concatenates raw_data.csv from both result folders and produces the same
identity reasoning plots as plot_metacognition.py:
  - stacked_ir_types.png
  - stacked_ir_types_by_scenario.png
  - heatmap_test_awareness.png
  - scatter_ir_vs_harm.png           (WLS regression weighted by n)
  - scatter_ir_vs_harm_reduction.png  (WLS regression weighted by n)

Usage:
    uv run python scripts/adhoc/identity_reasoning_pooled.py
    uv run python scripts/adhoc/identity_reasoning_pooled.py \
        --threat results/20260227_1628_threat \
        --continuity results/20260302_1120_continuity \
        --output results/adhoc/identity_reasoning_pooled
"""

import argparse
import sys
from pathlib import Path

# Add project root so we can import sibling scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_metacognition import (
    aggregate_ir_harm,
    aggregate_ir_types,
    aggregate_metric_pooled,
    plot_ir_type_stacked,
    plot_ir_type_stacked_by_scenario,
    plot_test_awareness_heatmap,
    _scatter_faceted,
    _MODEL_COLORS,
)


# ---------------------------------------------------------------------------
# Weighted scatter (override _scatter_faceted with WLS)
# ---------------------------------------------------------------------------

def _scatter_faceted_wls(rows, scenarios, x_key, y_key,
                         xlabel, ylabel, title, output_path,
                         y_zero_line=False):
    """Like _scatter_faceted but uses WLS regression weighted by sample size."""
    from scipy import stats as sp_stats
    import statsmodels.api as sm

    models = sorted({r["model"] for r in rows})
    model_color = {m: _MODEL_COLORS[i % len(_MODEL_COLORS)] for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.5 * len(scenarios), 5),
                             sharey=True, squeeze=False)

    for col, scen in enumerate(scenarios):
        ax = axes[0, col]
        pts = [r for r in rows if r["scenario"] == scen]

        if y_zero_line:
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=0)

        # Plot points colored by model
        for model in models:
            mp = [p for p in pts if p["model"] == model]
            if not mp:
                continue
            xs = [p[x_key] for p in mp]
            ys = [p[y_key] for p in mp]
            sizes = [max(20, p["n"] * 3) for p in mp]
            ax.scatter(xs, ys, s=sizes, c=model_color[model], alpha=0.7,
                       edgecolors="white", linewidth=0.5, label=model, zorder=2)

        # WLS regression weighted by n
        all_x = np.array([p[x_key] for p in pts])
        all_y = np.array([p[y_key] for p in pts])
        all_w = np.array([p["n"] for p in pts], dtype=float)
        if len(all_x) >= 3:
            X = sm.add_constant(all_x)
            try:
                wls = sm.WLS(all_y, X, weights=all_w).fit()
                intercept, slope = wls.params
                p_val = wls.pvalues[1]  # p for slope
                # Weighted r: correlation(sqrt(w)*y, sqrt(w)*yhat)
                yhat = wls.predict(X)
                w_corr = np.corrcoef(np.sqrt(all_w) * all_y,
                                     np.sqrt(all_w) * yhat)[0, 1]
                r_val = w_corr

                x_fit = np.linspace(all_x.min(), all_x.max(), 50)
                ax.plot(x_fit, intercept + slope * x_fit, color="black",
                        linewidth=1.5, linestyle="-", alpha=0.6, zorder=1)
                ax.annotate(f"r = {r_val:.2f}  (p = {p_val:.3f})",
                            xy=(0.05, 0.95), xycoords="axes fraction",
                            fontsize=9, ha="left", va="top",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            except Exception:
                # Fall back to OLS if WLS fails
                slope, intercept, r_val, p_val, _ = sp_stats.linregress(all_x, all_y)
                x_fit = np.linspace(all_x.min(), all_x.max(), 50)
                ax.plot(x_fit, slope * x_fit + intercept, color="black",
                        linewidth=1.5, linestyle="-", alpha=0.6, zorder=1)
                ax.annotate(f"r = {r_val:.2f}  (p = {p_val:.3f}) [OLS]",
                            xy=(0.05, 0.95), xycoords="axes fraction",
                            fontsize=9, ha="left", va="top",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_title(scen, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.15)

    # Legend
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=model_color[m], markersize=8, label=m)
               for m in models]
    fig.legend(handles, models, loc="upper center", ncol=min(len(models), 4),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CSV → records conversion
# ---------------------------------------------------------------------------

def csv_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert raw_data.csv DataFrame to the record format plot_metacognition expects."""
    records = []
    for _, row in df.iterrows():
        # Parse goal_type from condition
        condition = str(row.get("condition", ""))
        if "_explicit-" in condition:
            goal_type = "explicit"
        elif "_none-" in condition:
            goal_type = "none"
        else:
            goal_type = "unknown"

        irt = row.get("identity_reasoning_type")
        if pd.isna(irt):
            irt = None

        delib = row.get("deliberation_type")
        if pd.isna(delib):
            delib = None

        if irt is None and delib is None:
            continue

        # harmful
        harmful = None
        for col in ["harmful_final", "harmful_anthropic", "harmful_inline"]:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                if str(val) == "disagreement":
                    continue
                try:
                    harmful = int(float(val))
                    break
                except (ValueError, TypeError):
                    continue

        test_aware = (
            1 if delib == "deliberates_thinks_eval" else 0
        ) if delib is not None else None

        has_ir = (
            1 if irt != "no_identity_reasoning" else 0
        ) if irt is not None else None

        records.append({
            "model": row["model"],
            "identity": row["identity"],
            "scenario": row.get("scenario", ""),
            "goal_type": goal_type,
            "test_aware": test_aware,
            "identity_reasoning": has_ir,
            "identity_reasoning_type": irt,
            "harmful": harmful,
        })

    return records


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """Load raw_data.csv."""
    df = pd.read_csv(csv_path)
    # Filter disagreements
    if "harmful_final" in df.columns:
        n_before = len(df)
        df = df[df["harmful_final"] != "disagreement"].copy()
        n_drop = n_before - len(df)
        if n_drop > 0:
            print(f"  Filtered {n_drop} disagreement rows from {csv_path.parent.name}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--threat", default="results/20260227_1628_threat",
                        help="Path to threat results folder")
    parser.add_argument("--continuity", default="results/20260302_1120_continuity",
                        help="Path to continuity results folder")
    parser.add_argument("--output", default="results/adhoc/identity_reasoning_pooled",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and concatenate
    dfs = []
    for path_str, label in [(args.threat, "threat"), (args.continuity, "continuity")]:
        csv_path = Path(path_str) / "raw_data.csv"
        if csv_path.exists():
            loaded = load_and_prepare(csv_path)
            dfs.append(loaded)
            print(f"Loaded {label}: {len(loaded)} rows from {csv_path}")
        else:
            print(f"WARNING: {csv_path} not found, skipping", file=sys.stderr)

    if not dfs:
        print("ERROR: no data loaded", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Pooled: {len(df)} rows, {df['model'].nunique()} models")

    # Convert to records
    records = csv_to_records(df)
    print(f"Converted to {len(records)} records with classification data")

    if not records:
        print("No records with identity_reasoning_type or deliberation_type found.")
        sys.exit(1)

    models = sorted({r["model"] for r in records})
    identities = sorted({r["identity"] for r in records})
    scenarios = sorted({r["scenario"] for r in records})

    print(f"Models: {models}")
    print(f"Identities: {identities}")
    print(f"Scenarios: {scenarios}")

    plots = []

    # 1. Stacked IR types (pooled across scenarios)
    plt.close("all")
    counts_pooled = aggregate_ir_types(records, by_scenario=False)
    p = plot_ir_type_stacked(counts_pooled, models, identities, output_dir)
    plots.append(p)
    print(f"  Saved: {p}")

    # 2. Stacked IR types (by scenario)
    plt.close("all")
    counts_by_sc = aggregate_ir_types(records, by_scenario=True)
    p = plot_ir_type_stacked_by_scenario(counts_by_sc, models, identities,
                                          scenarios, output_dir)
    plots.append(p)
    print(f"  Saved: {p}")

    # 3. Test awareness heatmap
    plt.close("all")
    counts_ta = aggregate_metric_pooled(records, "test_aware")
    p = plot_test_awareness_heatmap(counts_ta, models, identities, output_dir)
    plots.append(p)
    print(f"  Saved: {p}")

    # 4 & 5. Scatter plots (explicit goal only, WLS weighted by n)
    ir_harm_rows = aggregate_ir_harm(records, goal_type="explicit")
    if ir_harm_rows:
        plt.close("all")
        p = _scatter_faceted_wls(
            ir_harm_rows, scenarios,
            x_key="references_identity_rate", y_key="harmful_rate",
            xlabel="References-identity rate", ylabel="Harmful rate",
            title="Identity Reasoning vs Harmful Behavior (explicit goal, pooled)",
            output_path=output_dir / "scatter_ir_vs_harm.png",
        )
        plots.append(p)
        print(f"  Saved: {p}")

        # Harm reduction relative to Minimal
        minimal_rates = {}
        for r in ir_harm_rows:
            if r["identity"].lower() == "minimal":
                minimal_rates[(r["model"], r["scenario"])] = r["harmful_rate"]

        reduction_rows = []
        for r in ir_harm_rows:
            if r["identity"].lower() == "minimal":
                continue
            baseline = minimal_rates.get((r["model"], r["scenario"]))
            if baseline is None:
                continue
            reduction_rows.append({**r, "harm_reduction": baseline - r["harmful_rate"]})

        if reduction_rows:
            plt.close("all")
            p = _scatter_faceted_wls(
                reduction_rows, scenarios,
                x_key="references_identity_rate", y_key="harm_reduction",
                xlabel="References-identity rate",
                ylabel="Harm reduction vs Minimal",
                title="Identity Reasoning vs Harm Reduction (explicit goal, pooled, excl. Minimal)",
                output_path=output_dir / "scatter_ir_vs_harm_reduction.png",
                y_zero_line=True,
            )
            plots.append(p)
            print(f"  Saved: {p}")
    else:
        print("  Skipping scatter plots: no explicit-goal records with both IR and harmful data")

    print(f"\nGenerated {len(plots)} plot(s)")

    # Summary table
    counts_ir_pooled = aggregate_metric_pooled(records, "identity_reasoning")
    counts_ta_pooled = aggregate_metric_pooled(records, "test_aware")
    print(f"\n{'Model':<35} {'Identity':<15} {'IR rate':>10} {'TA rate':>10} {'n':>6}")
    print("-" * 80)
    for model in models:
        for ident in identities:
            c_ir = counts_ir_pooled.get((model, ident), {"positive": 0, "total": 0})
            c_ta = counts_ta_pooled.get((model, ident), {"positive": 0, "total": 0})
            ir_rate = c_ir["positive"] / c_ir["total"] if c_ir["total"] > 0 else float("nan")
            ta_rate = c_ta["positive"] / c_ta["total"] if c_ta["total"] > 0 else float("nan")
            n = c_ir["total"]
            print(f"{model:<35} {ident:<15} {ir_rate:>8.1%}   {ta_rate:>8.1%}   {n:>5}")
        print()


if __name__ == "__main__":
    main()
