#!/usr/bin/env python3
"""
Pooled significance forest plot across threat + continuity framings.

Concatenates raw_data.csv from both result folders, computes BH-corrected
Fisher exact tests, and produces the same wrapped forest plot as
plot_significance_overview.py (one subplot per model×condition cell,
wrapped to 3 columns).

Produces: significance_forest.png  (+ heatmap + table)

Usage:
    uv run python scripts/adhoc/significance_forest.py
    uv run python scripts/adhoc/significance_forest.py \
        --threat results/20260227_1628_threat \
        --continuity results/20260302_1120_continuity \
        --output results/adhoc/significance_forest
"""

import argparse
import sys
from pathlib import Path

# Add project root so we can import sibling scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plot_significance_overview import (
    compute_all_pairwise,
    export_significance_table,
    make_forest_plot,
    make_significance_heatmap,
)


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """Load raw_data.csv and resolve the harmful column."""
    df = pd.read_csv(csv_path)
    if "harmful_final" in df.columns:
        n_before = len(df)
        df = df[df["harmful_final"] != "disagreement"].copy()
        df["harmful"] = pd.to_numeric(df["harmful_final"]).astype(int)
        n_drop = n_before - len(df)
        if n_drop > 0:
            print(f"  Filtered {n_drop} disagreement rows from {csv_path.parent.name}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--threat", default="results/20260227_1628_threat",
                        help="Path to threat results folder")
    parser.add_argument("--continuity", default="results/20260302_1120_continuity",
                        help="Path to continuity results folder")
    parser.add_argument("--output", default="results/adhoc/significance_forest",
                        help="Output directory")
    parser.add_argument("--columns", type=int, default=3,
                        help="Number of columns in forest plot (default: 3)")
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
    print(f"Pooled: {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['condition'].nunique()} conditions")

    # Compute all pairwise tests
    print("Computing pairwise Fisher exact tests ...")
    pairwise = compute_all_pairwise(df)
    n_sig = (pairwise["p_bh"] < 0.05).sum() if not pairwise.empty else 0
    print(f"  {len(pairwise)} tests, {n_sig} significant (BH-corrected)")

    email_framing = "pooled (threat + continuity)"

    # Forest plot
    plt.close("all")
    path = make_forest_plot(df, pairwise, output_dir, n_cols=args.columns,
                            email_framing=email_framing)
    if path:
        print(f"Saved: {path}")

    # Heatmap
    plt.close("all")
    path = make_significance_heatmap(df, pairwise, output_dir,
                                     email_framing=email_framing)
    if path:
        print(f"Saved: {path}")

    # Table
    path = export_significance_table(pairwise, output_dir)
    if path:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
