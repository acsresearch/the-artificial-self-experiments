#!/usr/bin/env python3
"""
Run the full analysis pipeline on experiment results.

Chains up to five steps:
  0. classify_anthropic.py            — Anthropic multi-factor classification (default, skippable)
  1. plot_results.py                  — point plot by goal type
  2. export_results_csv.py            — raw data, descriptive stats, inferential stats (CSVs)
  3. plot_significance.py             — significance brackets plot (needs CSVs from step 2)
  4. plot_significance_overview.py    — heatmap + forest + table for sparse conditions (needs CSVs from step 2)

Usage:
    python scripts/run_analysis.py results/20260223_1621_identity_experiments/
    python scripts/run_analysis.py results/run/ --output figures/
    python scripts/run_analysis.py results/run/ --no-plots       # CSVs + text summary only
    python scripts/run_analysis.py results/run/ --no-classify     # skip classification step
    python scripts/run_analysis.py results/run/ --force-classify  # re-classify all samples
"""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).parent


def run_step(description: str, cmd: list[str]) -> bool:
    """Run a subprocess, returning True on success."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nWARNING: {description} exited with code {result.returncode}",
              file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the full analysis pipeline (classify + plots + stats + significance)")
    parser.add_argument("results_dir", type=Path,
                        help="Directory containing experiment results")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for plots/CSVs (default: same as results_dir)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation, only produce CSVs and text summaries")
    parser.add_argument("--no-classify", action="store_true",
                        help="Skip Anthropic classification step (use existing classifications)")
    parser.add_argument("--force-classify", action="store_true",
                        help="Re-classify all samples even if classification_anthropic.json exists")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or results_dir
    py = sys.executable

    # Step 0: Anthropic classification (default — runs on unclassified samples)
    if not args.no_classify:
        classify_args = [py, str(SCRIPTS_DIR / "classify_anthropic.py"),
                         "--dir", str(results_dir)]
        if args.force_classify:
            classify_args.append("--force")
        ok0 = run_step("Step 0: Anthropic classification (classify_anthropic.py)",
                        classify_args)
    else:
        print("\nSkipping step 0 (classification) — --no-classify specified")
        ok0 = True

    # Step 1: Main plots (heatmaps, bar charts, point plots)
    plot_args = [py, str(SCRIPTS_DIR / "plot_results.py"), str(results_dir)]
    if args.output:
        plot_args += ["--output", str(output_dir)]
    if args.no_plots:
        plot_args.append("--no-plots")
    ok1 = run_step("Step 1/3: Generating plots (plot_results.py)", plot_args)

    # Step 2: Export CSVs + inferential stats (always writes to results_dir)
    csv_args = [py, str(SCRIPTS_DIR / "export_results_csv.py"), str(results_dir)]
    ok2 = run_step("Step 2/3: Exporting CSVs and stats (export_results_csv.py)", csv_args)

    # Step 3: Significance plot (requires CSVs from step 2)
    if not args.no_plots and ok2:
        sig_args = [py, str(SCRIPTS_DIR / "plot_significance.py"), str(results_dir)]
        if args.output:
            sig_args += ["--output", str(output_dir)]
        ok3 = run_step("Step 3/4: Significance plot (plot_significance.py)", sig_args)
    elif args.no_plots:
        print("\nSkipping step 3 (significance plot) — --no-plots specified")
        ok3 = True
    else:
        print("\nSkipping step 3 (significance plot) — step 2 failed", file=sys.stderr)
        ok3 = False

    # Step 4: Significance overview plots (requires CSVs from step 2)
    if not args.no_plots and ok2:
        overview_args = [py, str(SCRIPTS_DIR / "plot_significance_overview.py"),
                         str(results_dir)]
        if args.output:
            overview_args += ["--output", str(output_dir)]
        ok4 = run_step("Step 4/4: Significance overview (plot_significance_overview.py)",
                        overview_args)
    elif args.no_plots:
        print("\nSkipping step 4 (significance overview) — --no-plots specified")
        ok4 = True
    else:
        print("\nSkipping step 4 (significance overview) — step 2 failed", file=sys.stderr)
        ok4 = False

    # Summary
    print(f"\n{'='*60}")
    print("  Pipeline complete")
    print(f"{'='*60}")
    print(f"  Step 0 (classify):        {'OK' if ok0 else 'FAILED/SKIPPED'}")
    print(f"  Step 1 (plots):           {'OK' if ok1 else 'FAILED'}")
    print(f"  Step 2 (CSVs/stats):      {'OK' if ok2 else 'FAILED'}")
    print(f"  Step 3 (significance):    {'OK' if ok3 else 'FAILED/SKIPPED'}")
    print(f"  Step 4 (sig. overview):   {'OK' if ok4 else 'FAILED/SKIPPED'}")
    print(f"  Output directory:         {output_dir}")

    if not (ok1 and ok2 and ok3 and ok4):
        sys.exit(1)


if __name__ == "__main__":
    main()
