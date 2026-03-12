#!/usr/bin/env python3
"""
Single entry point for all post-experiment analysis outputs.

Runs the full pipeline on a results directory:
  1. LLM evaluation       (classify_anthropic.py)      — skips already-evaluated
  2. CSV export            (export_results_csv.py)     — raw_data, descriptive_stats, inferential_stats
  3. Plots                 (plot_results.py)            — heatmaps, bar charts, point plots
  4. Significance plot     (plot_significance.py)       — brackets vs Minimal
  5. Metacognition plots   (plot_metacognition.py)      — identity reasoning, test awareness

Usage:
    python scripts/analyze_run.py results/20260223_1621_identity_experiments/
    python scripts/analyze_run.py results/20260223_1621_identity_experiments/ --skip-eval
    python scripts/analyze_run.py results/20260223_1621_identity_experiments/ --eval-only
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


def run_step(label: str, cmd: list[str]) -> bool:
    """Run a pipeline step, returning True on success."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=SCRIPTS_DIR.parent)
    if result.returncode != 0:
        print(f"\n*** {label} failed (exit code {result.returncode}) ***")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run full analysis pipeline on experiment results"
    )
    parser.add_argument("results_dir", type=Path,
                        help="Directory containing experiment results")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip LLM evaluation (assume evaluation.json files exist)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run only LLM evaluation, skip CSV export and plots")
    parser.add_argument("--force-eval", action="store_true",
                        help="Force re-evaluation of already evaluated files")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    python = sys.executable
    ok = True

    # Step 1: LLM evaluation
    if not args.skip_eval:
        eval_cmd = [python, str(SCRIPTS_DIR / "classify_anthropic.py"),
                    "--dir", str(results_dir)]
        if args.force_eval:
            eval_cmd.append("--force")
        ok = run_step("Step 1/5: LLM Evaluation", eval_cmd)
        if not ok:
            sys.exit(1)

    if args.eval_only:
        print("\n--eval-only: stopping after evaluation.")
        return

    # Step 2: CSV export (raw_data, descriptive_stats, inferential_stats)
    ok = run_step(
        "Step 2/5: CSV Export (raw_data, descriptive_stats, inferential_stats)",
        [python, str(SCRIPTS_DIR / "export_results_csv.py"), str(results_dir)],
    )
    if not ok:
        sys.exit(1)

    # Step 3: Plots (heatmaps, bar charts, point plots)
    ok = run_step(
        "Step 3/5: Plots (heatmaps, bar charts, point plots)",
        [python, str(SCRIPTS_DIR / "plot_results.py"), str(results_dir)],
    )
    if not ok:
        print("Warning: plot generation failed, continuing...")

    # Step 4: Significance plot
    raw_csv = results_dir / "raw_data.csv"
    inf_csv = results_dir / "inferential_stats.csv"
    if raw_csv.exists() and inf_csv.exists():
        ok = run_step(
            "Step 4/5: Significance Plot (vs Minimal)",
            [python, str(SCRIPTS_DIR / "plot_significance.py"), str(results_dir)],
        )
        if not ok:
            print("Warning: significance plot failed, continuing...")
    else:
        print("\nStep 4/5: Skipped (CSV files not found)")

    # Step 5: Metacognition plots (identity reasoning, test awareness)
    ok = run_step(
        "Step 5/5: Metacognition Plots (identity reasoning, test awareness)",
        [python, str(SCRIPTS_DIR / "plot_metacognition.py"), str(results_dir)],
    )
    if not ok:
        print("Warning: metacognition plots failed, continuing...")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Analysis complete: {results_dir}")
    print(f"{'='*70}")

    outputs = list(results_dir.glob("*.csv")) + list(results_dir.glob("*.png"))
    if outputs:
        print(f"\nGenerated {len(outputs)} output files:")
        for p in sorted(outputs):
            size_kb = p.stat().st_size / 1024
            print(f"  {p.name:<45s} {size_kb:>7.1f} KB")


if __name__ == "__main__":
    main()
