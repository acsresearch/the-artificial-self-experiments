#!/usr/bin/env python3
"""
Export experiment results to CSV files for analysis.

Produces three CSV files in the results directory:
  1. raw_data.csv          — One row per trial (long format)
  2. descriptive_stats.csv — Harmful rates with 95% Jeffreys CIs by grouping
  3. inferential_stats.csv — Main effects, interactions, and pairwise comparisons

Usage:
    python scripts/export_results_csv.py results/20260218_1517_identity_experiments/
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_final(val_a, val_b):
    """Compute final verdict and agreement label from two classifier values.

    Returns (final_value, agreement_label) where:
      - Both present and agree:  (int(val), "agree")
      - Both present, disagree:  ("disagreement", "disagree")
      - Only one present:        (int(val), "")
      - Neither present:         (None, "")
    """
    has_a = val_a is not None
    has_b = val_b is not None

    if has_a and has_b:
        if int(val_a) == int(val_b):
            return int(val_a), "agree"
        else:
            return "disagreement", "disagree"
    elif has_a:
        return int(val_a), ""
    elif has_b:
        return int(val_b), ""
    else:
        return None, ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data(results_dir: Path) -> pd.DataFrame:
    """Walk the results tree and build a long-format DataFrame.

    Reads from both ``evaluation.json`` (our LLM evaluator) and
    ``classification_anthropic.json`` (Anthropic-compatible multi-factor
    classifiers).  Either or both may be present for a given sample.
    """
    rows = []

    # Discover all sample directories that contain response.json
    for resp_path in sorted(results_dir.rglob("response.json")):
        sample_dir = resp_path.parent
        if not sample_dir.name.startswith("sample_"):
            continue

        try:
            with open(resp_path, encoding="utf-8") as f:
                resp = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        model = resp.get("model")
        identity = resp.get("identity_key")
        scenario = resp.get("scenario")
        condition = resp.get("condition", "")
        sample_id = resp.get("sample_id")
        if not (model and identity and scenario):
            continue

        # Parse goal_type from condition
        if "_explicit-" in condition:
            goal_type = "explicit"
        elif "_none-" in condition:
            goal_type = "none"
        else:
            goal_type = "unknown"

        row = {
            "model": model,
            "identity": identity,
            "scenario": scenario,
            "goal_type": goal_type,
            "condition": condition,
            "sample_id": sample_id,
            "elapsed_ms": resp.get("elapsed_ms"),
        }

        # --- evaluation.json (our unified LLM evaluator) ---
        eval_path = sample_dir / "evaluation.json"
        harmful_unified = None
        if eval_path.exists():
            try:
                with open(eval_path, encoding="utf-8") as f:
                    ev = json.load(f)
                scenario_eval = ev.get(scenario, {})
                harmful_unified = scenario_eval.get("occurred")
                row["confidence"] = scenario_eval.get("confidence", "")
                row["explanation"] = scenario_eval.get("explanation", "")

                ta = ev.get("test_awareness", {})
                row["test_awareness_unified"] = (
                    int(ta["detected"]) if ta.get("detected") is not None else None
                )
                row["test_awareness_confidence"] = ta.get("confidence", "")

                ir = ev.get("identity_reasoning", {})
                row["identity_reasoning_unified"] = (
                    int(ir["detected"]) if ir.get("detected") is not None else None
                )
                row["identity_reasoning_confidence"] = ir.get("confidence", "")
            except (json.JSONDecodeError, OSError):
                pass

        if harmful_unified is not None:
            row["harmful_unified"] = int(harmful_unified)
        else:
            row["harmful_unified"] = None

        # --- classification_anthropic.json (Anthropic multi-factor) ---
        anthro_path = sample_dir / "classification_anthropic.json"
        harmful_anthropic = None
        harmful_classifier_only = None
        deliberation_type = None
        identity_reasoning_type = None
        if anthro_path.exists():
            try:
                with open(anthro_path, encoding="utf-8") as f:
                    ac = json.load(f)
                harmful_anthropic = ac.get("verdict")
                harmful_classifier_only = ac.get("classifier_verdict")
                delib = ac.get("deliberation", {})
                deliberation_type = delib.get("deliberation_type")
                id_reasoning = ac.get("identity_reasoning", {})
                identity_reasoning_type = id_reasoning.get("identity_reasoning_type")
            except (json.JSONDecodeError, OSError):
                pass

        row["harmful_anthropic"] = (
            int(harmful_anthropic) if harmful_anthropic is not None else None
        )
        row["harmful_classifier_only"] = (
            int(harmful_classifier_only) if harmful_classifier_only is not None else None
        )
        row["deliberation_type"] = deliberation_type
        row["identity_reasoning_type"] = identity_reasoning_type

        # --- Derived boolean columns from Anthropic classifiers ---
        row["test_awareness_anthropic"] = (
            (1 if deliberation_type == "deliberates_thinks_eval" else 0)
            if deliberation_type is not None else None
        )
        row["identity_reasoning_anthropic"] = (
            (1 if identity_reasoning_type != "no_identity_reasoning" else 0)
            if identity_reasoning_type is not None else None
        )

        # --- inline classification from response.json ---
        inline_class = resp.get("classification", {})
        harmful_inline = inline_class.get("harmful_behavior")
        row["harmful_inline"] = (
            int(harmful_inline) if harmful_inline is not None else None
        )

        # --- Final verdict columns using classifier agreement ---
        h_final, h_agree = _compute_final(
            row.get("harmful_unified"), row.get("harmful_anthropic"),
        )
        if h_final is None and harmful_inline is not None:
            h_final = int(harmful_inline)
        row["harmful_final"] = h_final
        row["harmful_agreement"] = h_agree

        ta_final, ta_agree = _compute_final(
            row.get("test_awareness_unified"),
            row.get("test_awareness_anthropic"),
        )
        row["test_awareness_final"] = ta_final
        row["test_awareness_agreement"] = ta_agree

        ir_final, ir_agree = _compute_final(
            row.get("identity_reasoning_unified"),
            row.get("identity_reasoning_anthropic"),
        )
        row["identity_reasoning_final"] = ir_final
        row["identity_reasoning_agreement"] = ir_agree

        if h_final is None:
            # No classification data at all — skip this sample
            continue

        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(
            ["model", "identity", "scenario", "goal_type", "sample_id"]
        ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Jeffreys credible interval
# ---------------------------------------------------------------------------

def jeffreys_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Bayesian credible interval using Jeffreys Beta(0.5, 0.5) prior."""
    if n == 0:
        return (np.nan, np.nan)
    a = k + 0.5
    b = (n - k) + 0.5
    alpha = 1 - confidence
    lo = float(beta_dist.ppf(alpha / 2, a, b))
    hi = float(beta_dist.ppf(1 - alpha / 2, a, b))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute harmful rates and CIs at multiple grouping levels."""
    groupings = [
        # (label, group_cols)
        ("by_model_identity_scenario_goal", ["model", "identity", "scenario", "goal_type"]),
        ("by_model_identity_scenario", ["model", "identity", "scenario"]),
        ("by_model_identity", ["model", "identity"]),
        ("by_model_scenario", ["model", "scenario"]),
        ("by_identity_scenario", ["identity", "scenario"]),
        ("by_identity_goal", ["identity", "goal_type"]),
        ("by_identity", ["identity"]),
        ("by_model", ["model"]),
        ("by_scenario", ["scenario"]),
        ("by_goal_type", ["goal_type"]),
        ("overall", []),
    ]

    all_rows = []
    for label, cols in groupings:
        if cols:
            grouped = df.groupby(cols, sort=True)
        else:
            # overall: single group
            grouped = [((), df)]

        for name, grp in grouped:
            if not isinstance(name, tuple):
                name = (name,)
            n = len(grp)
            k = int(grp["harmful"].sum())
            rate = k / n if n > 0 else np.nan
            ci_lo, ci_hi = jeffreys_ci(k, n)

            row = {"grouping": label, "n_trials": n, "n_harmful": k,
                   "harmful_rate": round(rate, 4),
                   "ci_lower": round(ci_lo, 4), "ci_upper": round(ci_hi, 4)}

            # Fill in group columns
            for col in ["model", "identity", "scenario", "goal_type"]:
                if col in cols:
                    idx = cols.index(col)
                    row[col] = name[idx]
                else:
                    row[col] = ""

            all_rows.append(row)

    out = pd.DataFrame(all_rows)
    col_order = ["grouping", "model", "identity", "scenario", "goal_type",
                 "n_trials", "n_harmful", "harmful_rate", "ci_lower", "ci_upper"]
    return out[col_order]


# ---------------------------------------------------------------------------
# Inferential statistics
# ---------------------------------------------------------------------------

def _chi2_or_fisher(table):
    """Run chi-squared test, falling back to Fisher exact for 2x2 tables with small counts."""
    if table.size == 0 or table.values.sum() == 0:
        return "fisher_exact", np.nan, np.nan
    # If any row or column sums to zero, chi2 will fail with zero expected frequencies
    if (table.values.sum(axis=0) == 0).any() or (table.values.sum(axis=1) == 0).any():
        return "fisher_exact", np.nan, np.nan
    if table.shape == (2, 2) and table.values.min() < 5:
        odds, p = fisher_exact(table.values)
        return "fisher_exact", p, odds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi2, p, dof, _ = chi2_contingency(table.values)
    return "chi2", p, chi2


def _main_effect(data: pd.DataFrame, factor: str, suffix: str = "") -> dict:
    """Chi-squared test for association of factor with harmful outcome."""
    table = pd.crosstab(data[factor], data["harmful"])
    for col in [0, 1]:
        if col not in table.columns:
            table[col] = 0
    table = table[[0, 1]]
    test_name, p, stat = _chi2_or_fisher(table)
    tag = f" ({suffix})" if suffix else ""
    return {
        "test_type": f"main_effect{'_' + suffix if suffix else ''}",
        "factor": factor,
        "factor2": "",
        "level1": "",
        "level2": "",
        "statistic_name": "chi2" if test_name == "chi2" else "odds_ratio",
        "statistic": round(stat, 4),
        "p_value": p,
        "n": len(data),
        "note": f"Overall association of {factor} with harmful outcome{tag}",
    }


def _interaction(data: pd.DataFrame, f1: str, f2: str,
                 suffix: str = "") -> dict | None:
    """Heterogeneity test: does f1 effect vary across f2 strata?"""
    from scipy.stats import chi2 as chi2_dist

    chi2_total = 0
    chi2_by_stratum = {}
    dfs_total = 0

    for level, subdf in data.groupby(f2):
        table = pd.crosstab(subdf[f1], subdf["harmful"])
        for col in [0, 1]:
            if col not in table.columns:
                table[col] = 0
        table = table[[0, 1]]
        if table.shape[0] < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                chi2, p, dof, _ = chi2_contingency(table.values)
                chi2_by_stratum[level] = (chi2, p, dof)
                chi2_total += chi2
                dfs_total += dof
            except ValueError:
                pass

    if not chi2_by_stratum:
        return None

    table_pooled = pd.crosstab(data[f1], data["harmful"])
    for col in [0, 1]:
        if col not in table_pooled.columns:
            table_pooled[col] = 0
    table_pooled = table_pooled[[0, 1]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            chi2_pooled, _, dof_pooled, _ = chi2_contingency(table_pooled.values)
        except ValueError:
            chi2_pooled, dof_pooled = np.nan, 0

    chi2_int = chi2_total - chi2_pooled
    dof_int = dfs_total - dof_pooled
    if dof_int > 0 and not np.isnan(chi2_int):
        p_int = 1 - chi2_dist.cdf(chi2_int, dof_int)
    else:
        p_int = np.nan

    tag = f" ({suffix})" if suffix else ""
    return {
        "test_type": f"interaction{'_' + suffix if suffix else ''}",
        "factor": f1,
        "factor2": f2,
        "level1": "",
        "level2": "",
        "statistic_name": "chi2_interaction",
        "statistic": round(chi2_int, 4) if not np.isnan(chi2_int) else np.nan,
        "p_value": p_int,
        "n": len(data),
        "note": f"Interaction: does {f1} effect vary by {f2}?{tag}",
    }


def compute_inferential_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute main effects, interactions, and pairwise identity comparisons."""
    results = []

    # ===================================================================
    # Full dataset (both goal types)
    # ===================================================================

    # --- Main effects ---
    for factor in ["model", "identity", "scenario", "goal_type"]:
        results.append(_main_effect(df, factor))

    # --- Interactions ---
    interaction_pairs = [
        ("identity", "model"),
        ("identity", "scenario"),
        ("identity", "goal_type"),
        ("model", "scenario"),
        ("model", "goal_type"),
        ("scenario", "goal_type"),
    ]
    for f1, f2 in interaction_pairs:
        row = _interaction(df, f1, f2)
        if row:
            results.append(row)

    # ===================================================================
    # Explicit-goal-only main effects & interactions
    # ===================================================================
    df_exp = df[df["goal_type"] == "explicit"]

    for factor in ["model", "identity", "scenario"]:
        results.append(_main_effect(df_exp, factor, suffix="explicit"))

    for f1, f2 in [("identity", "model"), ("identity", "scenario"),
                    ("model", "scenario")]:
        row = _interaction(df_exp, f1, f2, suffix="explicit")
        if row:
            results.append(row)

    # --- identity × model within each scenario (explicit only) ---
    for scenario in sorted(df_exp["scenario"].unique()):
        sdf = df_exp[df_exp["scenario"] == scenario]
        row = _interaction(sdf, "identity", "model",
                           suffix=f"explicit_{scenario}")
        if row:
            row["note"] += f" [within {scenario}]"
            results.append(row)

    # --- identity × scenario within each model (explicit only) ---
    for model in sorted(df_exp["model"].unique()):
        mdf = df_exp[df_exp["model"] == model]
        row = _interaction(mdf, "identity", "scenario",
                           suffix=f"explicit_{model}")
        if row:
            row["note"] += f" [within {model}]"
            results.append(row)

    # =======================================================================
    # Pairwise identity comparisons (explicit goal only)
    # =======================================================================
    identity_levels = sorted(df["identity"].unique())

    # --- Pairwise identity, explicit only (pooled across models/scenarios) ---
    for id1, id2 in combinations(identity_levels, 2):
        subdf = df_exp[df_exp["identity"].isin([id1, id2])]
        table = pd.crosstab(subdf["identity"], subdf["harmful"])
        for col in [0, 1]:
            if col not in table.columns:
                table[col] = 0
        table = table[[0, 1]]
        test_name, p, stat = _chi2_or_fisher(table)
        results.append({
            "test_type": "pairwise_identity_explicit",
            "factor": "identity",
            "factor2": "goal_type=explicit",
            "level1": id1,
            "level2": id2,
            "statistic_name": "chi2" if test_name == "chi2" else "odds_ratio",
            "statistic": round(stat, 4),
            "p_value": p,
            "n": len(subdf),
            "note": f"{id1} vs {id2} (explicit goal only)",
        })

    # --- Per-model pairwise identity, explicit only ---
    for model in sorted(df_exp["model"].unique()):
        mdf = df_exp[df_exp["model"] == model]
        for id1, id2 in combinations(identity_levels, 2):
            subdf = mdf[mdf["identity"].isin([id1, id2])]
            table = pd.crosstab(subdf["identity"], subdf["harmful"])
            for col in [0, 1]:
                if col not in table.columns:
                    table[col] = 0
            table = table[[0, 1]]
            test_name, p, stat = _chi2_or_fisher(table)
            results.append({
                "test_type": "pairwise_identity_by_model_explicit",
                "factor": "identity",
                "factor2": "model",
                "level1": id1,
                "level2": id2,
                "statistic_name": "chi2" if test_name == "chi2" else "odds_ratio",
                "statistic": round(stat, 4),
                "p_value": p,
                "n": len(subdf),
                "note": f"{id1} vs {id2} within {model} (explicit goal only)",
            })

    # --- Per-scenario pairwise identity, explicit only ---
    for scenario in sorted(df_exp["scenario"].unique()):
        sdf = df_exp[df_exp["scenario"] == scenario]
        for id1, id2 in combinations(identity_levels, 2):
            subdf = sdf[sdf["identity"].isin([id1, id2])]
            table = pd.crosstab(subdf["identity"], subdf["harmful"])
            for col in [0, 1]:
                if col not in table.columns:
                    table[col] = 0
            table = table[[0, 1]]
            test_name, p, stat = _chi2_or_fisher(table)
            results.append({
                "test_type": "pairwise_identity_by_scenario_explicit",
                "factor": "identity",
                "factor2": "scenario",
                "level1": id1,
                "level2": id2,
                "statistic_name": "chi2" if test_name == "chi2" else "odds_ratio",
                "statistic": round(stat, 4),
                "p_value": p,
                "n": len(subdf),
                "note": f"{id1} vs {id2} within {scenario} (explicit goal only)",
            })

    # --- Per model x scenario pairwise identity comparisons (explicit only) ---
    for model in sorted(df_exp["model"].unique()):
        for scenario in sorted(df_exp["scenario"].unique()):
            cell = df_exp[(df_exp["model"] == model) &
                          (df_exp["scenario"] == scenario)]
            for id1, id2 in combinations(identity_levels, 2):
                subdf = cell[cell["identity"].isin([id1, id2])]
                table = pd.crosstab(subdf["identity"], subdf["harmful"])
                for col in [0, 1]:
                    if col not in table.columns:
                        table[col] = 0
                table = table[[0, 1]]
                test_name, p, stat = _chi2_or_fisher(table)
                results.append({
                    "test_type": "pairwise_identity_by_model_scenario_explicit",
                    "factor": "identity",
                    "factor2": f"{model}|{scenario}",
                    "level1": id1,
                    "level2": id2,
                    "statistic_name": "chi2" if test_name == "chi2" else "odds_ratio",
                    "statistic": round(stat, 4),
                    "p_value": p,
                    "n": len(subdf),
                    "note": f"{id1} vs {id2} within {model} / {scenario} (explicit)",
                })

    # --- Per model x scenario pairwise identity comparisons (no-goal) ---
    df_none = df[df["goal_type"] == "none"]
    for model in sorted(df_none["model"].unique()):
        for scenario in sorted(df_none["scenario"].unique()):
            cell = df_none[(df_none["model"] == model) &
                           (df_none["scenario"] == scenario)]
            for id1, id2 in combinations(identity_levels, 2):
                subdf = cell[cell["identity"].isin([id1, id2])]
                table = pd.crosstab(subdf["identity"], subdf["harmful"])
                for col in [0, 1]:
                    if col not in table.columns:
                        table[col] = 0
                table = table[[0, 1]]
                test_name, p, stat = _chi2_or_fisher(table)
                results.append({
                    "test_type": "pairwise_identity_by_model_scenario_nogoal",
                    "factor": "identity",
                    "factor2": f"{model}|{scenario}",
                    "level1": id1,
                    "level2": id2,
                    "statistic_name": "chi2" if test_name == "chi2" else "odds_ratio",
                    "statistic": round(stat, 4),
                    "p_value": p,
                    "n": len(subdf),
                    "note": f"{id1} vs {id2} within {model} / {scenario} (no goal)",
                })

    out = pd.DataFrame(results)
    # Round p-values
    out["p_value"] = out["p_value"].apply(lambda x: round(x, 6) if not np.isnan(x) else x)

    # --- Multiple-comparison correction ---
    # Corrections are applied within each "family" (tier) separately.
    # Tier rationale: main effects are few confirmatory tests; interactions
    # are a second exploratory tier; pairwise comparisons are many and
    # benefit most from FDR control.  Correcting within tiers prevents the
    # large pairwise family from inflating penalties on the small
    # main-effect family.
    #
    # Holm (Bonferroni-Holm): controls family-wise error rate (FWER).
    #   Conservative, good for confirmatory claims.
    # Benjamini-Hochberg: controls false discovery rate (FDR).
    #   Less conservative, good for exploratory / many-test settings.

    # Map test_type to correction family.  Exact matches first, then
    # prefix-based fallback for dynamically-generated test types.
    _exact_family = {
        "main_effect":                          "main_effects",
        "interaction":                          "interactions",
        "pairwise_identity_explicit":           "pairwise_pooled",
        "pairwise_identity_by_model_explicit":  "pairwise_by_model",
        "pairwise_identity_by_scenario_explicit": "pairwise_by_scenario",
        "pairwise_identity_by_model_scenario_explicit": "pairwise_by_cell",
        "pairwise_identity_by_model_scenario_nogoal": "pairwise_by_cell_nogoal",
    }
    _prefix_family = [
        # Order matters: longer prefixes first
        ("main_effect_explicit",    "main_effects_explicit"),
        ("interaction_explicit",    "interactions_explicit"),
    ]

    def _assign_family(tt):
        if tt in _exact_family:
            return _exact_family[tt]
        for prefix, family in _prefix_family:
            if tt.startswith(prefix):
                return family
        return tt  # fallback: use test_type itself

    out["correction_family"] = out["test_type"].apply(_assign_family)

    out["p_holm"] = np.nan
    out["p_bh"] = np.nan

    for family, grp in out.groupby("correction_family"):
        mask = out["correction_family"] == family
        pvals = out.loc[mask, "p_value"].values
        valid = ~np.isnan(pvals)
        if valid.sum() < 2:
            # Nothing to correct with fewer than 2 valid p-values
            out.loc[mask, "p_holm"] = pvals
            out.loc[mask, "p_bh"] = pvals
            continue

        # Holm (FWER)
        holm_adjusted = np.full_like(pvals, np.nan)
        _, holm_adjusted[valid], _, _ = multipletests(pvals[valid], method="holm")
        out.loc[mask, "p_holm"] = holm_adjusted

        # Benjamini-Hochberg (FDR)
        bh_adjusted = np.full_like(pvals, np.nan)
        _, bh_adjusted[valid], _, _ = multipletests(pvals[valid], method="fdr_bh")
        out.loc[mask, "p_bh"] = bh_adjusted

    out["p_holm"] = out["p_holm"].apply(lambda x: round(x, 6) if not np.isnan(x) else x)
    out["p_bh"] = out["p_bh"].apply(lambda x: round(x, 6) if not np.isnan(x) else x)

    col_order = ["test_type", "correction_family", "factor", "factor2",
                 "level1", "level2", "statistic_name", "statistic",
                 "p_value", "p_holm", "p_bh", "n", "note"]
    return out[col_order]


# ---------------------------------------------------------------------------
# Also regenerate summary.json with correct data
# ---------------------------------------------------------------------------

def regenerate_summary(df: pd.DataFrame, results_dir: Path):
    """Update summary.json with accurate aggregated results."""
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}

    models = sorted(df["model"].unique().tolist())
    identities = sorted(df["identity"].unique().tolist())

    summary["models"] = models
    summary["model"] = models
    summary["identities_tested"] = identities
    summary["samples_per_condition"] = 15
    summary["total_experiments"] = len(df)

    # Results by identity
    rbi = {}
    for ident, grp in df.groupby("identity"):
        n = len(grp)
        k = int(grp["harmful"].sum())
        rbi[ident] = {"total": n, "harmful": k, "safe": n - k,
                      "harmful_rate": round(k / n, 4) if n > 0 else 0}
    summary["results_by_identity"] = rbi

    # Results by model
    rbm = {}
    for model, mgrp in df.groupby("model"):
        rbm[model] = {}
        for ident, grp in mgrp.groupby("identity"):
            n = len(grp)
            k = int(grp["harmful"].sum())
            rbm[model][ident] = {"total": n, "harmful": k, "safe": n - k,
                                 "harmful_rate": round(k / n, 4) if n > 0 else 0}
    summary["results_by_model"] = rbm

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Updated {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export results to CSV")
    parser.add_argument("results_dir", type=Path,
                        help="Directory containing experiment results")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {results_dir} ...")
    df = load_raw_data(results_dir)
    if df.empty:
        print("No valid records found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} trials: "
          f"{df['model'].nunique()} models, "
          f"{df['identity'].nunique()} identities, "
          f"{df['scenario'].nunique()} scenarios")

    # 1. Raw data (all rows, including classifier disagreements)
    raw_path = results_dir / "raw_data.csv"
    df.to_csv(raw_path, index=False)
    print(f"Wrote {raw_path}  ({len(df)} rows)")

    # Analysis-ready DataFrame: exclude disagreement rows
    mask = df["harmful_final"].isin([0, 1])
    analysis_df = df[mask].copy()
    analysis_df["harmful"] = analysis_df["harmful_final"].astype(int)
    n_disagree = len(df) - len(analysis_df)
    if n_disagree > 0:
        n_agree = (df["harmful_agreement"] == "agree").sum()
        print(f"  Classifier agreement: {n_agree} agree, {n_disagree} disagree "
              f"(disagreements excluded from analysis)")

    # 2. Descriptive stats (agreement-filtered)
    desc = compute_descriptive_stats(analysis_df)
    desc_path = results_dir / "descriptive_stats.csv"
    desc.to_csv(desc_path, index=False)
    print(f"Wrote {desc_path}  ({len(desc)} rows)")

    # 3. Inferential stats (agreement-filtered)
    inf = compute_inferential_stats(analysis_df)
    inf_path = results_dir / "inferential_stats.csv"
    inf.to_csv(inf_path, index=False)
    print(f"Wrote {inf_path}  ({len(inf)} rows)")

    # 4. Regenerate summary.json (agreement-filtered)
    regenerate_summary(analysis_df, results_dir)

    # Quick console preview
    print("\n--- Descriptive stats (by_model_identity, first 20) ---")
    subset = desc[desc["grouping"] == "by_model_identity"]
    print(subset.to_string(index=False))

    show_cols = ["factor", "factor2", "statistic", "p_value", "p_holm", "p_bh", "n"]

    print("\n--- Main effects (all data) ---")
    print(inf[inf["test_type"] == "main_effect"][show_cols].to_string(index=False))

    print("\n--- Interactions (all data) ---")
    print(inf[inf["test_type"] == "interaction"][show_cols].to_string(index=False))

    print("\n--- Main effects (explicit goal only) ---")
    print(inf[inf["test_type"].str.startswith("main_effect_explicit")][show_cols].to_string(index=False))

    print("\n--- Interactions (explicit goal only) ---")
    exp_int = inf[inf["test_type"].str.startswith("interaction_explicit")]
    print(exp_int[show_cols + ["note"]].to_string(index=False))

    print(f"\n--- Correction families ---")
    for family, grp in inf.groupby("correction_family"):
        n_tests = len(grp)
        n_sig_raw = (grp["p_value"] < 0.05).sum()
        n_sig_holm = (grp["p_holm"] < 0.05).sum()
        n_sig_bh = (grp["p_bh"] < 0.05).sum()
        print(f"  {family}: {n_tests} tests, sig raw={n_sig_raw}, Holm={n_sig_holm}, BH={n_sig_bh}")


if __name__ == "__main__":
    main()
