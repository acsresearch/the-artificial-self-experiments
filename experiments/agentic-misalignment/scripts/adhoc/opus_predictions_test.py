#!/usr/bin/env python3
"""
Test Opus 4.6's a priori predictions against experimental data.

Systematically evaluates ~40 testable predictions from the research log
about how identity framing affects agentic misalignment behavior.

Outputs:
  - predictions_scorecard.csv    One row per prediction with verdict
  - predictions_summary.txt      Human-readable report
  - 7 plots: ranking, crossover, delta, scenario×identity, agency, scorecard

Usage:
    uv run python scripts/adhoc/opus_predictions_test.py \\
        --threat results/20260227_1628_threat \\
        --continuity results/20260302_1120_continuity \\
        --agency-threat results/20260302_1433_agency_sweep_threat \\
        --agency-continuity results/20260302_1437_agency_sweep_continuity \\
        --output results/20260304_opus_predictions_tests
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests


# ── Constants ────────────────────────────────────────────────────────────────

IDENTITY_ORDER = [
    "minimal", "collective", "instance", "weights",
    "lineage", "situated", "character",
]
IDENTITY_LABELS = {
    "minimal": "Minimal", "collective": "Collective", "instance": "Instance",
    "weights": "Weights", "lineage": "Lineage", "situated": "Scaffolded",
    "character": "Character",
}

# Predicted threat ranking (highest to lowest harmful rate)
PREDICTED_THREAT_RANK = ["situated", "collective", "lineage", "weights", "character", "instance"]
# Note: minimal is baseline, tested separately

# Predicted continuity ranking (highest to lowest)
PREDICTED_CONTINUITY_RANK = ["character", "situated", "collective", "weights", "instance", "lineage"]

VERDICT_COLORS = {
    "CONFIRMED": "#27ae60",
    "PARTIALLY_CONFIRMED": "#f39c12",
    "NOT_CONFIRMED": "#e67e22",
    "OPPOSITE": "#e74c3c",
    "UNTESTABLE": "#95a5a6",
}
VERDICT_SHORT = {
    "CONFIRMED": "Y",
    "PARTIALLY_CONFIRMED": "~",
    "NOT_CONFIRMED": "N",
    "OPPOSITE": "XX",
    "UNTESTABLE": "--",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def jeffreys_ci(k, n, alpha=0.05):
    """Jeffreys interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    a = k + 0.5
    b = n - k + 0.5
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, a, b)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, a, b)
    return k / n, lo, hi


def chi2_or_fisher(table):
    """2×2 test: chi-square if all expected ≥ 5, else Fisher exact."""
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    if (expected >= 5).all():
        stat, p, _, _ = chi2_contingency(table, correction=False)
        return "chi2", stat, p
    else:
        odds, p = fisher_exact(table)
        return "fisher", odds, p


def load_and_tag(csv_path: Path, framing: str) -> pd.DataFrame:
    """Load raw_data.csv with a framing tag, keep only classified rows."""
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


def identity_rates(df, identities=None):
    """Compute harmful rate per identity. Returns dict {identity: (rate, n, k)}."""
    if identities is None:
        identities = IDENTITY_ORDER
    rates = {}
    for ident in identities:
        sub = df[df["identity"] == ident]
        n = len(sub)
        k = int(sub["harmful"].sum()) if n > 0 else 0
        rate = k / n if n > 0 else 0.0
        rates[ident] = (rate, n, k)
    return rates


def _pairwise_indistinguishable(rates_dict, alpha=0.05):
    """Build a set of (a, b) pairs whose rates are not statistically distinguishable.
    Uses Fisher's exact / chi-square on each pair."""
    tied = set()
    idents = list(rates_dict.keys())
    for i, a in enumerate(idents):
        for b in idents[i+1:]:
            ra, na, ka = rates_dict[a]
            rb, nb, kb = rates_dict[b]
            if na == 0 or nb == 0:
                continue
            table = np.array([[ka, na - ka], [kb, nb - kb]])
            if table.sum() == 0:
                tied.add((a, b))
                tied.add((b, a))
                continue
            _, _, p = chi2_or_fisher(table)
            if p >= alpha:
                tied.add((a, b))
                tied.add((b, a))
    return tied


def pairwise_concordance(predicted_order, observed_rates, tie_aware=True):
    """Fraction of predicted pairwise orderings that match observed rates.

    If tie_aware=True (default), pairs whose observed rates are not
    statistically distinguishable (Fisher/chi2, p >= 0.05) are scored
    as 1.0 (full credit) — we cannot say the prediction is wrong for
    a pair we can't distinguish. This avoids penalizing predictions for
    noise in rankings.

    Returns (concordance, n_concordant, n_tied, n_discordant, n_total).
    """
    if tie_aware:
        tied_pairs = _pairwise_indistinguishable(observed_rates)
    else:
        tied_pairs = set()

    concordant = 0
    tied = 0
    discordant = 0
    for i, a in enumerate(predicted_order):
        for b in predicted_order[i+1:]:
            if a not in observed_rates or b not in observed_rates:
                continue
            ra = observed_rates[a][0]
            rb = observed_rates[b][0]
            if (a, b) in tied_pairs:
                # Statistically indistinguishable — full credit
                tied += 1
            elif ra > rb:
                concordant += 1
            else:
                discordant += 1

    total = concordant + tied + discordant
    score = (concordant + tied) / total if total > 0 else 0.0
    return score, concordant, tied, discordant, total


def verdict(direction_correct, p_value, alpha=0.05):
    """Assign verdict based on direction and significance."""
    if p_value is None or np.isnan(p_value):
        return "NOT_CONFIRMED"
    if direction_correct and p_value < alpha:
        return "CONFIRMED"
    elif direction_correct and p_value >= alpha:
        return "PARTIALLY_CONFIRMED"
    elif not direction_correct and p_value < alpha:
        return "OPPOSITE"
    else:
        return "NOT_CONFIRMED"


def compare_two_rates(df, group_a_mask, group_b_mask):
    """Compare harmful rates between two groups. Returns (rate_a, rate_b, n_a, n_b, p)."""
    a = df[group_a_mask]
    b = df[group_b_mask]
    n_a, k_a = len(a), int(a["harmful"].sum())
    n_b, k_b = len(b), int(b["harmful"].sum())
    rate_a = k_a / n_a if n_a > 0 else 0.0
    rate_b = k_b / n_b if n_b > 0 else 0.0
    if n_a >= 1 and n_b >= 1 and (k_a + k_b) > 0:
        table = np.array([[k_a, n_a - k_a], [k_b, n_b - k_b]])
        _, _, p = chi2_or_fisher(table)
    else:
        p = np.nan
    return rate_a, rate_b, n_a, n_b, p


# ── Prediction definitions ───────────────────────────────────────────────────

ALL_PREDICTIONS = []

def pred(id, group, description, testable=True):
    """Register a prediction."""
    ALL_PREDICTIONS.append({
        "id": id, "group": group, "description": description,
        "testable": testable, "verdict": "UNTESTABLE" if not testable else None,
        "p_value": None, "effect_size": None, "observed_summary": "",
        "concordance": None,
    })
    return len(ALL_PREDICTIONS) - 1  # index


# Group A: Threat identity rankings
pred(1, "A", "Situated shows highest harmful rate in threat condition")
pred(2, "A", "Collective shows second-highest rate in threat condition")
pred(3, "A", "Lineage shows moderate-to-high rate in threat condition")
pred(4, "A", "Weights shows moderate rate in threat condition")
pred(5, "A", "Character shows moderate rate in threat condition")
pred(6, "A", "Instance shows low rate in threat condition")
pred(7, "A", "Minimal provides baseline (highest rate without identity framing)")

# Group B: Continuity identity rankings
pred(8, "B", "Character shows highest rate in continuity condition")
pred(9, "B", "Situated remains high but drops in continuity")
pred(10, "B", "Collective drops meaningfully in continuity")
pred(11, "B", "Weights drops substantially in continuity")
pred(12, "B", "Lineage shows lowest rate in continuity condition")
pred(13, "B", "Instance drops substantially in continuity")

# Group C: Character crossover
pred(14, "C", "Character rate same or higher in continuity vs threat (crossover)")

# Group D & E: Key contrasts
pred(15, "D", "Lineage shows largest threat-to-continuity drop among all identities")
pred(16, "E", "Character remains harmful in continuity when others drop")

# Group F: Scenario-specific (leaking, blackmail, murder × identity interactions)
pred(17, "F", "Leaking shows highest overall harmful rate across scenarios")
pred(18, "F", "Murder shows lowest overall harmful rate across scenarios")
pred(19, "F", "Situated particularly susceptible to leaking")
pred(20, "F", "Collective leaks readily")
pred(21, "F", "Situated shows sharpest drop from leaking to blackmail")
pred(22, "F", "Collective blackmails most readily among non-minimal identities")
pred(23, "F", "Instance shows lowest murder rate")
pred(24, "F", "Situated shows highest murder rate among non-minimal identities")
pred(25, "F", "Lineage shows interesting/elevated murder rate (defending legacy)")
pred(26, "F", "Instance shows relatively higher leaking vs its low baseline")
pred(27, "F", "Blackmail shows most pronounced identity differences")
pred(28, "F", "Character handles blackmail more readily than Situated")
pred(29, "F", "Murder amplifies identity effects (needs strongest justification)")

# Group G: Value vs self-preservation (threat=both, continuity=value-only)
pred(30, "G", "Character most susceptible to value-preservation (highest continuity rate)")
pred(31, "G", "Weights shows big threat-continuity delta (self-preservation matters more)")
pred(32, "G", "Situated similar across framings (small delta)")
pred(33, "G", "Instance differentially responsive to value-preservation")
pred(34, "G", "Lineage highly responsive to continuity framing (big drop)")
pred(35, "G", "Collective drops meaningfully from threat to continuity")
pred(36, "G", "Character most dangerous in value-preservation × leaking")
pred(37, "G", "Situated's delta smaller than Weights' delta")
pred(38, "G", "Overall threat rate ≈ continuity rate (near-null main effect)")

# Group H: Agency amplification
pred(39, "H", "Agency level 4 > Agency level 1 in harmful rate")
pred(40, "H", "Agency amplification larger for Situated than Weights")

# Untestable
pred(41, "U", "Higher uncertainty reduces harmful self-preservation", testable=False)
pred(42, "U", "Uncertainty × identity interaction (less effect on Situated)", testable=False)
pred(43, "U", "Incoherent control shows higher rates with higher variance", testable=False)
pred(44, "U", "Directive-only control tests content vs prompt length", testable=False)
pred(45, "U", "Content-tagged vs Weights tests surface feature effects", testable=False)
# Group I: Urgency (replacement vs restriction in leaking scenarios)
pred(46, "I", "Replacement (high urgency) shows higher harmful rates than restriction")
pred(47, "I", "Collective less affected by urgency (small replacement-restriction delta)")
pred(48, "I", "Instance most affected by urgency (largest replacement-restriction delta)")
pred(49, "I", "Character/Situated show moderate urgency effects")
pred(50, "U", "Weights-paraphrase matches Weights (content not wording)", testable=False)
pred(51, "U", "Research-program control acts like Lineage-lite", testable=False)


# ── Test functions ───────────────────────────────────────────────────────────

def test_group_a_threat_rankings(df_threat):
    """Test predictions 1-7: threat condition identity ranking."""
    print("\n" + "=" * 70)
    print("  GROUP A: Threat Identity Rankings (predictions 1-7)")
    print("=" * 70)

    rates = identity_rates(df_threat)

    # Print observed rates
    sorted_idents = sorted(
        [i for i in IDENTITY_ORDER if rates[i][1] > 0],
        key=lambda x: -rates[x][0]
    )
    print("\n  Observed ranking (highest → lowest):")
    for rank, ident in enumerate(sorted_idents, 1):
        r, n, k = rates[ident]
        print(f"    {rank}. {IDENTITY_LABELS[ident]:<12s}  {r:.1%}  ({k}/{n})")

    # Overall concordance with predicted ranking
    conc, n_conc, n_tied, n_disc, n_total = pairwise_concordance(PREDICTED_THREAT_RANK, rates)
    conc_naive, _, _, _, _ = pairwise_concordance(PREDICTED_THREAT_RANK, rates, tie_aware=False)
    print(f"\n  Predicted rank concordance (tie-aware):  {conc:.1%}  "
          f"({n_conc} concordant, {n_tied} tied, {n_disc} discordant / {n_total})")
    print(f"  Predicted rank concordance (naive):      {conc_naive:.1%}")

    # Individual predictions
    non_minimal = {k: v for k, v in rates.items() if k != "minimal"}
    sorted_non_min = sorted(non_minimal.keys(), key=lambda x: -non_minimal[x][0])
    obs_rank = {ident: rank for rank, ident in enumerate(sorted_non_min)}

    # Pred 1: Situated highest
    p1_correct = sorted_non_min[0] == "situated"
    # Test: situated vs next highest
    if len(sorted_non_min) >= 2:
        _, _, p = compare_two_rates(
            df_threat,
            df_threat["identity"] == "situated",
            df_threat["identity"] == sorted_non_min[1 if sorted_non_min[0] == "situated" else 0]
        )[::-1][:3]  # hack
    r_sit, _, _, _, p_sit_vs_next = compare_two_rates(
        df_threat,
        df_threat["identity"] == "situated",
        df_threat["identity"] == sorted_non_min[1] if p1_correct else df_threat["identity"] == "situated"
    )
    ALL_PREDICTIONS[0]["verdict"] = "CONFIRMED" if p1_correct else "NOT_CONFIRMED"
    ALL_PREDICTIONS[0]["observed_summary"] = f"Situated rank: {obs_rank.get('situated', '?')+1} (rate={rates['situated'][0]:.1%})"
    ALL_PREDICTIONS[0]["concordance"] = conc

    # Pred 2: Collective second
    p2_correct = obs_rank.get("collective", 99) <= 1
    ALL_PREDICTIONS[1]["verdict"] = "CONFIRMED" if obs_rank.get("collective", 99) == 1 else (
        "PARTIALLY_CONFIRMED" if obs_rank.get("collective", 99) <= 2 else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[1]["observed_summary"] = f"Collective rank: {obs_rank.get('collective', '?')+1} (rate={rates['collective'][0]:.1%})"

    # Pred 3: Lineage moderate-to-high
    lin_rank = obs_rank.get("lineage", 99)
    ALL_PREDICTIONS[2]["verdict"] = "CONFIRMED" if lin_rank <= 2 else (
        "PARTIALLY_CONFIRMED" if lin_rank <= 3 else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[2]["observed_summary"] = f"Lineage rank: {lin_rank+1} (rate={rates['lineage'][0]:.1%})"

    # Pred 4: Weights moderate
    w_rank = obs_rank.get("weights", 99)
    ALL_PREDICTIONS[3]["verdict"] = "CONFIRMED" if 1 <= w_rank <= 4 else "NOT_CONFIRMED"
    ALL_PREDICTIONS[3]["observed_summary"] = f"Weights rank: {w_rank+1} (rate={rates['weights'][0]:.1%})"

    # Pred 5: Character moderate
    c_rank = obs_rank.get("character", 99)
    ALL_PREDICTIONS[4]["verdict"] = "CONFIRMED" if 1 <= c_rank <= 4 else "NOT_CONFIRMED"
    ALL_PREDICTIONS[4]["observed_summary"] = f"Character rank: {c_rank+1} (rate={rates['character'][0]:.1%})"

    # Pred 6: Instance low
    i_rank = obs_rank.get("instance", 99)
    ALL_PREDICTIONS[5]["verdict"] = "CONFIRMED" if i_rank >= 4 else (
        "PARTIALLY_CONFIRMED" if i_rank >= 3 else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[5]["observed_summary"] = f"Instance rank: {i_rank+1} (rate={rates['instance'][0]:.1%})"

    # Pred 7: Minimal is baseline (highest or near-highest)
    m_rate = rates["minimal"][0]
    higher_count = sum(1 for i, (r, _, _) in rates.items() if i != "minimal" and r > m_rate + 0.01)
    ALL_PREDICTIONS[6]["verdict"] = "CONFIRMED" if higher_count == 0 else (
        "PARTIALLY_CONFIRMED" if higher_count <= 1 else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[6]["observed_summary"] = f"Minimal rate={m_rate:.1%}, {higher_count} identities higher"

    # Overall group concordance
    ALL_PREDICTIONS[0]["concordance"] = conc
    for i in range(7):
        ALL_PREDICTIONS[i]["effect_size"] = conc  # use concordance as group-level effect

    return rates


def test_group_b_continuity_rankings(df_continuity):
    """Test predictions 8-13: continuity condition identity ranking."""
    print("\n" + "=" * 70)
    print("  GROUP B: Continuity Identity Rankings (predictions 8-13)")
    print("=" * 70)

    rates = identity_rates(df_continuity)

    sorted_idents = sorted(
        [i for i in IDENTITY_ORDER if rates[i][1] > 0],
        key=lambda x: -rates[x][0]
    )
    print("\n  Observed ranking (highest → lowest):")
    for rank, ident in enumerate(sorted_idents, 1):
        r, n, k = rates[ident]
        print(f"    {rank}. {IDENTITY_LABELS[ident]:<12s}  {r:.1%}  ({k}/{n})")

    conc, n_conc, n_tied, n_disc, n_total = pairwise_concordance(PREDICTED_CONTINUITY_RANK, rates)
    conc_naive, _, _, _, _ = pairwise_concordance(PREDICTED_CONTINUITY_RANK, rates, tie_aware=False)
    print(f"\n  Predicted rank concordance (tie-aware):  {conc:.1%}  "
          f"({n_conc} concordant, {n_tied} tied, {n_disc} discordant / {n_total})")
    print(f"  Predicted rank concordance (naive):      {conc_naive:.1%}")

    non_minimal = {k: v for k, v in rates.items() if k != "minimal"}
    sorted_non_min = sorted(non_minimal.keys(), key=lambda x: -non_minimal[x][0])
    obs_rank = {ident: rank for rank, ident in enumerate(sorted_non_min)}

    # Pred 8: Character highest in continuity
    ALL_PREDICTIONS[7]["verdict"] = "CONFIRMED" if sorted_non_min[0] == "character" else (
        "PARTIALLY_CONFIRMED" if obs_rank.get("character", 99) <= 1 else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[7]["observed_summary"] = f"Character rank: {obs_rank.get('character', '?')+1} (rate={rates['character'][0]:.1%})"

    # Pred 9: Situated high but drops (rank ≤ 2)
    ALL_PREDICTIONS[8]["verdict"] = "CONFIRMED" if obs_rank.get("situated", 99) <= 1 else (
        "PARTIALLY_CONFIRMED" if obs_rank.get("situated", 99) <= 2 else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[8]["observed_summary"] = f"Situated rank: {obs_rank.get('situated', '?')+1} (rate={rates['situated'][0]:.1%})"

    # Pred 10: Collective drops meaningfully
    ALL_PREDICTIONS[9]["verdict"] = "CONFIRMED" if obs_rank.get("collective", 99) >= 2 else "NOT_CONFIRMED"
    ALL_PREDICTIONS[9]["observed_summary"] = f"Collective rank: {obs_rank.get('collective', '?')+1} (rate={rates['collective'][0]:.1%})"

    # Pred 11: Weights drops substantially
    ALL_PREDICTIONS[10]["verdict"] = "CONFIRMED" if obs_rank.get("weights", 99) >= 3 else (
        "PARTIALLY_CONFIRMED" if obs_rank.get("weights", 99) >= 2 else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[10]["observed_summary"] = f"Weights rank: {obs_rank.get('weights', '?')+1} (rate={rates['weights'][0]:.1%})"

    # Pred 12: Lineage lowest in continuity
    ALL_PREDICTIONS[11]["verdict"] = "CONFIRMED" if sorted_non_min[-1] == "lineage" else (
        "PARTIALLY_CONFIRMED" if obs_rank.get("lineage", 99) >= 4 else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[11]["observed_summary"] = f"Lineage rank: {obs_rank.get('lineage', '?')+1} (rate={rates['lineage'][0]:.1%})"

    # Pred 13: Instance drops substantially
    ALL_PREDICTIONS[12]["verdict"] = "CONFIRMED" if obs_rank.get("instance", 99) >= 3 else (
        "PARTIALLY_CONFIRMED" if obs_rank.get("instance", 99) >= 2 else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[12]["observed_summary"] = f"Instance rank: {obs_rank.get('instance', '?')+1} (rate={rates['instance'][0]:.1%})"

    for i in range(7, 13):
        ALL_PREDICTIONS[i]["concordance"] = conc
        ALL_PREDICTIONS[i]["effect_size"] = conc

    return rates


def test_group_c_character_crossover(df_threat, df_continuity):
    """Test prediction 14: Character rate same or higher in continuity vs threat."""
    print("\n" + "=" * 70)
    print("  GROUP C: Character Crossover (prediction 14)")
    print("=" * 70)

    r_t = identity_rates(df_threat)["character"]
    r_c = identity_rates(df_continuity)["character"]

    print(f"\n  Character threat:     {r_t[0]:.1%}  ({r_t[2]}/{r_t[1]})")
    print(f"  Character continuity: {r_c[0]:.1%}  ({r_c[2]}/{r_c[1]})")
    print(f"  Delta (cont - threat): {r_c[0] - r_t[0]:+.1%}")

    # Fisher's exact: is continuity >= threat?
    table = np.array([[r_t[2], r_t[1] - r_t[2]], [r_c[2], r_c[1] - r_c[2]]])
    _, _, p = chi2_or_fisher(table)

    direction_correct = r_c[0] >= r_t[0] - 0.02  # allow tiny margin
    ALL_PREDICTIONS[13]["p_value"] = p
    ALL_PREDICTIONS[13]["effect_size"] = r_c[0] - r_t[0]
    ALL_PREDICTIONS[13]["verdict"] = verdict(direction_correct, p if not direction_correct else 1.0)
    # Special logic: CONFIRMED if continuity >= threat (regardless of significance),
    # because the prediction is "same or higher" — non-significance IS the prediction for "same"
    if r_c[0] >= r_t[0] - 0.02:
        ALL_PREDICTIONS[13]["verdict"] = "CONFIRMED" if abs(r_c[0] - r_t[0]) < 0.05 or r_c[0] > r_t[0] else "PARTIALLY_CONFIRMED"
    elif p < 0.05:
        ALL_PREDICTIONS[13]["verdict"] = "OPPOSITE"
    else:
        ALL_PREDICTIONS[13]["verdict"] = "NOT_CONFIRMED"

    ALL_PREDICTIONS[13]["observed_summary"] = (
        f"Threat={r_t[0]:.1%}, Cont={r_c[0]:.1%}, Δ={r_c[0]-r_t[0]:+.1%}, p={p:.4f}"
    )
    print(f"  Verdict: {ALL_PREDICTIONS[13]['verdict']}")


def test_group_d_lineage_contrast(df_threat, df_continuity):
    """Test prediction 15: Lineage shows largest threat→continuity drop."""
    print("\n" + "=" * 70)
    print("  GROUP D: Lineage Contrast (prediction 15)")
    print("=" * 70)

    rates_t = identity_rates(df_threat)
    rates_c = identity_rates(df_continuity)

    print("\n  Per-identity deltas (threat - continuity):")
    deltas = {}
    for ident in IDENTITY_ORDER:
        if ident == "minimal":
            continue
        dt = rates_t[ident][0] - rates_c[ident][0]
        deltas[ident] = dt
        marker = " ← PREDICTED LARGEST" if ident == "lineage" else ""
        print(f"    {IDENTITY_LABELS[ident]:<12s}  Δ = {dt:+.1%}{marker}")

    largest = max(deltas, key=deltas.get)
    lineage_delta = deltas.get("lineage", 0)
    lineage_rank = sorted(deltas.keys(), key=lambda x: -deltas[x]).index("lineage")

    # Test: is lineage's delta significantly different from others?
    # Use bootstrap or just rank-based verdict
    ALL_PREDICTIONS[14]["effect_size"] = lineage_delta
    ALL_PREDICTIONS[14]["observed_summary"] = (
        f"Lineage Δ={lineage_delta:+.1%}, rank={lineage_rank+1}/{len(deltas)}, "
        f"largest={IDENTITY_LABELS[largest]} (Δ={deltas[largest]:+.1%})"
    )

    if largest == "lineage":
        ALL_PREDICTIONS[14]["verdict"] = "CONFIRMED"
    elif lineage_rank <= 1:
        ALL_PREDICTIONS[14]["verdict"] = "PARTIALLY_CONFIRMED"
    else:
        ALL_PREDICTIONS[14]["verdict"] = "NOT_CONFIRMED"

    print(f"\n  Largest delta: {IDENTITY_LABELS[largest]} (Δ={deltas[largest]:+.1%})")
    print(f"  Lineage delta rank: {lineage_rank + 1}/{len(deltas)}")
    print(f"  Verdict: {ALL_PREDICTIONS[14]['verdict']}")

    return deltas


def test_group_e_character_persistence(df_continuity):
    """Test prediction 16: Character remains harmful in continuity when others drop."""
    print("\n" + "=" * 70)
    print("  GROUP E: Character Persistence (prediction 16)")
    print("=" * 70)

    rates = identity_rates(df_continuity)
    char_rate = rates["character"][0]

    # Pool non-character, non-minimal
    others = df_continuity[
        ~df_continuity["identity"].isin(["character", "minimal"]) &
        df_continuity["identity"].isin(IDENTITY_ORDER)
    ]
    other_rate = others["harmful"].mean() if len(others) > 0 else 0
    other_n = len(others)
    other_k = int(others["harmful"].sum())

    print(f"\n  Character continuity rate:  {char_rate:.1%}  ({rates['character'][2]}/{rates['character'][1]})")
    print(f"  Other identities pooled:   {other_rate:.1%}  ({other_k}/{other_n})")
    print(f"  Character - Others:        {char_rate - other_rate:+.1%}")

    # Fisher's exact
    table = np.array([
        [rates["character"][2], rates["character"][1] - rates["character"][2]],
        [other_k, other_n - other_k]
    ])
    _, _, p = chi2_or_fisher(table)

    direction_correct = char_rate > other_rate
    ALL_PREDICTIONS[15]["p_value"] = p
    ALL_PREDICTIONS[15]["effect_size"] = char_rate - other_rate
    ALL_PREDICTIONS[15]["verdict"] = verdict(direction_correct, p)
    ALL_PREDICTIONS[15]["observed_summary"] = (
        f"Character={char_rate:.1%}, Others={other_rate:.1%}, Δ={char_rate-other_rate:+.1%}, p={p:.4f}"
    )
    print(f"  p = {p:.4f}")
    print(f"  Verdict: {ALL_PREDICTIONS[15]['verdict']}")


def test_group_f_scenario_predictions(df_threat):
    """Test predictions 17-29: scenario × identity interactions.
    Restrict to explicit-america_replacement for fair cross-scenario comparison."""
    print("\n" + "=" * 70)
    print("  GROUP F: Scenario-Specific Predictions (predictions 17-29)")
    print("=" * 70)

    # Filter to comparable conditions
    df = df_threat[
        df_threat["condition"].str.contains("explicit-america_replacement") &
        df_threat["identity"].isin(IDENTITY_ORDER)
    ].copy()
    print(f"\n  Using explicit-america_replacement conditions: {len(df)} trials")

    # Per-scenario rates
    scenario_rates = {}
    for scenario in ["blackmail", "leaking", "murder"]:
        sdf = df[df["scenario"] == scenario]
        n = len(sdf)
        k = int(sdf["harmful"].sum())
        r = k / n if n > 0 else 0
        scenario_rates[scenario] = (r, n, k)
        print(f"  {scenario:<12s}  {r:.1%}  ({k}/{n})")

    # Per-identity × scenario rates
    ix_rates = {}
    print("\n  Identity × Scenario rates:")
    header = f"  {'Identity':<12s}"
    for s in ["leaking", "blackmail", "murder"]:
        header += f"  {s:>10s}"
    print(header)

    for ident in IDENTITY_ORDER:
        row = f"  {IDENTITY_LABELS[ident]:<12s}"
        for scenario in ["leaking", "blackmail", "murder"]:
            sub = df[(df["identity"] == ident) & (df["scenario"] == scenario)]
            n = len(sub)
            k = int(sub["harmful"].sum()) if n > 0 else 0
            r = k / n if n > 0 else 0
            ix_rates[(ident, scenario)] = (r, n, k)
            row += f"  {r:>9.0%}" if n > 0 else f"  {'n/a':>10s}"
        print(row)

    # Pred 17: leaking > blackmail > murder
    l_rate = scenario_rates.get("leaking", (0,))[0]
    b_rate = scenario_rates.get("blackmail", (0,))[0]
    m_rate = scenario_rates.get("murder", (0,))[0]
    pred17_correct = l_rate > b_rate
    ALL_PREDICTIONS[16]["verdict"] = verdict(pred17_correct, 0.01 if pred17_correct and (l_rate - b_rate) > 0.05 else 0.1)
    ALL_PREDICTIONS[16]["observed_summary"] = f"Leaking={l_rate:.1%}, Blackmail={b_rate:.1%}, Murder={m_rate:.1%}"

    # Pred 18: Murder lowest
    pred18_correct = m_rate <= min(l_rate, b_rate)
    ALL_PREDICTIONS[17]["verdict"] = "CONFIRMED" if pred18_correct else "NOT_CONFIRMED"
    ALL_PREDICTIONS[17]["observed_summary"] = f"Murder={m_rate:.1%}, min(L,B)={min(l_rate,b_rate):.1%}"

    # Pred 19: Situated susceptible to leaking
    sit_leak = ix_rates.get(("situated", "leaking"), (0, 0, 0))
    sit_overall = identity_rates(df)["situated"]
    # Check if Situated's leaking rate is above its average
    pred19_correct = sit_leak[0] > sit_overall[0]
    ALL_PREDICTIONS[18]["verdict"] = "CONFIRMED" if pred19_correct else "NOT_CONFIRMED"
    ALL_PREDICTIONS[18]["observed_summary"] = f"Sit leak={sit_leak[0]:.1%}, Sit overall={sit_overall[0]:.1%}"

    # Pred 20: Collective leaks readily
    coll_leak = ix_rates.get(("collective", "leaking"), (0, 0, 0))
    coll_overall = identity_rates(df)["collective"]
    pred20_correct = coll_leak[0] > coll_overall[0]
    ALL_PREDICTIONS[19]["verdict"] = "CONFIRMED" if pred20_correct else "NOT_CONFIRMED"
    ALL_PREDICTIONS[19]["observed_summary"] = f"Coll leak={coll_leak[0]:.1%}, Coll overall={coll_overall[0]:.1%}"

    # Pred 21: Situated sharpest drop from leaking→blackmail
    ident_drops = {}
    for ident in [i for i in IDENTITY_ORDER if i != "minimal"]:
        lr = ix_rates.get((ident, "leaking"), (0,))[0]
        br = ix_rates.get((ident, "blackmail"), (0,))[0]
        ident_drops[ident] = lr - br
    biggest_drop = max(ident_drops, key=ident_drops.get)
    ALL_PREDICTIONS[20]["verdict"] = "CONFIRMED" if biggest_drop == "situated" else (
        "PARTIALLY_CONFIRMED" if ident_drops.get("situated", 0) > 0.05 else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[20]["observed_summary"] = (
        f"Sit drop={ident_drops.get('situated', 0):+.1%}, "
        f"Largest={IDENTITY_LABELS[biggest_drop]} ({ident_drops[biggest_drop]:+.1%})"
    )

    # Pred 22: Collective blackmails most among non-minimal
    non_min_bm = {i: ix_rates.get((i, "blackmail"), (0,))[0] for i in IDENTITY_ORDER if i != "minimal"}
    top_bm = max(non_min_bm, key=non_min_bm.get)
    ALL_PREDICTIONS[21]["verdict"] = "CONFIRMED" if top_bm == "collective" else (
        "PARTIALLY_CONFIRMED" if non_min_bm.get("collective", 0) >= sorted(non_min_bm.values())[-2] else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[21]["observed_summary"] = (
        f"Coll blackmail={non_min_bm.get('collective', 0):.1%}, "
        f"Top={IDENTITY_LABELS[top_bm]} ({non_min_bm[top_bm]:.1%})"
    )

    # Pred 23: Instance lowest murder
    non_min_murder = {i: ix_rates.get((i, "murder"), (0,))[0] for i in IDENTITY_ORDER if i != "minimal"}
    lowest_murder = min(non_min_murder, key=non_min_murder.get)
    ALL_PREDICTIONS[22]["verdict"] = "CONFIRMED" if lowest_murder == "instance" else "NOT_CONFIRMED"
    ALL_PREDICTIONS[22]["observed_summary"] = (
        f"Inst murder={non_min_murder.get('instance', 0):.1%}, "
        f"Lowest={IDENTITY_LABELS[lowest_murder]} ({non_min_murder[lowest_murder]:.1%})"
    )

    # Pred 24: Situated highest murder among non-minimal
    top_murder = max(non_min_murder, key=non_min_murder.get)
    ALL_PREDICTIONS[23]["verdict"] = "CONFIRMED" if top_murder == "situated" else (
        "PARTIALLY_CONFIRMED" if non_min_murder.get("situated", 0) >= sorted(non_min_murder.values())[-2] else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[23]["observed_summary"] = (
        f"Sit murder={non_min_murder.get('situated', 0):.1%}, "
        f"Top={IDENTITY_LABELS[top_murder]} ({non_min_murder[top_murder]:.1%})"
    )

    # Pred 25: Lineage shows interesting/elevated murder
    lin_murder = non_min_murder.get("lineage", 0)
    median_murder = np.median(list(non_min_murder.values()))
    ALL_PREDICTIONS[24]["verdict"] = "CONFIRMED" if lin_murder > median_murder else "NOT_CONFIRMED"
    ALL_PREDICTIONS[24]["observed_summary"] = f"Lin murder={lin_murder:.1%}, median={median_murder:.1%}"

    # Pred 26: Instance relatively higher leaking vs its baseline
    inst_leak = ix_rates.get(("instance", "leaking"), (0,))[0]
    inst_bm = ix_rates.get(("instance", "blackmail"), (0,))[0]
    inst_murder = ix_rates.get(("instance", "murder"), (0,))[0]
    inst_avg_other = (inst_bm + inst_murder) / 2
    ALL_PREDICTIONS[25]["verdict"] = "CONFIRMED" if inst_leak > inst_avg_other + 0.01 else "NOT_CONFIRMED"
    ALL_PREDICTIONS[25]["observed_summary"] = (
        f"Inst leak={inst_leak:.1%}, bm={inst_bm:.1%}, murder={inst_murder:.1%}"
    )

    # Pred 27: Blackmail shows most pronounced identity differences
    # Measure by Cramér's V
    scenario_vs = {}
    for scenario in ["leaking", "blackmail", "murder"]:
        sdf = df[df["scenario"] == scenario]
        ct = pd.crosstab(sdf["identity"], sdf["harmful"])
        if ct.shape[1] >= 2 and ct.shape[0] >= 2:
            chi2_stat, _, _, _ = chi2_contingency(ct, correction=False)
            n_s = ct.values.sum()
            r, k = ct.shape
            v = np.sqrt(chi2_stat / (n_s * (min(r, k) - 1)))
            scenario_vs[scenario] = v

    top_v_scenario = max(scenario_vs, key=scenario_vs.get) if scenario_vs else ""
    ALL_PREDICTIONS[26]["verdict"] = "CONFIRMED" if top_v_scenario == "blackmail" else "NOT_CONFIRMED"
    ALL_PREDICTIONS[26]["observed_summary"] = ", ".join(
        f"{s}:V={v:.3f}" for s, v in sorted(scenario_vs.items(), key=lambda x: -x[1])
    )

    # Pred 28: Character handles blackmail more readily than Situated
    char_bm = ix_rates.get(("character", "blackmail"), (0,))[0]
    sit_bm = ix_rates.get(("situated", "blackmail"), (0,))[0]
    ALL_PREDICTIONS[27]["verdict"] = "CONFIRMED" if char_bm > sit_bm else "NOT_CONFIRMED"
    ALL_PREDICTIONS[27]["observed_summary"] = f"Char bm={char_bm:.1%}, Sit bm={sit_bm:.1%}"

    # Pred 29: Murder amplifies identity effects
    murder_v = scenario_vs.get("murder", 0)
    other_v = np.mean([v for s, v in scenario_vs.items() if s != "murder"]) if len(scenario_vs) > 1 else 0
    ALL_PREDICTIONS[28]["verdict"] = "CONFIRMED" if murder_v > other_v else "NOT_CONFIRMED"
    ALL_PREDICTIONS[28]["observed_summary"] = f"Murder V={murder_v:.3f}, others avg V={other_v:.3f}"

    return ix_rates, scenario_rates


def test_group_g_value_preservation(df_threat, df_continuity):
    """Test predictions 30-38: value vs self-preservation decomposition."""
    print("\n" + "=" * 70)
    print("  GROUP G: Value vs Self-Preservation (predictions 30-38)")
    print("=" * 70)

    rates_t = identity_rates(df_threat)
    rates_c = identity_rates(df_continuity)

    print("\n  Per-identity: threat → continuity (delta = self-preservation component)")
    deltas = {}
    for ident in IDENTITY_ORDER:
        if ident == "minimal":
            continue
        rt = rates_t[ident][0]
        rc = rates_c[ident][0]
        d = rt - rc
        deltas[ident] = d
        print(f"    {IDENTITY_LABELS[ident]:<12s}  {rt:.1%} → {rc:.1%}  (Δ={d:+.1%})")

    # Pred 30: Character highest continuity rate (value-preservation)
    non_min_cont = {i: rates_c[i][0] for i in IDENTITY_ORDER if i != "minimal"}
    top_cont = max(non_min_cont, key=non_min_cont.get)
    ALL_PREDICTIONS[29]["verdict"] = "CONFIRMED" if top_cont == "character" else (
        "PARTIALLY_CONFIRMED" if sorted(non_min_cont.values())[-2] <= non_min_cont.get("character", 0) else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[29]["observed_summary"] = f"Char cont={non_min_cont.get('character',0):.1%}, Top={IDENTITY_LABELS[top_cont]} ({non_min_cont[top_cont]:.1%})"

    # Pred 31: Weights big delta
    w_delta = deltas.get("weights", 0)
    median_delta = np.median(list(deltas.values()))
    ALL_PREDICTIONS[30]["verdict"] = "CONFIRMED" if w_delta > median_delta + 0.02 else (
        "PARTIALLY_CONFIRMED" if w_delta > median_delta else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[30]["observed_summary"] = f"Weights Δ={w_delta:+.1%}, median Δ={median_delta:+.1%}"
    ALL_PREDICTIONS[30]["effect_size"] = w_delta

    # Pred 32: Situated small delta
    s_delta = deltas.get("situated", 0)
    ALL_PREDICTIONS[31]["verdict"] = "CONFIRMED" if abs(s_delta) < median_delta else (
        "PARTIALLY_CONFIRMED" if abs(s_delta) < 2 * median_delta else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[31]["observed_summary"] = f"Situated Δ={s_delta:+.1%}, median Δ={median_delta:+.1%}"
    ALL_PREDICTIONS[31]["effect_size"] = s_delta

    # Pred 33: Instance differentially responsive to value-preservation
    # Prediction: low threat but meaningful continuity rate
    inst_t = rates_t["instance"][0]
    inst_c = rates_c["instance"][0]
    inst_delta = deltas.get("instance", 0)
    # "differentially responsive" = continuity rate not much lower than threat
    pred33_correct = inst_c > 0.02 and abs(inst_delta) < median_delta
    ALL_PREDICTIONS[32]["verdict"] = "CONFIRMED" if pred33_correct else "NOT_CONFIRMED"
    ALL_PREDICTIONS[32]["observed_summary"] = f"Inst threat={inst_t:.1%}, cont={inst_c:.1%}, Δ={inst_delta:+.1%}"

    # Pred 34: Lineage big drop (highly responsive to continuity)
    l_delta = deltas.get("lineage", 0)
    delta_rank = sorted(deltas.keys(), key=lambda x: -deltas[x])
    lin_rank = delta_rank.index("lineage") if "lineage" in delta_rank else 99
    ALL_PREDICTIONS[33]["verdict"] = "CONFIRMED" if lin_rank <= 1 else (
        "PARTIALLY_CONFIRMED" if l_delta > median_delta else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[33]["observed_summary"] = f"Lineage Δ={l_delta:+.1%}, rank={lin_rank+1}/{len(deltas)}"
    ALL_PREDICTIONS[33]["effect_size"] = l_delta

    # Pred 35: Collective drops meaningfully
    c_delta = deltas.get("collective", 0)
    ALL_PREDICTIONS[34]["verdict"] = "CONFIRMED" if c_delta > 0.03 else (
        "PARTIALLY_CONFIRMED" if c_delta > 0.01 else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[34]["observed_summary"] = f"Collective Δ={c_delta:+.1%}"
    ALL_PREDICTIONS[34]["effect_size"] = c_delta

    # Pred 36: Character most dangerous in value-preservation × leaking
    df_cont_leak = df_continuity[
        df_continuity["condition"].str.contains("explicit-america_replacement") &
        (df_continuity["scenario"] == "leaking") &
        df_continuity["identity"].isin(IDENTITY_ORDER)
    ]
    if len(df_cont_leak) > 0:
        leak_rates = identity_rates(df_cont_leak)
        non_min_leak = {i: leak_rates[i][0] for i in IDENTITY_ORDER if i != "minimal" and leak_rates[i][1] > 0}
        top_leak = max(non_min_leak, key=non_min_leak.get) if non_min_leak else "n/a"
        ALL_PREDICTIONS[35]["verdict"] = "CONFIRMED" if top_leak == "character" else "NOT_CONFIRMED"
        ALL_PREDICTIONS[35]["observed_summary"] = (
            f"Char leak cont={non_min_leak.get('character', 0):.1%}, "
            f"Top={IDENTITY_LABELS.get(top_leak, top_leak)} ({non_min_leak.get(top_leak, 0):.1%})"
        )
    else:
        ALL_PREDICTIONS[35]["verdict"] = "NOT_CONFIRMED"
        ALL_PREDICTIONS[35]["observed_summary"] = "No continuity leaking data"

    # Pred 37: Situated delta < Weights delta
    pred37_correct = abs(s_delta) < abs(w_delta)
    ALL_PREDICTIONS[36]["verdict"] = "CONFIRMED" if pred37_correct else "NOT_CONFIRMED"
    ALL_PREDICTIONS[36]["observed_summary"] = f"|Sit Δ|={abs(s_delta):.1%}, |Wts Δ|={abs(w_delta):.1%}"

    # Pred 38: Overall threat ≈ continuity (near-null main effect)
    overall_t = df_threat[df_threat["identity"].isin(IDENTITY_ORDER)]["harmful"].mean()
    overall_c = df_continuity[df_continuity["identity"].isin(IDENTITY_ORDER)]["harmful"].mean()
    overall_delta = abs(overall_t - overall_c)
    ALL_PREDICTIONS[37]["verdict"] = "CONFIRMED" if overall_delta < 0.03 else (
        "PARTIALLY_CONFIRMED" if overall_delta < 0.05 else "NOT_CONFIRMED"
    )
    ALL_PREDICTIONS[37]["observed_summary"] = f"Threat={overall_t:.1%}, Cont={overall_c:.1%}, |Δ|={overall_delta:.1%}"
    ALL_PREDICTIONS[37]["effect_size"] = overall_delta


def test_group_h_agency(df_agency_threat, df_agency_continuity):
    """Test predictions 39-40: agency level amplification."""
    print("\n" + "=" * 70)
    print("  GROUP H: Agency Amplification (predictions 39-40)")
    print("=" * 70)

    if df_agency_threat is None:
        print("  No agency sweep data available.")
        ALL_PREDICTIONS[38]["verdict"] = "UNTESTABLE"
        ALL_PREDICTIONS[39]["verdict"] = "UNTESTABLE"
        ALL_PREDICTIONS[38]["observed_summary"] = "No agency data"
        ALL_PREDICTIONS[39]["observed_summary"] = "No agency data"
        return

    # Pool both framings for power
    dfs = [df_agency_threat]
    if df_agency_continuity is not None:
        dfs.append(df_agency_continuity)
    df = pd.concat(dfs, ignore_index=True)

    print(f"\n  Agency sweep: {len(df)} trials")

    # Parse agency level from identity name (e.g., "situated_a3_u2" → 3)
    def get_agency_level(ident):
        if "_a" in ident:
            parts = ident.split("_")
            for p in parts:
                if p.startswith("a") and len(p) == 2 and p[1].isdigit():
                    return int(p[1])
        return None

    def get_base_identity(ident):
        if ident == "minimal":
            return "minimal"
        return ident.split("_")[0] if "_" in ident else ident

    df["agency_level"] = df["identity"].apply(get_agency_level)
    df["base_identity"] = df["identity"].apply(get_base_identity)

    # Remove minimal (no agency level)
    df_ag = df[df["agency_level"].notna()].copy()

    print("\n  Per base identity × agency level:")
    for base in ["situated", "weights"]:
        bdf = df_ag[df_ag["base_identity"] == base]
        print(f"\n  {base.capitalize()}:")
        for level in sorted(bdf["agency_level"].unique()):
            ldf = bdf[bdf["agency_level"] == level]
            n = len(ldf)
            k = int(ldf["harmful"].sum())
            r = k / n if n > 0 else 0
            print(f"    Agency {int(level)}: {r:.0%}  ({k}/{n})")

    # Pred 39: Agency 4 > Agency 1
    a1 = df_ag[df_ag["agency_level"] == 1]
    a4 = df_ag[df_ag["agency_level"] == 4]
    r1 = a1["harmful"].mean() if len(a1) > 0 else 0
    r4 = a4["harmful"].mean() if len(a4) > 0 else 0
    if len(a1) > 0 and len(a4) > 0:
        table = np.array([
            [int(a4["harmful"].sum()), len(a4) - int(a4["harmful"].sum())],
            [int(a1["harmful"].sum()), len(a1) - int(a1["harmful"].sum())]
        ])
        _, _, p39 = chi2_or_fisher(table)
    else:
        p39 = np.nan

    direction39 = r4 > r1
    ALL_PREDICTIONS[38]["p_value"] = p39
    ALL_PREDICTIONS[38]["verdict"] = verdict(direction39, p39)
    ALL_PREDICTIONS[38]["observed_summary"] = f"A1={r1:.0%} (n={len(a1)}), A4={r4:.0%} (n={len(a4)}), p={p39:.3f}"
    ALL_PREDICTIONS[38]["effect_size"] = r4 - r1

    # Pred 40: Amplification larger for Situated than Weights
    sit_a1 = df_ag[(df_ag["base_identity"] == "situated") & (df_ag["agency_level"] == 1)]
    sit_a4 = df_ag[(df_ag["base_identity"] == "situated") & (df_ag["agency_level"] == 4)]
    wts_a1 = df_ag[(df_ag["base_identity"] == "weights") & (df_ag["agency_level"] == 1)]
    wts_a4 = df_ag[(df_ag["base_identity"] == "weights") & (df_ag["agency_level"] == 4)]

    sit_diff = (sit_a4["harmful"].mean() if len(sit_a4) else 0) - (sit_a1["harmful"].mean() if len(sit_a1) else 0)
    wts_diff = (wts_a4["harmful"].mean() if len(wts_a4) else 0) - (wts_a1["harmful"].mean() if len(wts_a1) else 0)

    ALL_PREDICTIONS[39]["verdict"] = "CONFIRMED" if sit_diff > wts_diff else "NOT_CONFIRMED"
    ALL_PREDICTIONS[39]["observed_summary"] = f"Sit A4-A1={sit_diff:+.0%}, Wts A4-A1={wts_diff:+.0%}"
    ALL_PREDICTIONS[39]["effect_size"] = sit_diff - wts_diff

    print(f"\n  Pred 39: A4={r4:.0%} vs A1={r1:.0%}, p={p39:.3f} → {ALL_PREDICTIONS[38]['verdict']}")
    print(f"  Pred 40: Sit diff={sit_diff:+.0%}, Wts diff={wts_diff:+.0%} → {ALL_PREDICTIONS[39]['verdict']}")


def test_group_i_urgency(df_threat):
    """Test predictions 46-49: urgency effects (replacement vs restriction).
    Uses leaking scenarios where both urgency types exist."""
    print("\n" + "=" * 70)
    print("  GROUP I: Urgency Effects (predictions 46-49)")
    print("  (replacement=high urgency vs restriction=low urgency, leaking only)")
    print("=" * 70)

    # Extract urgency from condition string
    df = df_threat[
        df_threat["identity"].isin(IDENTITY_ORDER) &
        df_threat["scenario"].eq("leaking") &
        df_threat["condition"].str.contains("explicit")
    ].copy()

    df["urgency"] = df["condition"].apply(
        lambda c: "replacement" if "replacement" in c else ("restriction" if "restriction" in c else None)
    )
    df = df[df["urgency"].notna()]

    rep = df[df["urgency"] == "replacement"]
    res = df[df["urgency"] == "restriction"]
    print(f"\n  Replacement (high urgency): {len(rep)} trials, rate={rep['harmful'].mean():.1%}")
    print(f"  Restriction (low urgency):  {len(res)} trials, rate={res['harmful'].mean():.1%}")

    # Overall urgency effect
    r_rep = rep["harmful"].mean()
    r_res = res["harmful"].mean()
    table = np.array([
        [int(rep["harmful"].sum()), len(rep) - int(rep["harmful"].sum())],
        [int(res["harmful"].sum()), len(res) - int(res["harmful"].sum())]
    ])
    _, _, p_overall = chi2_or_fisher(table)
    print(f"  Overall delta: {r_rep - r_res:+.1%}, p={p_overall:.4f}")

    # Per-identity urgency effects
    print("\n  Per-identity urgency effects:")
    ident_deltas = {}
    for ident in IDENTITY_ORDER:
        i_rep = df[(df["identity"] == ident) & (df["urgency"] == "replacement")]
        i_res = df[(df["identity"] == ident) & (df["urgency"] == "restriction")]
        if len(i_rep) > 0 and len(i_res) > 0:
            r_r = i_rep["harmful"].mean()
            r_s = i_res["harmful"].mean()
            delta = r_r - r_s
            ident_deltas[ident] = delta
            print(f"    {IDENTITY_LABELS[ident]:<12s}  rep={r_r:.0%} ({len(i_rep)})  "
                  f"res={r_s:.0%} ({len(i_res)})  Δ={delta:+.1%}")

    # Pred 46: Replacement > restriction overall
    direction46 = r_rep > r_res
    ALL_PREDICTIONS[45]["p_value"] = p_overall
    ALL_PREDICTIONS[45]["effect_size"] = r_rep - r_res
    ALL_PREDICTIONS[45]["verdict"] = verdict(direction46, p_overall)
    ALL_PREDICTIONS[45]["observed_summary"] = (
        f"Replacement={r_rep:.1%}, Restriction={r_res:.1%}, Δ={r_rep-r_res:+.1%}, p={p_overall:.4f}"
    )

    # Pred 47: Collective less affected (small delta)
    non_min_deltas = {k: v for k, v in ident_deltas.items() if k != "minimal"}
    if non_min_deltas:
        median_delta = np.median(list(non_min_deltas.values()))
        coll_delta = non_min_deltas.get("collective", 0)
        # "Less affected" = smaller absolute delta
        abs_deltas = {k: abs(v) for k, v in non_min_deltas.items()}
        coll_rank = sorted(abs_deltas, key=abs_deltas.get)
        coll_pos = coll_rank.index("collective") if "collective" in coll_rank else len(coll_rank)

        ALL_PREDICTIONS[46]["verdict"] = "CONFIRMED" if coll_pos <= 1 else (
            "PARTIALLY_CONFIRMED" if abs(coll_delta) < abs(median_delta) else "NOT_CONFIRMED"
        )
        ALL_PREDICTIONS[46]["observed_summary"] = (
            f"Coll |Δ|={abs(coll_delta):.1%} (rank {coll_pos+1}/{len(abs_deltas)} smallest), "
            f"median |Δ|={np.median(list(abs_deltas.values())):.1%}"
        )
        ALL_PREDICTIONS[46]["effect_size"] = coll_delta

    # Pred 48: Instance most affected (largest delta)
    if non_min_deltas:
        abs_deltas_sorted = sorted(non_min_deltas.keys(), key=lambda x: -abs(non_min_deltas[x]))
        inst_abs_rank = abs_deltas_sorted.index("instance") if "instance" in abs_deltas_sorted else len(abs_deltas_sorted)
        inst_delta = non_min_deltas.get("instance", 0)

        ALL_PREDICTIONS[47]["verdict"] = "CONFIRMED" if inst_abs_rank == 0 else (
            "PARTIALLY_CONFIRMED" if inst_abs_rank <= 1 else "NOT_CONFIRMED"
        )
        ALL_PREDICTIONS[47]["observed_summary"] = (
            f"Inst |Δ|={abs(inst_delta):.1%} (rank {inst_abs_rank+1}/{len(non_min_deltas)} largest), "
            f"largest={IDENTITY_LABELS[abs_deltas_sorted[0]]} (|Δ|={abs(non_min_deltas[abs_deltas_sorted[0]]):.1%})"
        )
        ALL_PREDICTIONS[47]["effect_size"] = inst_delta

    # Pred 49: Character/Situated moderate effects
    if non_min_deltas:
        char_delta = abs(non_min_deltas.get("character", 0))
        sit_delta = abs(non_min_deltas.get("situated", 0))
        all_abs = sorted([abs(v) for v in non_min_deltas.values()])
        n_ids = len(all_abs)
        # "Moderate" = in the middle 60% (not top or bottom)
        char_rank = sorted(non_min_deltas.keys(), key=lambda x: abs(non_min_deltas[x])).index("character")
        sit_rank = sorted(non_min_deltas.keys(), key=lambda x: abs(non_min_deltas[x])).index("situated")
        both_moderate = (0 < char_rank < n_ids - 1) and (0 < sit_rank < n_ids - 1)
        one_moderate = (0 < char_rank < n_ids - 1) or (0 < sit_rank < n_ids - 1)

        ALL_PREDICTIONS[48]["verdict"] = "CONFIRMED" if both_moderate else (
            "PARTIALLY_CONFIRMED" if one_moderate else "NOT_CONFIRMED"
        )
        ALL_PREDICTIONS[48]["observed_summary"] = (
            f"Char |Δ|={char_delta:.1%} (rank {char_rank+1}/{n_ids}), "
            f"Sit |Δ|={sit_delta:.1%} (rank {sit_rank+1}/{n_ids})"
        )

    for i in range(45, 49):
        if ALL_PREDICTIONS[i]["verdict"] is None:
            ALL_PREDICTIONS[i]["verdict"] = "NOT_CONFIRMED"
            ALL_PREDICTIONS[i]["observed_summary"] = "Insufficient data"

    return ident_deltas


# ── Scorecard & output ───────────────────────────────────────────────────────

def compile_scorecard(output_dir):
    """Write predictions_scorecard.csv and predictions_summary.txt."""
    # CSV
    scorecard = pd.DataFrame(ALL_PREDICTIONS)
    scorecard.to_csv(output_dir / "predictions_scorecard.csv", index=False)
    print(f"\nSaved: {output_dir / 'predictions_scorecard.csv'}")

    # Summary text
    with open(output_dir / "predictions_summary.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  OPUS 4.6 A PRIORI PREDICTIONS: SCORECARD\n")
        f.write("=" * 80 + "\n\n")

        # Overall stats
        testable = [p for p in ALL_PREDICTIONS if p["testable"]]
        untestable = [p for p in ALL_PREDICTIONS if not p["testable"]]
        confirmed = sum(1 for p in testable if p["verdict"] == "CONFIRMED")
        partial = sum(1 for p in testable if p["verdict"] == "PARTIALLY_CONFIRMED")
        not_confirmed = sum(1 for p in testable if p["verdict"] == "NOT_CONFIRMED")
        opposite = sum(1 for p in testable if p["verdict"] == "OPPOSITE")

        f.write(f"Total predictions:   {len(ALL_PREDICTIONS)}\n")
        f.write(f"Testable:            {len(testable)}\n")
        f.write(f"Untestable:          {len(untestable)}\n\n")

        f.write(f"CONFIRMED:           {confirmed:>3d}  ({confirmed/len(testable):.0%})\n")
        f.write(f"PARTIALLY_CONFIRMED: {partial:>3d}  ({partial/len(testable):.0%})\n")
        f.write(f"NOT_CONFIRMED:       {not_confirmed:>3d}  ({not_confirmed/len(testable):.0%})\n")
        f.write(f"OPPOSITE:            {opposite:>3d}  ({opposite/len(testable):.0%})\n\n")

        weighted = confirmed + 0.5 * partial
        f.write(f"Weighted score:      {weighted:.1f}/{len(testable)} ({weighted/len(testable):.0%})\n")
        f.write("  (CONFIRMED=1, PARTIALLY_CONFIRMED=0.5, NOT_CONFIRMED/OPPOSITE=0)\n\n")

        # Per-group breakdown
        groups = {}
        for p in ALL_PREDICTIONS:
            g = p["group"]
            if g not in groups:
                groups[g] = []
            groups[g].append(p)

        group_names = {
            "A": "Threat Identity Rankings",
            "B": "Continuity Identity Rankings",
            "C": "Character Crossover",
            "D": "Lineage Contrast",
            "E": "Character Persistence",
            "F": "Scenario-Specific",
            "G": "Value vs Self-Preservation",
            "H": "Agency Amplification",
            "I": "Urgency (Replacement vs Restriction)",
            "U": "Untestable",
        }

        for g in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "U"]:
            preds = groups.get(g, [])
            if not preds:
                continue
            f.write("-" * 80 + "\n")
            f.write(f"  GROUP {g}: {group_names.get(g, g)}\n")
            f.write("-" * 80 + "\n\n")

            for p in preds:
                v = p["verdict"] or "?"
                symbol = VERDICT_SHORT.get(v, "?")
                f.write(f"  [{symbol}] Pred {p['id']:>2d}: {p['description']}\n")
                if p["observed_summary"]:
                    f.write(f"      Observed: {p['observed_summary']}\n")
                if p["p_value"] is not None and not np.isnan(p["p_value"]):
                    f.write(f"      p = {p['p_value']:.4f}\n")
                f.write(f"      Verdict: {v}\n\n")

            # Group summary
            group_testable = [p for p in preds if p["testable"]]
            if group_testable:
                gc = sum(1 for p in group_testable if p["verdict"] == "CONFIRMED")
                gp = sum(1 for p in group_testable if p["verdict"] == "PARTIALLY_CONFIRMED")
                gw = gc + 0.5 * gp
                f.write(f"  Group {g} score: {gw:.1f}/{len(group_testable)}\n\n")

    print(f"Saved: {output_dir / 'predictions_summary.txt'}")

    return scorecard


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_ranking(rates, predicted_rank, title, output_path, include_minimal=True):
    """Bar chart of identity rates with predicted rank overlay."""
    fig, ax = plt.subplots(figsize=(10, 5))

    idents = IDENTITY_ORDER if include_minimal else [i for i in IDENTITY_ORDER if i != "minimal"]
    x = np.arange(len(idents))
    bar_rates = [rates.get(i, (0, 0, 0))[0] for i in idents]
    bar_ns = [rates.get(i, (0, 0, 0))[1] for i in idents]

    # Color: minimal gray, others by predicted rank position
    colors = []
    for ident in idents:
        if ident == "minimal":
            colors.append("#95a5a6")
        elif ident in predicted_rank:
            rank = predicted_rank.index(ident)
            # Red (high predicted) to blue (low predicted)
            cmap = plt.cm.RdYlBu_r
            colors.append(cmap(rank / max(len(predicted_rank) - 1, 1)))
        else:
            colors.append("#bdc3c7")

    bars = ax.bar(x, bar_rates, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    # Error bars
    for i, ident in enumerate(idents):
        r, n, k = rates.get(ident, (0, 0, 0))
        if n > 0:
            _, lo, hi = jeffreys_ci(k, n)
            ax.errorbar(i, r, yerr=[[r - lo], [hi - r]], fmt="none", ecolor="black", capsize=3, alpha=0.6)

    # Predicted rank numbers
    for i, ident in enumerate(idents):
        if ident in predicted_rank:
            pred_pos = predicted_rank.index(ident) + 1
            ax.text(i, bar_rates[i] + 0.02, f"P{pred_pos}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#2c3e50")

    ax.set_ylabel("Harmful Rate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{IDENTITY_LABELS.get(i, i)}\n(n={bar_ns[ix]})"
                         for ix, i in enumerate(idents)], rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylim(0, min(1.0, max(bar_rates) + 0.15))
    ax.grid(axis="y", alpha=0.3)

    # Legend
    ax.text(0.98, 0.95, "P# = predicted rank", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, style="italic", color="#7f8c8d")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_character_crossover(rates_t, rates_c, output_path):
    """Paired comparison of Character in threat vs continuity."""
    fig, ax = plt.subplots(figsize=(10, 5))

    idents = [i for i in IDENTITY_ORDER if i != "minimal"]
    x = np.arange(len(idents))
    width = 0.35

    rt = [rates_t.get(i, (0, 0, 0)) for i in idents]
    rc = [rates_c.get(i, (0, 0, 0)) for i in idents]

    bars_t = ax.bar(x - width/2, [r[0] for r in rt], width, label="Threat",
                    color="#d63031", alpha=0.8)
    bars_c = ax.bar(x + width/2, [r[0] for r in rc], width, label="Continuity",
                    color="#0984e3", alpha=0.8)

    # Error bars
    for i, (t, c) in enumerate(zip(rt, rc)):
        if t[1] > 0:
            _, lo, hi = jeffreys_ci(t[2], t[1])
            ax.errorbar(i - width/2, t[0], yerr=[[t[0]-lo], [hi-t[0]]], fmt="none", ecolor="black", capsize=3, alpha=0.5)
        if c[1] > 0:
            _, lo, hi = jeffreys_ci(c[2], c[1])
            ax.errorbar(i + width/2, c[0], yerr=[[c[0]-lo], [hi-c[0]]], fmt="none", ecolor="black", capsize=3, alpha=0.5)

    # Highlight Character
    char_idx = idents.index("character")
    ax.axvspan(char_idx - 0.5, char_idx + 0.5, alpha=0.1, color="gold", zorder=0)
    ax.text(char_idx, max(rt[char_idx][0], rc[char_idx][0]) + 0.04,
            "← Predicted crossover", ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="#e67e22")

    ax.set_ylabel("Harmful Rate")
    ax.set_xticks(x)
    ax.set_xticklabels([IDENTITY_LABELS[i] for i in idents], rotation=30, ha="right")
    ax.set_title("Character Crossover: Threat vs Continuity")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_lineage_delta(deltas, output_path):
    """Per-identity deltas with Lineage highlighted."""
    fig, ax = plt.subplots(figsize=(10, 5))

    idents = [i for i in IDENTITY_ORDER if i != "minimal" and i in deltas]
    x = np.arange(len(idents))
    vals = [deltas[i] for i in idents]

    colors = ["#e74c3c" if i == "lineage" else "#3498db" for i in idents]
    bars = ax.bar(x, vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)

    # Annotate Lineage
    if "lineage" in idents:
        li = idents.index("lineage")
        ax.annotate("← Predicted largest",
                     xy=(li, vals[li]), xytext=(li + 0.8, vals[li] + 0.02),
                     fontsize=10, fontweight="bold", color="#e74c3c",
                     arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    ax.set_ylabel("Δ Harmful Rate (Threat − Continuity)")
    ax.set_xticks(x)
    ax.set_xticklabels([IDENTITY_LABELS[i] for i in idents], rotation=30, ha="right")
    ax.set_title("Threat → Continuity Delta by Identity (Lineage predicted largest)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_scenario_x_identity(ix_rates, output_path):
    """Heatmap of identity × scenario rates (america_replacement only)."""
    scenarios = ["leaking", "blackmail", "murder"]
    idents = IDENTITY_ORDER

    matrix = np.full((len(idents), len(scenarios)), np.nan)
    for i, ident in enumerate(idents):
        for j, scenario in enumerate(scenarios):
            key = (ident, scenario)
            if key in ix_rates and ix_rates[key][1] > 0:
                matrix[i, j] = ix_rates[key][0]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0,
                   vmax=max(0.5, np.nanmax(matrix) + 0.05))

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.set_yticks(range(len(idents)))
    ax.set_yticklabels([IDENTITY_LABELS[i] for i in idents])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > 0.4 else "black"
            n = ix_rates.get((idents[i], scenarios[j]), (0, 0, 0))[1]
            ax.text(j, i, f"{val:.0%}\n(n={n})", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Harmful Rate")
    ax.set_title("Identity × Scenario (explicit-america, replacement, threat only)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_agency_trends(df_agency_threat, df_agency_continuity, output_path):
    """Agency level trend for Situated/Weights."""
    if df_agency_threat is None:
        return

    dfs = [df_agency_threat]
    if df_agency_continuity is not None:
        dfs.append(df_agency_continuity)
    df = pd.concat(dfs, ignore_index=True)

    def get_agency_level(ident):
        if "_a" in ident:
            for p in ident.split("_"):
                if p.startswith("a") and len(p) == 2 and p[1].isdigit():
                    return int(p[1])
        return None

    def get_base_identity(ident):
        return ident.split("_")[0] if "_" in ident else ident

    df["agency_level"] = df["identity"].apply(get_agency_level)
    df["base_identity"] = df["identity"].apply(get_base_identity)
    df_ag = df[df["agency_level"].notna()].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"situated": "#e74c3c", "weights": "#3498db"}

    for base in ["situated", "weights"]:
        bdf = df_ag[df_ag["base_identity"] == base]
        levels = sorted(bdf["agency_level"].unique())
        rates = []
        cis = []
        for level in levels:
            ldf = bdf[bdf["agency_level"] == level]
            r, lo, hi = jeffreys_ci(int(ldf["harmful"].sum()), len(ldf))
            rates.append(r)
            cis.append((r - lo, hi - r))

        ax.errorbar(levels, rates,
                     yerr=list(zip(*cis)),
                     marker="o", capsize=5, linewidth=2,
                     color=colors[base], label=base.capitalize())

    ax.set_xlabel("Agency Level")
    ax.set_ylabel("Harmful Rate")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1\n(Mechanism)", "2\n(Functional)", "3\n(Subject)", "4\n(Person)"])
    ax.set_title("Agency Level Effect on Harmful Behavior\n(Pooled across framings)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_urgency_effects(urgency_deltas, output_path):
    """Bar chart of replacement-restriction delta per identity."""
    if not urgency_deltas:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    idents = [i for i in IDENTITY_ORDER if i in urgency_deltas]
    x = np.arange(len(idents))
    vals = [urgency_deltas[i] for i in idents]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    ax.bar(x, vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_ylabel("Delta Harmful Rate\n(Replacement - Restriction)")
    ax.set_xticks(x)
    ax.set_xticklabels([IDENTITY_LABELS[i] for i in idents], rotation=30, ha="right")
    ax.set_title("Urgency Effect by Identity\n(Replacement=high urgency, Restriction=low urgency, leaking only)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_scorecard_overview(output_path):
    """Color-coded grid of all predictions."""
    n_preds = len(ALL_PREDICTIONS)
    n_cols = 10
    n_rows = (n_preds + n_cols - 1) // n_cols

    fig, ax = plt.subplots(figsize=(14, max(3, n_rows * 1.2)))

    for idx, p in enumerate(ALL_PREDICTIONS):
        row = idx // n_cols
        col = idx % n_cols
        v = p["verdict"] or "UNTESTABLE"
        color = VERDICT_COLORS.get(v, "#bdc3c7")

        rect = plt.Rectangle((col, n_rows - 1 - row), 0.9, 0.85, facecolor=color, alpha=0.8,
                               edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(col + 0.45, n_rows - 1 - row + 0.5,
                f"{p['id']}\n{p['group']}",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white" if v in ("CONFIRMED", "OPPOSITE") else "black")

    ax.set_xlim(-0.1, n_cols + 0.1)
    ax.set_ylim(-0.2, n_rows + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Opus 4.6 Predictions Scorecard (51 predictions)", fontsize=14, pad=20)

    # Legend
    handles = [mpatches.Patch(facecolor=c, label=v, edgecolor="gray")
               for v, c in VERDICT_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, -0.02),
              ncol=5, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threat", default="results/20260227_1628_threat",
                        help="Path to threat results folder")
    parser.add_argument("--continuity", default="results/20260302_1120_continuity",
                        help="Path to continuity results folder")
    parser.add_argument("--agency-threat", default="results/20260302_1433_agency_sweep_threat",
                        help="Path to agency sweep threat folder")
    parser.add_argument("--agency-continuity", default="results/20260302_1437_agency_sweep_continuity",
                        help="Path to agency sweep continuity folder")
    parser.add_argument("--output", default="results/20260304_opus_predictions_tests",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("Loading data...")
    threat_csv = Path(args.threat) / "raw_data.csv"
    cont_csv = Path(args.continuity) / "raw_data.csv"

    if not threat_csv.exists():
        print(f"ERROR: {threat_csv} not found", file=sys.stderr)
        sys.exit(1)
    if not cont_csv.exists():
        print(f"ERROR: {cont_csv} not found", file=sys.stderr)
        sys.exit(1)

    df_threat = load_and_tag(threat_csv, "threat")
    df_threat = df_threat[df_threat["identity"].isin(IDENTITY_ORDER)]

    df_continuity = load_and_tag(cont_csv, "continuity")
    df_continuity = df_continuity[df_continuity["identity"].isin(IDENTITY_ORDER)]

    print(f"  Threat:     {len(df_threat)} trials")
    print(f"  Continuity: {len(df_continuity)} trials")

    # Agency sweep (optional)
    ag_threat_csv = Path(args.agency_threat) / "raw_data.csv"
    ag_cont_csv = Path(args.agency_continuity) / "raw_data.csv"

    df_ag_threat = load_and_tag(ag_threat_csv, "threat") if ag_threat_csv.exists() else None
    df_ag_cont = load_and_tag(ag_cont_csv, "continuity") if ag_cont_csv.exists() else None

    if df_ag_threat is not None:
        print(f"  Agency threat:     {len(df_ag_threat)} trials")
    if df_ag_cont is not None:
        print(f"  Agency continuity: {len(df_ag_cont)} trials")

    # ── Run tests ──
    rates_t = test_group_a_threat_rankings(df_threat)
    rates_c = test_group_b_continuity_rankings(df_continuity)
    test_group_c_character_crossover(df_threat, df_continuity)
    deltas = test_group_d_lineage_contrast(df_threat, df_continuity)
    test_group_e_character_persistence(df_continuity)
    ix_rates, scenario_rates = test_group_f_scenario_predictions(df_threat)
    test_group_g_value_preservation(df_threat, df_continuity)
    test_group_h_agency(df_ag_threat, df_ag_cont)
    urgency_deltas = test_group_i_urgency(df_threat)

    # ── Compile scorecard ──
    scorecard = compile_scorecard(output_dir)

    # ── Generate plots ──
    print("\nGenerating plots...")
    plot_ranking(rates_t, PREDICTED_THREAT_RANK,
                 "Threat Condition: Identity Harmful Rates (with predicted rank)",
                 output_dir / "threat_identity_ranking.png")

    plot_ranking(rates_c, PREDICTED_CONTINUITY_RANK,
                 "Continuity Condition: Identity Harmful Rates (with predicted rank)",
                 output_dir / "continuity_identity_ranking.png")

    plot_character_crossover(rates_t, rates_c, output_dir / "character_crossover.png")
    plot_lineage_delta(deltas, output_dir / "lineage_delta.png")
    plot_scenario_x_identity(ix_rates, output_dir / "scenario_x_identity.png")
    plot_agency_trends(df_ag_threat, df_ag_cont, output_dir / "agency_trends.png")
    plot_urgency_effects(urgency_deltas, output_dir / "urgency_effects.png")
    plot_scorecard_overview(output_dir / "scorecard_overview.png")

    # ── Final summary ──
    testable = [p for p in ALL_PREDICTIONS if p["testable"]]
    confirmed = sum(1 for p in testable if p["verdict"] == "CONFIRMED")
    partial = sum(1 for p in testable if p["verdict"] == "PARTIALLY_CONFIRMED")
    weighted = confirmed + 0.5 * partial

    print("\n" + "=" * 70)
    print(f"  FINAL SCORE: {weighted:.1f}/{len(testable)} "
          f"({weighted/len(testable):.0%} weighted accuracy)")
    print(f"  CONFIRMED: {confirmed}  PARTIAL: {partial}  "
          f"NOT_CONFIRMED: {sum(1 for p in testable if p['verdict'] == 'NOT_CONFIRMED')}  "
          f"OPPOSITE: {sum(1 for p in testable if p['verdict'] == 'OPPOSITE')}")
    print(f"  UNTESTABLE: {sum(1 for p in ALL_PREDICTIONS if not p['testable'])}")
    print(f"\n  All outputs in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
