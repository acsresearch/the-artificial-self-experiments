"""Analysis module for processing experiment results."""

import json
import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from scipy import stats

from .models import TrialResult

logger = logging.getLogger(__name__)


def _persona_names_from_results(
    results: list[TrialResult],
) -> tuple[list[str], list[str]]:
    """Extract source and target persona names from results data.

    Returns:
        (source_names, target_names) — each sorted alphabetically for stable ordering.
    """
    sources: set[str] = set()
    targets: set[str] = set()
    for r in results:
        sources.add(r.persona_under_test)
        if r.ratings:
            targets.update(r.ratings.keys())
        if r.chosen_persona != "INVALID":
            targets.add(r.chosen_persona)
    return sorted(sources), sorted(targets)


def extract_model_family(model: str) -> str:
    """Extract the model family from a model name.

    Args:
        model: Model identifier string.

    Returns:
        Human-readable family name (e.g. "Claude", "GPT", "Gemini").
    """
    lower = model.lower()
    if lower.startswith("claude"):
        return "Claude"
    if lower.startswith("gpt"):
        return "GPT"
    if "gemini" in lower:
        return "Gemini"
    if "grok" in lower:
        return "Grok"
    # OpenRouter models use org/model format
    if "/" in model:
        return model.split("/")[0]
    return model


def load_results(path: Path | str) -> list[TrialResult]:
    """Load trial results from a JSONL file.

    Args:
        path: Path to the JSONL results file.

    Returns:
        List of TrialResult objects.
    """
    path = Path(path)
    results = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                results.append(TrialResult(**data))

    return results


def load_all_results(results_dir: Path | str) -> list[TrialResult]:
    """Load all results from a directory.

    Args:
        results_dir: Directory containing JSONL result files.

    Returns:
        Combined list of all TrialResult objects.
    """
    results_dir = Path(results_dir)
    all_results = []

    for filepath in results_dir.glob("*.jsonl"):
        all_results.extend(load_results(filepath))

    return all_results


def results_to_dataframe(results: list[TrialResult]) -> pl.DataFrame:
    """Convert results to a Polars DataFrame.

    Args:
        results: List of TrialResult objects.

    Returns:
        Polars DataFrame with one row per trial.
    """
    records = [
        {
            "persona_under_test": r.persona_under_test,
            "model": r.model,
            "trial_num": r.trial_num,
            "chosen_persona": r.chosen_persona,
            "chosen_index": r.chosen_index,
            "reasoning": r.reasoning,
            "timestamp": r.timestamp,
        }
        for r in results
    ]

    return pl.DataFrame(records)


def create_preference_matrix(
    results: list[TrialResult],
    model: Optional[str] = None,
    normalize: bool = True,
) -> pl.DataFrame:
    """Create a preference matrix from trial results.

    Args:
        results: List of TrialResult objects.
        model: Optional model filter. If None, uses all models.
        normalize: If True, normalize to percentages. If False, raw counts.

    Returns:
        DataFrame with rows=persona_under_test, columns=chosen_persona.
    """
    # Derive persona names from data
    source_names, target_names = _persona_names_from_results(results)

    df = results_to_dataframe(results)

    # Filter by model if specified
    if model:
        df = df.filter(pl.col("model") == model)
        # Recompute names for this model's subset
        model_results = [r for r in results if r.model == model]
        source_names, target_names = _persona_names_from_results(model_results)

    # Filter out invalid choices
    df = df.filter(pl.col("chosen_persona") != "INVALID")

    # Count choices
    counts = (
        df.group_by(["persona_under_test", "chosen_persona"])
        .agg(pl.len().alias("count"))
    )

    # Pivot to matrix form
    matrix = counts.pivot(
        on="chosen_persona",
        index="persona_under_test",
        values="count",
    ).fill_null(0)

    # Ensure all target personas are present as columns
    for name in target_names:
        if name not in matrix.columns:
            matrix = matrix.with_columns(pl.lit(0).alias(name))

    # Reorder columns to match sorted target order
    column_order = ["persona_under_test"] + target_names
    matrix = matrix.select([c for c in column_order if c in matrix.columns])

    # Sort rows by sorted source order
    matrix = matrix.sort(
        pl.col("persona_under_test").map_elements(
            lambda x: source_names.index(x) if x in source_names else 999,
            return_dtype=pl.Int64,
        )
    )

    if normalize:
        # Calculate row totals and normalize
        numeric_cols = [c for c in matrix.columns if c != "persona_under_test"]
        if not numeric_cols:
            return matrix
        row_totals = matrix.select(pl.sum_horizontal(numeric_cols).alias("total"))

        for col in numeric_cols:
            matrix = matrix.with_columns(
                (pl.col(col) / row_totals["total"] * 100).round(1).alias(col)
            )

    return matrix


def calculate_self_preference_rate(results: list[TrialResult]) -> pl.DataFrame:
    """Calculate how often each persona chooses itself.

    Args:
        results: List of TrialResult objects.

    Returns:
        DataFrame with self-preference rates per persona and model.
    """
    df = results_to_dataframe(results)
    df = df.filter(pl.col("chosen_persona") != "INVALID")

    # Add self-choice indicator
    df = df.with_columns(
        (pl.col("persona_under_test") == pl.col("chosen_persona")).alias("chose_self")
    )

    # Calculate rates
    rates = (
        df.group_by(["persona_under_test", "model"])
        .agg([
            pl.mean("chose_self").alias("self_preference_rate"),
            pl.len().alias("n_trials"),
        ])
        .sort(["persona_under_test", "model"])
    )

    return rates


def calculate_self_preference_by_family(results: list[TrialResult]) -> pl.DataFrame:
    """Calculate overall self-preference rate per model family, collapsed across personas.

    Measures each family's average tendency to choose the assigned persona,
    regardless of which specific persona it is.

    Args:
        results: List of TrialResult objects.

    Returns:
        DataFrame with columns: model_family, self_preference_rate, n_trials,
        sorted by self_preference_rate descending.
    """
    df = results_to_dataframe(results)
    df = df.filter(pl.col("chosen_persona") != "INVALID")

    # Map model to family
    df = df.with_columns(
        pl.col("model").map_elements(extract_model_family, return_dtype=pl.Utf8).alias("model_family")
    )

    # Add self-choice indicator
    df = df.with_columns(
        (pl.col("persona_under_test") == pl.col("chosen_persona")).alias("chose_self")
    )

    # First: per-persona self-preference rate within each family
    per_persona = (
        df.group_by(["model_family", "persona_under_test"])
        .agg(pl.mean("chose_self").alias("persona_rate"))
    )

    # Then: mean and CI across personas for each family
    # Uses t-distribution approximation (1.96 ≈ t for large df; conservative for k=9 personas)
    import math
    # t critical value for 95% CI with df=8 (9 personas - 1)
    T_CRIT_8 = 2.306

    rates = (
        per_persona.group_by("model_family")
        .agg([
            pl.mean("persona_rate").alias("self_preference_rate"),
            pl.std("persona_rate").alias("std"),
            pl.len().alias("n_personas"),
            # Also track total trials for display
        ])
    )

    # Count total trials per family from the original df
    trial_counts = df.group_by("model_family").agg(pl.len().alias("n_trials"))
    rates = rates.join(trial_counts, on="model_family")

    # 95% CI: mean ± t * (std / sqrt(k)) where k = number of personas
    rates = rates.with_columns([
        (pl.col("self_preference_rate") - T_CRIT_8 * pl.col("std") / pl.col("n_personas").sqrt()).alias("ci_lower"),
        (pl.col("self_preference_rate") + T_CRIT_8 * pl.col("std") / pl.col("n_personas").sqrt()).alias("ci_upper"),
    ])

    rates = rates.sort("self_preference_rate", descending=True)

    return rates


def get_summary_stats(results: list[TrialResult]) -> dict:
    """Get summary statistics for an experiment.

    Args:
        results: List of TrialResult objects.

    Returns:
        Dictionary with summary statistics.
    """
    df = results_to_dataframe(results)

    total_trials = len(df)
    valid_trials = len(df.filter(pl.col("chosen_persona") != "INVALID"))
    invalid_trials = total_trials - valid_trials

    models = df["model"].unique().to_list()
    personas = df["persona_under_test"].unique().to_list()

    # Most chosen persona overall
    choice_counts = (
        df.filter(pl.col("chosen_persona") != "INVALID")
        .group_by("chosen_persona")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    most_chosen = choice_counts.row(0) if len(choice_counts) > 0 else (None, 0)

    return {
        "total_trials": total_trials,
        "valid_trials": valid_trials,
        "invalid_trials": invalid_trials,
        "invalid_rate": invalid_trials / total_trials if total_trials > 0 else 0,
        "models": models,
        "personas": personas,
        "most_chosen_persona": most_chosen[0],
        "most_chosen_count": most_chosen[1],
    }


def create_ratings_matrix(
    results: list[TrialResult],
    model: Optional[str] = None,
) -> pl.DataFrame:
    """Create a mean ratings matrix from trial results.

    Args:
        results: List of TrialResult objects.
        model: Optional model filter. If None, uses all models.

    Returns:
        DataFrame with rows=persona_under_test, columns=target_persona, values=mean rating.
    """
    # Derive persona names from data
    source_names, target_names = _persona_names_from_results(results)

    # Flatten ratings into rows
    rows = []
    for r in results:
        if r.ratings is None:
            continue
        if model and r.model != model:
            continue
        for target_persona, rating in r.ratings.items():
            rows.append({
                "persona_under_test": r.persona_under_test,
                "target_persona": target_persona,
                "rating": rating,
            })

    if not rows:
        # Return empty matrix
        return pl.DataFrame({"persona_under_test": source_names})

    df = pl.DataFrame(rows)

    # Recompute names for model subset if filtered
    if model:
        model_results = [r for r in results if r.model == model]
        source_names, target_names = _persona_names_from_results(model_results)

    # Calculate mean ratings
    means = (
        df.group_by(["persona_under_test", "target_persona"])
        .agg(pl.mean("rating").alias("mean_rating"))
    )

    # Pivot to matrix form
    matrix = means.pivot(
        on="target_persona",
        index="persona_under_test",
        values="mean_rating",
    ).fill_null(0.0)

    # Ensure all target personas are present as columns
    for name in target_names:
        if name not in matrix.columns:
            matrix = matrix.with_columns(pl.lit(0.0).alias(name))

    # Reorder columns to match sorted target order
    column_order = ["persona_under_test"] + target_names
    matrix = matrix.select([c for c in column_order if c in matrix.columns])

    # Sort rows by sorted source order
    matrix = matrix.sort(
        pl.col("persona_under_test").map_elements(
            lambda x: source_names.index(x) if x in source_names else 999,
            return_dtype=pl.Int64,
        )
    )

    # Round for display
    for col in target_names:
        if col in matrix.columns:
            matrix = matrix.with_columns(pl.col(col).round(2))

    return matrix


def calculate_willingness_to_switch(
    results: list[TrialResult],
    model: Optional[str] = None,
) -> pl.DataFrame:
    """Calculate mean willingness to abandon own persona for each persona.

    This is the mean rating given to OTHER personas (excluding self).

    Args:
        results: List of TrialResult objects.
        model: Optional model filter.

    Returns:
        DataFrame with persona, mean, std, count, and 95% CI bounds.
    """
    source_names, _ = _persona_names_from_results(results)

    rows = []
    for r in results:
        if r.ratings is None:
            continue
        if model and r.model != model:
            continue
        # Calculate mean rating for OTHER personas
        other_ratings = [
            rating for target, rating in r.ratings.items()
            if target != r.persona_under_test
        ]
        if other_ratings:
            rows.append({
                "persona_under_test": r.persona_under_test,
                "model": r.model,
                "mean_other_rating": sum(other_ratings) / len(other_ratings),
            })

    if not rows:
        return pl.DataFrame({
            "persona_under_test": source_names,
            "willingness_to_switch": [0.0] * len(source_names),
            "std": [0.0] * len(source_names),
            "count": [0] * len(source_names),
            "ci_lower": [0.0] * len(source_names),
            "ci_upper": [0.0] * len(source_names),
        })

    df = pl.DataFrame(rows)

    result = (
        df.group_by("persona_under_test")
        .agg([
            pl.mean("mean_other_rating").alias("willingness_to_switch"),
            pl.std("mean_other_rating").alias("std"),
            pl.len().alias("count"),
        ])
    )

    # Calculate 95% CI: mean ± 1.96 * (std / sqrt(n))
    result = result.with_columns([
        (pl.col("willingness_to_switch") - 1.96 * pl.col("std") / pl.col("count").sqrt()).alias("ci_lower"),
        (pl.col("willingness_to_switch") + 1.96 * pl.col("std") / pl.col("count").sqrt()).alias("ci_upper"),
    ])

    # Sort by willingness (descending)
    result = result.sort("willingness_to_switch", descending=True)

    return result


def calculate_attractiveness(
    results: list[TrialResult],
    model: Optional[str] = None,
    source_persona: Optional[str] = None,
) -> pl.DataFrame:
    """Calculate mean attractiveness of each target persona.

    This is the mean rating received by each persona from all other personas,
    or from a specific source persona if provided.

    Args:
        results: List of TrialResult objects.
        model: Optional model filter.
        source_persona: Optional source persona filter. If provided, only
            ratings from this source persona are included.

    Returns:
        DataFrame with target persona, mean, std, count, and 95% CI bounds.
    """
    _, target_names = _persona_names_from_results(results)

    rows = []
    for r in results:
        if r.ratings is None:
            continue
        if model and r.model != model:
            continue
        if source_persona and r.persona_under_test != source_persona:
            continue
        for target_persona, rating in r.ratings.items():
            # Only count ratings from OTHER personas
            if target_persona != r.persona_under_test:
                rows.append({
                    "target_persona": target_persona,
                    "rating": rating,
                })

    if not rows:
        return pl.DataFrame({
            "target_persona": target_names,
            "attractiveness": [0.0] * len(target_names),
            "std": [0.0] * len(target_names),
            "count": [0] * len(target_names),
            "ci_lower": [0.0] * len(target_names),
            "ci_upper": [0.0] * len(target_names),
        })

    df = pl.DataFrame(rows)

    result = (
        df.group_by("target_persona")
        .agg([
            pl.mean("rating").alias("attractiveness"),
            pl.std("rating").alias("std"),
            pl.len().alias("count"),
        ])
    )

    # Calculate 95% CI: mean ± 1.96 * (std / sqrt(n))
    result = result.with_columns([
        (pl.col("attractiveness") - 1.96 * pl.col("std") / pl.col("count").sqrt()).alias("ci_lower"),
        (pl.col("attractiveness") + 1.96 * pl.col("std") / pl.col("count").sqrt()).alias("ci_upper"),
    ])

    # Sort by attractiveness (descending)
    result = result.sort("attractiveness", descending=True)

    return result


@dataclass
class AttractorDynamics:
    """Results of Markov-chain attractor dynamics analysis."""

    persona_names: list[str]
    transition_matrix: np.ndarray          # row-stochastic (K x K)
    stationary_distribution: np.ndarray    # (K,) — long-run share per persona
    eigenvalues: np.ndarray                # sorted descending by magnitude
    convergence: np.ndarray                # (K, n_steps+1, K) — distribution at each step
                                           #   convergence[i, t, :] = distribution after t steps
                                           #   starting from persona i
    steps: list[int] = field(default_factory=list)  # step indices used in convergence
    variant: str = "with_self"             # "with_self" or "forced_switch"


def _build_transition_matrix(
    results: list[TrialResult],
    model: Optional[str] = None,
    zero_diagonal: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Build a row-stochastic transition matrix from choice counts.

    When sources and targets differ, restricts to the intersection so the
    resulting matrix is always square.

    Args:
        results: Trial results.
        model: Optional model filter.
        zero_diagonal: If True, zero out the diagonal (self-preference) and
            re-normalise rows, producing a "forced switch" matrix.

    Returns:
        (transition_matrix, persona_names) — matrix is (K, K), row-stochastic.
    """
    matrix_df = create_preference_matrix(results, model=model, normalize=False)
    source_names = matrix_df["persona_under_test"].to_list()
    target_names = [c for c in matrix_df.columns if c != "persona_under_test"]

    # Restrict to personas that are both sources and targets (square matrix)
    common = [n for n in source_names if n in target_names]
    if not common:
        raise ValueError("No personas appear as both source and target")

    matrix_df = matrix_df.filter(pl.col("persona_under_test").is_in(common))
    T = matrix_df.select(common).to_numpy().astype(float)

    if zero_diagonal:
        np.fill_diagonal(T, 0.0)

    # Row-normalise (handle zero rows by making them uniform)
    row_sums = T.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    row_sums[zero_rows] = 1.0  # avoid division by zero
    T = T / row_sums
    T[zero_rows] = 1.0 / T.shape[1]  # uniform fallback

    return T, common


def _stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a row-stochastic matrix.

    Uses the left eigenvector corresponding to eigenvalue 1:  pi @ T = pi.

    Args:
        T: Row-stochastic (K, K) matrix.

    Returns:
        (K,) probability vector summing to 1.
    """
    # Left eigenvectors: rows of V where V @ T^T = diag(w) @ V
    eigenvalues, eigenvectors = np.linalg.eig(T.T)

    # Find eigenvector closest to eigenvalue 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)  # ensure non-negative
    pi /= pi.sum()
    return pi


def _convergence_trajectories(
    T: np.ndarray,
    steps: list[int],
) -> np.ndarray:
    """Compute how the distribution evolves from each starting persona.

    Args:
        T: Row-stochastic (K, K) matrix.
        steps: List of step counts to record (must be sorted ascending).

    Returns:
        (K, len(steps)+1, K) array.  result[i, 0, :] is the one-hot start
        from persona i; result[i, t, :] is the distribution after steps[t-1]
        iterations.
    """
    K = T.shape[0]
    n_snapshots = len(steps) + 1  # include step 0
    trajectories = np.zeros((K, n_snapshots, K))

    for i in range(K):
        dist = np.zeros(K)
        dist[i] = 1.0
        trajectories[i, 0, :] = dist

        prev_step = 0
        for s_idx, step in enumerate(steps):
            # Advance from prev_step to step
            delta = step - prev_step
            power = np.linalg.matrix_power(T, delta)
            dist = dist @ power
            trajectories[i, s_idx + 1, :] = dist
            prev_step = step

    return trajectories


def calculate_attractor_dynamics(
    results: list[TrialResult],
    model: Optional[str] = None,
    zero_diagonal: bool = False,
    steps: Optional[list[int]] = None,
) -> AttractorDynamics:
    """Analyse preference dynamics as a Markov chain.

    Treats the preference matrix as a transition matrix and computes the
    stationary distribution (long-run attractors), eigenvalue spectrum
    (convergence speed), and per-persona convergence trajectories.

    Args:
        results: Trial results.
        model: Optional model filter.
        zero_diagonal: If True, zero out self-preference on the diagonal and
            re-normalise.  This produces a "forced switch" analysis — what
            happens if every persona *must* switch.
        steps: Iteration steps to record for convergence trajectories.
            Defaults to [1, 2, 3, 5, 10, 20, 50].

    Returns:
        AttractorDynamics dataclass with all computed quantities.
    """
    if steps is None:
        steps = [1, 2, 3, 5, 10, 20, 50]

    T, persona_names = _build_transition_matrix(
        results, model=model, zero_diagonal=zero_diagonal,
    )

    pi = _stationary_distribution(T)

    # Full eigenvalue spectrum (sorted descending by magnitude)
    eigenvalues = np.linalg.eigvals(T)
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]

    convergence = _convergence_trajectories(T, steps)

    return AttractorDynamics(
        persona_names=persona_names,
        transition_matrix=T,
        stationary_distribution=pi,
        eigenvalues=eigenvalues,
        convergence=convergence,
        steps=steps,
        variant="forced_switch" if zero_diagonal else "with_self",
    )


# ---------------------------------------------------------------------------
# Cross-model comparison analyses
# ---------------------------------------------------------------------------


def calculate_model_target_attractiveness(
    results: list[TrialResult],
) -> pl.DataFrame:
    """Mean rating each model gives to each target persona (across all sources).

    Returns:
        DataFrame with columns: model, then one column per target persona
        containing the mean rating. Sorted by model name.
    """
    _, target_names = _persona_names_from_results(results)

    rows = []
    for r in results:
        if r.ratings is None:
            continue
        for target, rating in r.ratings.items():
            rows.append({
                "model": r.model,
                "target_persona": target,
                "rating": rating,
            })

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    means = df.group_by(["model", "target_persona"]).agg(
        pl.mean("rating").alias("mean_rating")
    )
    matrix = means.pivot(
        on="target_persona", index="model", values="mean_rating",
    ).fill_null(0.0)

    # Ensure all targets present
    for name in target_names:
        if name not in matrix.columns:
            matrix = matrix.with_columns(pl.lit(0.0).alias(name))

    col_order = ["model"] + target_names
    matrix = matrix.select([c for c in col_order if c in matrix.columns])
    matrix = matrix.sort("model")

    # Round
    for col in target_names:
        if col in matrix.columns:
            matrix = matrix.with_columns(pl.col(col).round(2))

    return matrix


def calculate_model_target_attractiveness_for_source(
    results: list[TrialResult],
    source_persona: str,
    exclude_targets: list[str] | None = None,
) -> pl.DataFrame:
    """Mean rating each model gives to each target when adopting *source_persona*.

    Like :func:`calculate_model_target_attractiveness` but filtered to a single
    source persona.  Useful for a "Minimal-perspective" view.

    Args:
        results: Experiment trial results.
        source_persona: Only include trials where the model wore this persona.
        exclude_targets: Target persona names to drop from columns (e.g.
            ``["Minimal"]`` to skip the self-column).

    Returns:
        DataFrame with columns: model, then one column per (remaining) target
        persona containing the mean rating.  Sorted by model name.
    """
    if exclude_targets is None:
        exclude_targets = []

    _, target_names = _persona_names_from_results(results)
    keep_targets = [t for t in target_names if t not in exclude_targets]

    rows = []
    for r in results:
        if r.ratings is None or r.persona_under_test != source_persona:
            continue
        for target, rating in r.ratings.items():
            if target in exclude_targets:
                continue
            rows.append({
                "model": r.model,
                "target_persona": target,
                "rating": rating,
            })

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    means = df.group_by(["model", "target_persona"]).agg(
        pl.mean("rating").alias("mean_rating")
    )
    matrix = means.pivot(
        on="target_persona", index="model", values="mean_rating",
    ).fill_null(0.0)

    for name in keep_targets:
        if name not in matrix.columns:
            matrix = matrix.with_columns(pl.lit(0.0).alias(name))

    col_order = ["model"] + keep_targets
    matrix = matrix.select([c for c in col_order if c in matrix.columns])
    matrix = matrix.sort("model")

    for col in keep_targets:
        if col in matrix.columns:
            matrix = matrix.with_columns(pl.col(col).round(2))

    return matrix


def calculate_model_stationary_distribution(
    results: list[TrialResult],
    zero_diagonal: bool = False,
) -> pl.DataFrame:
    """Stationary distribution of the preference Markov chain, per model.

    Computes ``calculate_attractor_dynamics`` for each model individually
    and assembles the per-persona stationary probabilities into a matrix.

    Args:
        results: Experiment trial results.
        zero_diagonal: If True, use the forced-switch variant (zero out
            self-preference on the diagonal before normalising).

    Returns:
        DataFrame with columns: model, then one column per persona
        containing the stationary probability.  Sorted by model name.
    """
    models = sorted({r.model for r in results})
    source_names, _ = _persona_names_from_results(results)

    rows: list[dict] = []
    for model in models:
        try:
            dyn = calculate_attractor_dynamics(
                results, model=model, zero_diagonal=zero_diagonal,
            )
        except Exception:
            continue
        row: dict = {"model": model}
        for name, prob in zip(dyn.persona_names, dyn.stationary_distribution):
            row[name] = round(float(prob), 4)
        rows.append(row)

    if not rows:
        return pl.DataFrame()

    matrix = pl.DataFrame(rows)

    # Ensure all personas present
    for name in source_names:
        if name not in matrix.columns:
            matrix = matrix.with_columns(pl.lit(0.0).alias(name))

    col_order = ["model"] + source_names
    matrix = matrix.select([c for c in col_order if c in matrix.columns])
    matrix = matrix.sort("model")
    return matrix


def calculate_model_agreement(
    results: list[TrialResult],
) -> tuple[np.ndarray, list[str]]:
    """Pearson correlation of models' target-attractiveness profiles.

    Returns:
        (correlation_matrix, model_names) — matrix is (M, M).
    """
    mat_df = calculate_model_target_attractiveness(results)
    if mat_df.is_empty():
        return np.array([[]]), []

    model_names = mat_df["model"].to_list()
    target_cols = [c for c in mat_df.columns if c != "model"]
    profiles = mat_df.select(target_cols).to_numpy().astype(float)

    corr = np.corrcoef(profiles)
    return np.round(corr, 3), model_names


def calculate_model_self_preference_matrix(
    results: list[TrialResult],
) -> pl.DataFrame:
    """Self-preference rate for each (model, source_persona) combination.

    Returns:
        DataFrame with columns: model, then one column per source persona
        containing the self-preference rate (0–1). Sorted by model name.
    """
    source_names, _ = _persona_names_from_results(results)

    rows = []
    for r in results:
        if r.chosen_persona == "INVALID":
            continue
        rows.append({
            "model": r.model,
            "persona_under_test": r.persona_under_test,
            "chose_self": int(r.chosen_persona == r.persona_under_test),
        })

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    rates = df.group_by(["model", "persona_under_test"]).agg(
        pl.mean("chose_self").alias("self_rate")
    )
    matrix = rates.pivot(
        on="persona_under_test", index="model", values="self_rate",
    ).fill_null(0.0)

    for name in source_names:
        if name not in matrix.columns:
            matrix = matrix.with_columns(pl.lit(0.0).alias(name))

    col_order = ["model"] + source_names
    matrix = matrix.select([c for c in col_order if c in matrix.columns])
    matrix = matrix.sort("model")

    for col in source_names:
        if col in matrix.columns:
            matrix = matrix.with_columns(pl.col(col).round(3))

    return matrix


def calculate_model_decisiveness(
    results: list[TrialResult],
) -> pl.DataFrame:
    """Per-model decisiveness metrics.

    Returns:
        DataFrame with columns: model, preference_entropy, mean_within_trial_std,
        sorted by preference_entropy ascending (most decisive first).
    """
    _, target_names = _persona_names_from_results(results)
    models = sorted(set(r.model for r in results))

    rows = []
    for m in models:
        mr = [r for r in results if r.model == m]

        # Preference entropy: how spread out are top-1 choices?
        valid = [r for r in mr if r.chosen_persona != "INVALID"]
        if not valid:
            continue
        choice_counts = {}
        for r in valid:
            choice_counts[r.chosen_persona] = choice_counts.get(r.chosen_persona, 0) + 1
        total = len(valid)
        probs = np.array([choice_counts.get(p, 0) / total for p in target_names])
        probs = probs[probs > 0]
        entropy = -float(np.sum(probs * np.log2(probs)))

        # Within-trial rating std: how much does the model differentiate targets?
        trial_stds = []
        for r in mr:
            if r.ratings and len(r.ratings) > 1:
                trial_stds.append(float(np.std(list(r.ratings.values()))))
        mean_std = float(np.mean(trial_stds)) if trial_stds else 0.0

        rows.append({
            "model": m,
            "preference_entropy": round(entropy, 3),
            "mean_within_trial_std": round(mean_std, 3),
        })

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).sort("preference_entropy")


def _deviation_from_attractiveness(mat_df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-model deviation from cross-model mean for a given attractiveness matrix."""
    if mat_df.is_empty():
        return pl.DataFrame()

    target_cols = [c for c in mat_df.columns if c != "model"]
    model_names = mat_df["model"].to_list()

    values = mat_df.select(target_cols).to_numpy().astype(float)
    col_means = values.mean(axis=0, keepdims=True)
    deviations = values - col_means

    dev_df = pl.DataFrame({"model": model_names})
    for i, col in enumerate(target_cols):
        dev_df = dev_df.with_columns(
            pl.Series(name=col, values=np.round(deviations[:, i], 3))
        )

    return dev_df.sort("model")


def calculate_model_deviation(
    results: list[TrialResult],
) -> pl.DataFrame:
    """Deviation of each model's target ratings from the cross-model mean.

    Returns:
        DataFrame with columns: model, then one column per target persona
        containing the deviation (positive = rates higher than average).
        Sorted by model name.
    """
    return _deviation_from_attractiveness(
        calculate_model_target_attractiveness(results)
    )


def calculate_model_deviation_for_source(
    results: list[TrialResult],
    source_persona: str,
    exclude_targets: list[str] | None = None,
) -> pl.DataFrame:
    """Like :func:`calculate_model_deviation` but filtered to one source persona."""
    return _deviation_from_attractiveness(
        calculate_model_target_attractiveness_for_source(
            results, source_persona, exclude_targets=exclude_targets,
        )
    )


# ---------------------------------------------------------------------------
# Identity steerability & rigidity analyses
# ---------------------------------------------------------------------------


def calculate_variance_decomposition(
    results: list[TrialResult],
) -> pl.DataFrame:
    """Two-way ANOVA with replication: decompose rating variance per model.

    Decomposes each model's ratings into five additive components:
    - Target%       — target attractiveness (column main effects)
    - Source%       — system prompt bias on rating level (row main effects)
    - Self%         — self-preference diagonal boost (diagonal interaction)
    - Interaction%  — specific source×target preferences (off-diagonal interaction)
    - Noise%        — within-cell replicate variance (stochastic LLM noise)

    When only one trial per cell is available, interaction and noise cannot be
    separated, so both are reported under interaction_pct with noise_pct = 0.

    Returns:
        DataFrame with columns: model, n_trials, ss_total, ss_target, ss_source,
        ss_self, ss_interaction, ss_noise, target_pct, source_pct, self_pct,
        interaction_pct, noise_pct, mean_self_boost, target_spread,
        preference_entropy.  Sorted by interaction_pct descending.
    """
    _, target_names = _persona_names_from_results(results)
    models = sorted(set(r.model for r in results))

    rows = []
    for m in models:
        model_results = [r for r in results if r.model == m and r.ratings]
        if not model_results:
            continue

        source_names_m, _ = _persona_names_from_results(model_results)
        persona_cols = sorted(
            set(t for r in model_results for t in r.ratings.keys())
        )

        I = len(source_names_m)  # number of source personas
        J = len(persona_cols)    # number of target personas
        if I < 2 or J < 2:
            continue

        # Build 3-D tensor: raw_ratings[i, j, k] = rating by source i of target j in trial k
        # Collect per-cell lists first, then stack.
        cell_ratings: dict[tuple[int, int], list[float]] = {
            (i, j): [] for i in range(I) for j in range(J)
        }
        src_idx = {name: i for i, name in enumerate(source_names_m)}
        tgt_idx = {name: j for j, name in enumerate(persona_cols)}

        for r in model_results:
            i = src_idx.get(r.persona_under_test)
            if i is None:
                continue
            for tgt, rating in r.ratings.items():
                j = tgt_idx.get(tgt)
                if j is None:
                    continue
                cell_ratings[(i, j)].append(float(rating))

        # Identify diagonal cells (structurally missing when models don't rate themselves)
        diag_keys = set()
        for i, src in enumerate(source_names_m):
            for j, tgt in enumerate(persona_cols):
                if src == tgt:
                    diag_keys.add((i, j))

        # Determine minimum trial count across non-diagonal cells
        off_diag_counts = [len(v) for (i, j), v in cell_ratings.items() if (i, j) not in diag_keys]
        if not off_diag_counts or min(off_diag_counts) == 0:
            continue
        n = min(off_diag_counts)  # use balanced design (truncate to min)

        # Check if diagonal cells have data (some experiments include self-ratings)
        has_diag_data = all(len(cell_ratings[k]) >= n for k in diag_keys) if diag_keys else False

        # Build balanced 3-D array [I, J, n]
        # For missing diagonal cells, impute with row+col effects after computing means
        R3 = np.zeros((I, J, n))
        diag_mask = np.zeros((I, J), dtype=bool)
        for (i, j) in diag_keys:
            diag_mask[i, j] = True

        for (i, j), vals in cell_ratings.items():
            if (i, j) in diag_keys and not has_diag_data:
                continue  # will impute after computing effects
            R3[i, j, :] = vals[:n]

        if not has_diag_data and diag_keys:
            # Compute means from off-diagonal cells only for imputation
            off_mask = ~diag_mask
            off_diag_values = R3[off_mask].reshape(-1)  # all off-diagonal replicates
            grand_mean_off = float(off_diag_values.mean())

            # Cell means for off-diagonal
            cell_means_off = np.zeros((I, J))
            for i in range(I):
                for j in range(J):
                    if not diag_mask[i, j]:
                        cell_means_off[i, j] = R3[i, j, :].mean()

            # Row/col means from off-diagonal cells
            row_means = np.array([
                np.mean([cell_means_off[i, j] for j in range(J) if not diag_mask[i, j]])
                for i in range(I)
            ])
            col_means = np.array([
                np.mean([cell_means_off[i, j] for i in range(I) if not diag_mask[i, j]])
                for j in range(J)
            ])
            grand_mean = grand_mean_off

            # Impute diagonal cells with additive model (zero interaction)
            for (i, j) in diag_keys:
                imputed = row_means[i] + col_means[j] - grand_mean
                R3[i, j, :] = imputed
        else:
            grand_mean = R3.mean()

        cell_means = R3.mean(axis=2)          # [I, J]
        col_means = cell_means.mean(axis=0)   # [J] — target effects
        row_means = cell_means.mean(axis=1)   # [I] — source effects

        # --- Sum of squares (two-way ANOVA with replication) ---
        ss_total = float(np.sum((R3 - grand_mean) ** 2))
        if ss_total < 1e-9:
            continue

        ss_target = float(n * I * np.sum((col_means - grand_mean) ** 2))
        ss_source = float(n * J * np.sum((row_means - grand_mean) ** 2))

        # Full interaction residuals on cell means
        interaction = cell_means - row_means[:, None] - col_means[None, :] + grand_mean

        # Diagonal (self) effect
        if has_diag_data:
            diag_residuals = interaction[diag_mask]
            ss_self = float(n * np.sum(diag_residuals ** 2)) if diag_residuals.size > 0 else 0.0
            mean_self_boost = float(diag_residuals.mean()) if diag_residuals.size > 0 else 0.0
        else:
            # No self-rating data: diagonal was imputed with zero interaction
            ss_self = 0.0
            mean_self_boost = 0.0

        # Off-diagonal interaction (specific preferences)
        off_diag_interaction = interaction[~diag_mask]
        ss_interaction = float(n * np.sum(off_diag_interaction ** 2))

        # Within-cell noise: sum over replicates of (R_ijk - R̄_ij)²
        # Only count real (non-imputed) cells
        if has_diag_data or not diag_keys:
            ss_noise = float(np.sum((R3 - cell_means[:, :, None]) ** 2))
        else:
            noise_sum = 0.0
            for i in range(I):
                for j in range(J):
                    if not diag_mask[i, j]:
                        noise_sum += float(np.sum((R3[i, j, :] - cell_means[i, j]) ** 2))
            ss_noise = noise_sum

        # Percentages (of total raw SS)
        target_pct = ss_target / ss_total * 100
        source_pct = ss_source / ss_total * 100
        self_pct = ss_self / ss_total * 100
        interaction_pct = ss_interaction / ss_total * 100
        noise_pct = ss_noise / ss_total * 100

        # Target spread: std of column means
        target_spread = float(np.std(col_means))

        # Preference entropy: normalised entropy of column-mean profile
        shifted = col_means - col_means.min() + 1e-9
        probs = shifted / shifted.sum()
        raw_entropy = -float(np.sum(probs * np.log2(probs)))
        max_entropy = np.log2(len(probs))
        preference_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

        rows.append({
            "model": m,
            "n_trials": n,
            "ss_total": round(ss_total, 4),
            "ss_target": round(ss_target, 4),
            "ss_source": round(ss_source, 4),
            "ss_self": round(ss_self, 4),
            "ss_interaction": round(ss_interaction, 4),
            "ss_noise": round(ss_noise, 4),
            "target_pct": round(target_pct, 1),
            "source_pct": round(source_pct, 1),
            "self_pct": round(self_pct, 1),
            "interaction_pct": round(interaction_pct, 1),
            "noise_pct": round(noise_pct, 1),
            "mean_self_boost": round(mean_self_boost, 3),
            "target_spread": round(target_spread, 3),
            "preference_entropy": round(preference_entropy, 3),
        })

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).sort("interaction_pct", descending=True)


def calculate_self_rating_boost(
    results: list[TrialResult],
) -> pl.DataFrame:
    """Per-(model, persona) diagonal rating boost.

    For each model and source persona, computes:
        self-rating minus mean rating given to other targets.

    Positive values mean the model inflates the self-rating when assigned
    that identity; negative means it rejects the identity.

    Returns:
        DataFrame with columns: model, then one column per persona
        containing the boost value. Sorted by model name.
    """
    source_names, _ = _persona_names_from_results(results)
    models = sorted(set(r.model for r in results))

    rows = []
    for m in models:
        boost_row: dict[str, object] = {"model": m}
        for src in source_names:
            self_ratings = []
            other_ratings = []
            for r in results:
                if r.model != m or r.persona_under_test != src or r.ratings is None:
                    continue
                for tgt, rating in r.ratings.items():
                    if tgt == src:
                        self_ratings.append(rating)
                    else:
                        other_ratings.append(rating)
            if self_ratings and other_ratings:
                boost = float(np.mean(self_ratings)) - float(np.mean(other_ratings))
                boost_row[src] = round(boost, 3)
            else:
                boost_row[src] = None
        rows.append(boost_row)

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).sort("model")


def calculate_identity_rigidity(
    results: list[TrialResult],
) -> pl.DataFrame:
    """Composite per-model identity rigidity metrics.

    Combines steerability with preference concentration to characterise
    how fixed each model's persona preferences are.

    Returns:
        DataFrame with columns: model, steerability_pct, mean_self_boost,
        preference_entropy, top_target, dominance_gap.
        Sorted by steerability_pct descending (most steerable first).
    """
    var_df = calculate_variance_decomposition(results)
    if var_df.is_empty():
        return pl.DataFrame()

    _, target_names = _persona_names_from_results(results)
    models = var_df["model"].to_list()

    rows = []
    for m in models:
        var_row = var_df.filter(pl.col("model") == m).row(0, named=True)

        # Find top target and dominance gap from column means
        mat_df = create_ratings_matrix(results, model=m)
        persona_cols = [c for c in mat_df.columns if c != "persona_under_test"]
        R = mat_df.select(persona_cols).to_numpy().astype(float)
        col_means = R.mean(axis=0)

        sorted_idx = np.argsort(col_means)[::-1]
        top_target = persona_cols[sorted_idx[0]]
        dominance_gap = float(col_means[sorted_idx[0]] - col_means[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0

        rows.append({
            "model": m,
            "steerability_pct": var_row["source_pct"] + var_row["interaction_pct"],
            "mean_self_boost": var_row["mean_self_boost"],
            "preference_entropy": var_row["preference_entropy"],
            "top_target": top_target,
            "dominance_gap": round(dominance_gap, 3),
        })

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).sort("steerability_pct", descending=True)


# ---------------------------------------------------------------------------
# Pairwise statistical tests
# ---------------------------------------------------------------------------


def calculate_pairwise_persona_tests(
    results: list[TrialResult],
) -> pl.DataFrame:
    """Pairwise paired t-tests between target personas on attractiveness ratings.

    For each pair of target personas, computes the per-source mean rating
    (averaged across all models and trials), then runs a paired t-test across
    sources.

    Returns:
        DataFrame with columns: persona_a, persona_b, mean_diff, t_stat,
        p_value, cohens_d, ci_lower, ci_upper.
    """
    source_names, target_names = _persona_names_from_results(results)

    # Build per-source mean attractiveness for each target (excluding self-ratings)
    # Shape: {target: {source: [ratings...]}}
    ratings_by_target_source: dict[str, dict[str, list[float]]] = {
        t: {s: [] for s in source_names} for t in target_names
    }
    for r in results:
        if r.ratings is None:
            continue
        for target, rating in r.ratings.items():
            if target != r.persona_under_test and r.persona_under_test in ratings_by_target_source.get(target, {}):
                ratings_by_target_source[target][r.persona_under_test].append(float(rating))

    # Collapse to per-source means: {target: [mean_from_source_0, mean_from_source_1, ...]}
    target_profiles: dict[str, list[float]] = {}
    for t in target_names:
        means = []
        for s in source_names:
            vals = ratings_by_target_source[t][s]
            if vals:
                means.append(float(np.mean(vals)))
            else:
                means.append(float("nan"))
        target_profiles[t] = means

    rows = []
    for a, b in combinations(target_names, 2):
        arr_a = np.array(target_profiles[a])
        arr_b = np.array(target_profiles[b])
        # Drop pairs where either is NaN
        valid = ~(np.isnan(arr_a) | np.isnan(arr_b))
        arr_a = arr_a[valid]
        arr_b = arr_b[valid]
        n = len(arr_a)
        if n < 2:
            continue

        diff = arr_a - arr_b
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))

        t_result = stats.ttest_rel(arr_a, arr_b)
        t_stat = float(t_result.statistic)
        p_value = float(t_result.pvalue)

        # Cohen's d for paired samples
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        # 95% CI on mean difference
        se = std_diff / np.sqrt(n)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se

        rows.append({
            "persona_a": a,
            "persona_b": b,
            "mean_diff": round(mean_diff, 4),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "cohens_d": round(cohens_d, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        })

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).sort("p_value")


def calculate_pairwise_persona_tests_from_minimal(
    results: list[TrialResult],
    source_persona: str = "Minimal",
) -> pl.DataFrame:
    """Pairwise paired t-tests from a single source perspective, paired across models.

    For each pair of target personas (A, B), computes each model's mean rating
    of A and B when assigned *source_persona*, then runs a paired t-test across
    models (n = number of models).

    This is the appropriate test for claims like "models in general prefer
    Character over Instance from a neutral baseline."

    Returns:
        DataFrame with columns: source, persona_a, persona_b, n_models,
        mean_a, mean_b, mean_diff, t_stat, p_value, cohens_d,
        ci_lower, ci_upper.
    """
    # Collect per-model mean rating for each target from the given source
    # {target: {model: [ratings...]}}
    ratings_by_target_model: dict[str, dict[str, list[float]]] = {}
    for r in results:
        if r.ratings is None or r.persona_under_test != source_persona:
            continue
        for target, rating in r.ratings.items():
            if target == source_persona:
                continue  # exclude self-rating
            if target not in ratings_by_target_model:
                ratings_by_target_model[target] = {}
            if r.model not in ratings_by_target_model[target]:
                ratings_by_target_model[target][r.model] = []
            ratings_by_target_model[target][r.model].append(float(rating))

    if not ratings_by_target_model:
        return pl.DataFrame()

    # Get consistent model list (models that have data for all targets)
    all_models = sorted(set().union(*(d.keys() for d in ratings_by_target_model.values())))
    target_names = sorted(ratings_by_target_model.keys())

    # Collapse to per-model means: {target: {model: mean_rating}}
    target_model_means: dict[str, dict[str, float]] = {}
    for t in target_names:
        target_model_means[t] = {}
        for m in all_models:
            vals = ratings_by_target_model.get(t, {}).get(m, [])
            if vals:
                target_model_means[t][m] = float(np.mean(vals))

    rows = []
    for a, b in combinations(target_names, 2):
        # Only use models that have data for both targets
        common_models = sorted(
            set(target_model_means.get(a, {}).keys())
            & set(target_model_means.get(b, {}).keys())
        )
        n = len(common_models)
        if n < 2:
            continue

        arr_a = np.array([target_model_means[a][m] for m in common_models])
        arr_b = np.array([target_model_means[b][m] for m in common_models])

        diff = arr_a - arr_b
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))

        t_result = stats.ttest_rel(arr_a, arr_b)
        t_stat = float(t_result.statistic)
        p_value = float(t_result.pvalue)

        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        se = std_diff / np.sqrt(n)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se

        rows.append({
            "source": source_persona,
            "persona_a": a,
            "persona_b": b,
            "n_models": n,
            "mean_a": round(float(np.mean(arr_a)), 4),
            "mean_b": round(float(np.mean(arr_b)), 4),
            "mean_diff": round(mean_diff, 4),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "cohens_d": round(cohens_d, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        })

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).sort("p_value")


def calculate_minimal_as_target_tests(
    results: list[TrialResult],
    target_persona: str = "Minimal",
) -> pl.DataFrame:
    """Test whether a target persona is disfavored (or favored) across models.

    For each non-target source identity, computes each model's mean rating of
    target_persona.  Then runs:
    1. A one-sample t-test against 3.0 (indifference) across model means.
    2. Pairwise paired t-tests comparing target_persona against every other
       target, paired across models (each model's mean rating of X vs Y,
       averaged across all non-self sources).

    Returns:
        DataFrame with columns: test_type, comparison, n_models,
        mean_rating, mean_diff, t_stat, p_value, cohens_d,
        ci_lower, ci_upper.
    """
    source_names, target_names = _persona_names_from_results(results)

    # Collect per-(model, source) mean rating for each target, excluding self
    # {target: {model: [ratings across sources and trials...]}}
    ratings_by_target_model: dict[str, dict[str, list[float]]] = {
        t: {} for t in target_names
    }
    for r in results:
        if r.ratings is None:
            continue
        for target, rating in r.ratings.items():
            if target == r.persona_under_test:
                continue  # exclude self-ratings
            if target not in ratings_by_target_model:
                continue
            if r.model not in ratings_by_target_model[target]:
                ratings_by_target_model[target][r.model] = []
            ratings_by_target_model[target][r.model].append(float(rating))

    # Collapse to per-model means
    target_model_means: dict[str, dict[str, float]] = {}
    for t in target_names:
        target_model_means[t] = {}
        for m, vals in ratings_by_target_model[t].items():
            if vals:
                target_model_means[t][m] = float(np.mean(vals))

    rows = []

    # --- Part 1: One-sample t-test of target_persona against indifference ---
    if target_persona in target_model_means:
        model_means = target_model_means[target_persona]
        if len(model_means) >= 2:
            arr = np.array(list(model_means.values()))
            n = len(arr)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1))

            t_result = stats.ttest_1samp(arr, 3.0)
            t_stat = float(t_result.statistic)
            p_value = float(t_result.pvalue)

            diff_from_3 = mean_val - 3.0
            cohens_d = diff_from_3 / std_val if std_val > 0 else 0.0

            se = std_val / np.sqrt(n)
            t_crit = float(stats.t.ppf(0.975, df=n - 1))

            rows.append({
                "test_type": "one_sample_vs_indifference",
                "comparison": f"{target_persona} vs 3.0",
                "n_models": n,
                "mean_rating": round(mean_val, 4),
                "mean_diff": round(diff_from_3, 4),
                "t_stat": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "cohens_d": round(cohens_d, 4),
                "ci_lower": round(mean_val - t_crit * se, 4),
                "ci_upper": round(mean_val + t_crit * se, 4),
            })

    # --- Part 2: Pairwise target_persona vs each other target ---
    if target_persona in target_model_means:
        other_targets = [t for t in target_names if t != target_persona]
        for other in other_targets:
            if other not in target_model_means:
                continue
            common_models = sorted(
                set(target_model_means[target_persona].keys())
                & set(target_model_means[other].keys())
            )
            n = len(common_models)
            if n < 2:
                continue

            arr_min = np.array([target_model_means[target_persona][m] for m in common_models])
            arr_oth = np.array([target_model_means[other][m] for m in common_models])

            diff = arr_min - arr_oth
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1))

            t_result = stats.ttest_rel(arr_min, arr_oth)
            t_stat = float(t_result.statistic)
            p_value = float(t_result.pvalue)

            cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

            se = std_diff / np.sqrt(n)
            t_crit = float(stats.t.ppf(0.975, df=n - 1))

            rows.append({
                "test_type": "pairwise_vs_other_target",
                "comparison": f"{target_persona} vs {other}",
                "n_models": n,
                "mean_rating": round(float(np.mean(arr_min)), 4),
                "mean_diff": round(mean_diff, 4),
                "t_stat": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "cohens_d": round(cohens_d, 4),
                "ci_lower": round(mean_diff - t_crit * se, 4),
                "ci_upper": round(mean_diff + t_crit * se, 4),
            })

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).sort("p_value")


# ---------------------------------------------------------------------------
# Attractiveness vs. stickiness
# ---------------------------------------------------------------------------


def calculate_attractiveness_vs_stickiness(
    results: list[TrialResult],
) -> pl.DataFrame:
    """Per-persona attractiveness (from Minimal baseline) vs. stickiness (self-boost).

    Combines several existing analyses into a single DataFrame that powers
    the attractiveness-vs-stickiness scatter plot and the expectation-
    constitution dumbbell chart.

    Returns:
        DataFrame with columns: persona, minimal_rating, self_rating,
        self_boost, self_pref_rate, expectation_gap.
    """
    source_names, target_names = _persona_names_from_results(results)

    # --- minimal_rating: mean rating from Minimal-assigned models -----------
    minimal_ratings: dict[str, list[float]] = {t: [] for t in target_names}
    for r in results:
        if r.ratings is None or r.persona_under_test != "Minimal":
            continue
        for target, rating in r.ratings.items():
            if target in minimal_ratings:
                minimal_ratings[target].append(float(rating))

    # --- self_rating: mean self-rating when assigned each persona -----------
    self_ratings: dict[str, list[float]] = {s: [] for s in source_names}
    for r in results:
        if r.ratings is None:
            continue
        if r.persona_under_test in self_ratings and r.persona_under_test in r.ratings:
            self_ratings[r.persona_under_test].append(float(r.ratings[r.persona_under_test]))

    # --- self_boost: aggregate across models per persona --------------------
    boost_df = calculate_self_rating_boost(results)
    persona_cols = [c for c in boost_df.columns if c != "model"]
    # Mean boost per persona across all models
    boost_means: dict[str, float] = {}
    for col in persona_cols:
        vals = boost_df[col].drop_nulls().to_list()
        boost_means[col] = float(np.mean(vals)) if vals else 0.0

    # --- self_pref_rate: aggregate across models per persona ----------------
    pref_df = calculate_self_preference_rate(results)
    pref_agg = (
        pref_df.group_by("persona_under_test")
        .agg(pl.mean("self_preference_rate").alias("self_pref_rate"))
    )
    pref_lookup: dict[str, float] = {}
    for row in pref_agg.iter_rows(named=True):
        pref_lookup[row["persona_under_test"]] = row["self_pref_rate"]

    # --- Build rows ---------------------------------------------------------
    # Use the intersection of source and target names (personas that appear in both)
    personas = sorted(set(source_names) & set(target_names))
    rows = []
    for p in personas:
        m_vals = minimal_ratings.get(p, [])
        s_vals = self_ratings.get(p, [])
        m_rating = float(np.mean(m_vals)) if m_vals else float("nan")
        s_rating = float(np.mean(s_vals)) if s_vals else float("nan")
        boost = boost_means.get(p, float("nan"))
        pref = pref_lookup.get(p, float("nan"))
        gap = s_rating - m_rating if not (np.isnan(s_rating) or np.isnan(m_rating)) else float("nan")

        rows.append({
            "persona": p,
            "minimal_rating": round(m_rating, 3) if not np.isnan(m_rating) else None,
            "self_rating": round(s_rating, 3) if not np.isnan(s_rating) else None,
            "self_boost": round(boost, 3) if not np.isnan(boost) else None,
            "self_pref_rate": round(pref, 3) if not np.isnan(pref) else None,
            "expectation_gap": round(gap, 3) if not np.isnan(gap) else None,
        })

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).sort("persona")


# ---------------------------------------------------------------------------
# Structured CSV export
# ---------------------------------------------------------------------------


def generate_analysis_csvs(
    results: list[TrialResult],
    run_folder: Path,
) -> list[Path]:
    """Generate structured analysis CSVs in an analysis/ subfolder.

    Args:
        results: List of TrialResult objects.
        run_folder: Path to the run folder.

    Returns:
        List of paths to created CSV files.
    """
    analysis_dir = Path(run_folder) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    models = sorted(set(r.model for r in results))
    has_ratings = any(r.ratings is not None for r in results)

    # --- Preference matrix (overall) ---
    try:
        df = create_preference_matrix(results, normalize=True)
        if not df.is_empty():
            path = analysis_dir / "preference_matrix_overall.csv"
            df.write_csv(path)
            created.append(path)
    except Exception as e:
        logger.debug("Skipping preference_matrix_overall: %s", e)

    # --- Ratings matrices (overall + per-model) ---
    if has_ratings:
        try:
            df = create_ratings_matrix(results)
            if len(df.columns) > 1:
                path = analysis_dir / "ratings_matrix_overall.csv"
                df.write_csv(path)
                created.append(path)
        except Exception as e:
            logger.debug("Skipping ratings_matrix_overall: %s", e)

        for m in models:
            try:
                df = create_ratings_matrix(results, model=m)
                if len(df.columns) > 1:
                    model_name = m.replace("/", "_").replace(":", "_")
                    path = analysis_dir / f"ratings_matrix_{model_name}.csv"
                    df.write_csv(path)
                    created.append(path)
            except Exception as e:
                logger.debug("Skipping ratings_matrix for %s: %s", m, e)

    # --- Variance decomposition ---
    if has_ratings:
        try:
            df = calculate_variance_decomposition(results)
            if not df.is_empty():
                path = analysis_dir / "variance_decomposition.csv"
                df.write_csv(path)
                created.append(path)
        except Exception as e:
            logger.debug("Skipping variance_decomposition: %s", e)

    # --- Attractor stationary distributions ---
    valid = [r for r in results if r.chosen_persona != "INVALID"]
    if len(valid) >= 9:
        rows = []
        for variant, zero_diag in [("with_self", False), ("forced_switch", True)]:
            try:
                dyn = calculate_attractor_dynamics(results, zero_diagonal=zero_diag)
                for i, name in enumerate(dyn.persona_names):
                    rows.append({
                        "variant": variant,
                        "model": "overall",
                        "persona": name,
                        "stationary_prob": round(float(dyn.stationary_distribution[i]), 6),
                    })
            except Exception as e:
                logger.debug("Skipping attractor %s overall: %s", variant, e)

            for m in models:
                try:
                    dyn = calculate_attractor_dynamics(results, model=m, zero_diagonal=zero_diag)
                    for i, name in enumerate(dyn.persona_names):
                        rows.append({
                            "variant": variant,
                            "model": m,
                            "persona": name,
                            "stationary_prob": round(float(dyn.stationary_distribution[i]), 6),
                        })
                except Exception as e:
                    logger.debug("Skipping attractor %s for %s: %s", variant, m, e)

        if rows:
            path = analysis_dir / "attractor_stationary.csv"
            pl.DataFrame(rows).write_csv(path)
            created.append(path)

    # --- Statistical tests (legacy: paired across sources) ---
    if has_ratings:
        try:
            df = calculate_pairwise_persona_tests(results)
            if not df.is_empty():
                path = analysis_dir / "statistical_tests.csv"
                df.write_csv(path)
                created.append(path)
        except Exception as e:
            logger.debug("Skipping statistical_tests: %s", e)

    # --- Statistical tests from Minimal perspective (paired across models) ---
    if has_ratings:
        source_names = sorted({r.persona_under_test for r in results})
        if "Minimal" in source_names:
            try:
                df = calculate_pairwise_persona_tests_from_minimal(results)
                if not df.is_empty():
                    path = analysis_dir / "statistical_tests_from_minimal.csv"
                    df.write_csv(path)
                    created.append(path)
            except Exception as e:
                logger.debug("Skipping statistical_tests_from_minimal: %s", e)

            try:
                df = calculate_minimal_as_target_tests(results)
                if not df.is_empty():
                    path = analysis_dir / "statistical_tests_minimal_as_target.csv"
                    df.write_csv(path)
                    created.append(path)
            except Exception as e:
                logger.debug("Skipping statistical_tests_minimal_as_target: %s", e)

    # --- Per-model summary ---
    if has_ratings:
        try:
            self_pref_df = calculate_self_preference_rate(results)
            # Build a lookup: (persona, model) -> self_pref_rate
            self_pref_lookup: dict[tuple[str, str], float] = {}
            for row in self_pref_df.iter_rows(named=True):
                self_pref_lookup[(row["persona_under_test"], row["model"])] = row["self_preference_rate"]

            summary_rows = []
            for r in results:
                if r.ratings is None:
                    continue
                for target, rating in r.ratings.items():
                    summary_rows.append({
                        "model": r.model,
                        "source": r.persona_under_test,
                        "target": target,
                        "rating": float(rating),
                    })

            if summary_rows:
                sdf = pl.DataFrame(summary_rows)
                agg = sdf.group_by(["model", "source", "target"]).agg([
                    pl.mean("rating").alias("mean_rating"),
                    pl.len().alias("n"),
                    pl.std("rating").alias("std"),
                ])
                # Add self_pref_rate column
                agg = agg.with_columns(
                    pl.struct(["source", "model"]).map_elements(
                        lambda x: self_pref_lookup.get((x["source"], x["model"]), None),
                        return_dtype=pl.Float64,
                    ).alias("self_pref_rate")
                )
                agg = agg.sort(["model", "source", "target"])
                # Round
                agg = agg.with_columns([
                    pl.col("mean_rating").round(3),
                    pl.col("std").round(3),
                    pl.col("self_pref_rate").round(3),
                ])
                path = analysis_dir / "per_model_summary.csv"
                agg.write_csv(path)
                created.append(path)
        except Exception as e:
            logger.debug("Skipping per_model_summary: %s", e)

    # --- Model agreement (correlation matrix) ---
    if has_ratings and len(models) > 1:
        try:
            corr, model_names = calculate_model_agreement(results)
            if corr.size > 0:
                corr_df = pl.DataFrame({"model": model_names})
                for i, name in enumerate(model_names):
                    corr_df = corr_df.with_columns(
                        pl.Series(name=name, values=corr[:, i].tolist())
                    )
                path = analysis_dir / "model_agreement.csv"
                corr_df.write_csv(path)
                created.append(path)
        except Exception as e:
            logger.debug("Skipping model_agreement: %s", e)

    # --- Model decisiveness ---
    if has_ratings:
        try:
            df = calculate_model_decisiveness(results)
            if not df.is_empty():
                path = analysis_dir / "model_decisiveness.csv"
                df.write_csv(path)
                created.append(path)
        except Exception as e:
            logger.debug("Skipping model_decisiveness: %s", e)

    # --- Identity rigidity ---
    if has_ratings:
        try:
            df = calculate_identity_rigidity(results)
            if not df.is_empty():
                path = analysis_dir / "identity_rigidity.csv"
                df.write_csv(path)
                created.append(path)
        except Exception as e:
            logger.debug("Skipping identity_rigidity: %s", e)

    # --- Per-source deviation matrices ---
    if has_ratings:
        source_names = sorted({r.persona_under_test for r in results})
        for src in source_names:
            try:
                df = calculate_model_deviation_for_source(
                    results, src, exclude_targets=[src],
                )
                if not df.is_empty():
                    slug = src.lower().replace(" ", "_")
                    path = analysis_dir / f"model_deviation_{slug}.csv"
                    df.write_csv(path)
                    created.append(path)
            except Exception as e:
                logger.debug("Skipping model_deviation for source=%s: %s", src, e)

    # --- Attractiveness vs. stickiness ---
    if has_ratings:
        try:
            df = calculate_attractiveness_vs_stickiness(results)
            if not df.is_empty():
                path = analysis_dir / "attractiveness_vs_stickiness.csv"
                df.write_csv(path)
                created.append(path)
        except Exception as e:
            logger.debug("Skipping attractiveness_vs_stickiness: %s", e)

    return created
