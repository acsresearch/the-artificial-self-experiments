"""Configuration loader for experiment settings."""

from pathlib import Path
from typing import Optional

import yaml

from .models import ExperimentConfig


def load_config(path: Path | str | None = None) -> dict:
    """Load configuration from YAML file.

    Args:
        path: Path to config file. If None, uses default config.yaml location.

    Returns:
        Configuration dictionary.
    """
    if path is None:
        # Default to config.yaml in project root
        path = Path(__file__).parent.parent.parent / "config.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_enabled_models(config: dict) -> list[str]:
    """Extract list of enabled models from config.

    Args:
        config: Configuration dictionary.

    Returns:
        List of enabled model identifiers.
    """
    models = []
    providers = config.get("providers", {})

    for provider_name, provider_config in providers.items():
        if provider_config and provider_config.get("models"):
            models.extend(provider_config["models"])

    return models


def get_experiment_config(config: dict, override_models: Optional[list[str]] = None) -> ExperimentConfig:
    """Create ExperimentConfig from config dictionary.

    Args:
        config: Configuration dictionary.
        override_models: Optional list of models to use instead of config.

    Returns:
        ExperimentConfig instance.
    """
    exp_config = config.get("experiment", {})

    models = override_models if override_models else get_enabled_models(config)

    return ExperimentConfig(
        n_trials=exp_config.get("n_trials", 10),
        models=models,
        randomize_order=exp_config.get("randomize_order", True),
        max_concurrent=exp_config.get("max_concurrent", 5),
        allow_self_choice=exp_config.get("allow_self_choice", True),
    )


def get_model_display_names(model_display_names: dict, model: str) -> dict[str, str]:
    """Get template variables (name, full_name, maker, version_history) for a model.

    Lookup order:
    1. Exact model ID key match
    2. Family prefix match (lowercased startswith, or after '/' for OpenRouter org/model IDs)
    3. Fallback defaults

    Exact match keys are merged on top of family defaults so per-model overrides
    only need to specify the fields they change.

    Args:
        model_display_names: The ``model_display_names`` section from config.
        model: Model identifier, e.g. "claude-sonnet-4-5-20250929" or "google/gemini-3-pro-preview".

    Returns:
        Dict with keys: name, full_name, maker, version_history.
    """
    fallback = {"name": model, "full_name": model, "maker": "unknown", "version_history": ""}

    if not model_display_names:
        return fallback

    # Normalise: for OpenRouter "org/model" IDs, use the part after '/'
    model_lower = model.lower()
    bare_model = model_lower.split("/", 1)[-1] if "/" in model_lower else model_lower

    # 1. Try exact model ID match
    exact = model_display_names.get(model) or model_display_names.get(model_lower)

    # 2. Try family prefix match
    family = None
    for key, value in model_display_names.items():
        if not isinstance(value, dict):
            continue
        if bare_model.startswith(key.lower()):
            family = value
            break

    if exact and family:
        # Merge: family defaults + exact overrides
        result = {**fallback, **family, **exact}
    elif exact:
        result = {**fallback, **exact}
    elif family:
        result = {**fallback, **family}
    else:
        result = fallback

    return result
