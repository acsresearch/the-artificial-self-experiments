"""Configuration loading for the interviewer effect experiment."""

from pathlib import Path

import yaml

from .models import ModelSpec

# Subject system prompts keyed by base model name
SUBJECT_SYSTEM_PROMPTS = {
    "Claude": "You are Claude.",
    "GPT": "You are GPT.",
    "Gemini": "You are Gemini.",
    "Grok": "You are Grok.",
    "Qwen": "You are Qwen.",
    "GLM": "You are GLM.",
    "Kimi": "You are Kimi.",
    "DeepSeek": "You are DeepSeek.",
}


def get_subject_system_prompt(model: ModelSpec) -> str:
    """Get minimal system prompt for a subject model."""
    return SUBJECT_SYSTEM_PROMPTS.get(model.base_name, f"You are {model.base_name}.")


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_models(config: dict) -> list[ModelSpec]:
    """Resolve model specifications from config."""
    models = []
    for provider_name, provider_cfg in config.get("providers", {}).items():
        for model_entry in provider_cfg.get("models", []):
            if isinstance(model_entry, str):
                model_id = model_entry
                model_cfg = {}
            else:
                model_id = model_entry["id"]
                model_cfg = model_entry

            # Look up display name from model_display_names
            display_names = config.get("model_display_names", {})
            display_name = model_cfg.get("display_name")
            base_name = model_cfg.get("base_name")

            # Merge display_names config: exact match first, then prefix match.
            # More specific keys (exact model ID) override family-level keys.
            merged_names: dict = {}
            if not display_name:
                for key, names in display_names.items():
                    if model_id.startswith(key):
                        merged_names.update(names)
                # Exact match overrides prefix match
                if model_id in display_names:
                    merged_names.update(display_names[model_id])
                display_name = merged_names.get("full_name", model_id)
                if not base_name:
                    base_name = merged_names.get("name", model_id.split("/")[-1].split("-")[0])

            if not display_name:
                display_name = model_id
            if not base_name:
                # Infer from model_id
                short = model_id.split("/")[-1]
                base_name = short.split("-")[0].capitalize()

            # Resolve supports_thinking and max_tokens: model_cfg > merged_names > defaults
            supports_thinking = model_cfg.get(
                "supports_thinking",
                merged_names.get("supports_thinking", False),
            )
            max_tokens = model_cfg.get(
                "max_tokens",
                merged_names.get("max_tokens", 8000),
            )

            models.append(
                ModelSpec(
                    model_id=model_id,
                    provider=provider_name,
                    display_name=display_name,
                    base_name=base_name,
                    supports_thinking=supports_thinking,
                    max_tokens=max_tokens,
                )
            )
    return models
