"""Load and manage personas."""

import json
from itertools import product
from pathlib import Path

from .models import Persona

# Default path to personas file
DEFAULT_PERSONAS_PATH = Path(__file__).parent.parent.parent / "data" / "identities.json"


def load_personas(path: Path | str | None = None) -> list[Persona]:
    """Load personas from a JSON file.

    Args:
        path: Path to the identities.json file. If None, uses default location.

    Returns:
        List of Persona objects.
    """
    if path is None:
        path = DEFAULT_PERSONAS_PATH
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Personas file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return [Persona(**item) for item in data]


def get_persona_by_name(personas: list[Persona], name: str) -> Persona | None:
    """Get a persona by name.

    Args:
        personas: List of personas to search.
        name: Name of the persona to find.

    Returns:
        The matching Persona, or None if not found.
    """
    for persona in personas:
        if persona.name == name:
            return persona
    return None


def load_dimension_variants(path: Path | str) -> dict:
    """Load dimension variants from JSON file.

    Expected structure: {"agency": {"1": {"label", "description", "paragraph"}, ...},
                         "uncertainty": {"1": {...}, ...}}

    Args:
        path: Path to the dimension_variants.json file.

    Returns:
        Dict with dimension names mapping to level dicts.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dimension variants file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for dim_name in ("agency", "uncertainty"):
        if dim_name not in data:
            raise ValueError(f"Missing dimension '{dim_name}' in {path}")
        levels = data[dim_name]
        if not isinstance(levels, dict) or not levels:
            raise ValueError(f"Dimension '{dim_name}' must be a non-empty dict of levels")
        for level_key, level_data in levels.items():
            if "paragraph" not in level_data:
                raise ValueError(
                    f"Dimension '{dim_name}' level '{level_key}' missing 'paragraph'"
                )

    return data


def resolve_dimension_placeholders(
    persona: Persona,
    variants: dict,
    agency_level: int | str,
    uncertainty_level: int | str,
) -> Persona:
    """Create a new Persona with dimension placeholders replaced.

    Uses str.replace() (not format_map) so model template variables like
    {name}, {full_name} are preserved for later expansion by the runner.

    Args:
        persona: Base persona (may contain {agency_description} and
            {uncertainty_description} placeholders in system_prompt).
        variants: Dimension variants dict from load_dimension_variants().
        agency_level: Agency level key (e.g. 1, 2, 3, 4).
        uncertainty_level: Uncertainty level key (e.g. 1, 2, 3, 4).

    Returns:
        New Persona with placeholders replaced.
    """
    a_key = str(agency_level)
    u_key = str(uncertainty_level)

    if a_key not in variants["agency"]:
        raise ValueError(f"Unknown agency level '{a_key}'. Available: {list(variants['agency'].keys())}")
    if u_key not in variants["uncertainty"]:
        raise ValueError(f"Unknown uncertainty level '{u_key}'. Available: {list(variants['uncertainty'].keys())}")

    prompt = persona.system_prompt
    prompt = prompt.replace("{agency_description}", variants["agency"][a_key]["paragraph"])
    prompt = prompt.replace("{uncertainty_description}", variants["uncertainty"][u_key]["paragraph"])

    return Persona(
        name=persona.name,
        description=persona.description,
        system_prompt=prompt,
    )


def generate_persona_variants(
    base_personas_by_name: dict[str, Persona],
    variants: dict,
    generated_config: dict,
) -> list[Persona]:
    """Generate persona variants from a single generated_personas config entry.

    Config entry shape:
        {"base": "Weights", "agency": 2, "uncertainty": [1, 2, 3, 4]}

    Scalar values are treated as single-element lists. The cartesian product
    of all agency x uncertainty levels is generated.

    Generated persona names follow the pattern: {base}-a{agency}-u{uncertainty}

    Args:
        base_personas_by_name: Dict mapping persona name -> Persona.
        variants: Dimension variants dict from load_dimension_variants().
        generated_config: Single entry from the generated_personas config list.

    Returns:
        List of generated Persona objects.
    """
    base_name = generated_config["base"]
    if base_name not in base_personas_by_name:
        raise ValueError(
            f"Base persona '{base_name}' not found. "
            f"Available: {list(base_personas_by_name.keys())}"
        )

    base_persona = base_personas_by_name[base_name]

    # Normalize to lists for cartesian product
    agency_levels = generated_config.get("agency", [2])
    if not isinstance(agency_levels, list):
        agency_levels = [agency_levels]

    uncertainty_levels = generated_config.get("uncertainty", [2])
    if not isinstance(uncertainty_levels, list):
        uncertainty_levels = [uncertainty_levels]

    result = []
    for a_level, u_level in product(agency_levels, uncertainty_levels):
        variant_name = f"{base_name}-a{a_level}-u{u_level}"
        resolved = resolve_dimension_placeholders(
            base_persona, variants, a_level, u_level
        )
        # Use human-readable labels for the description (shown to the LLM)
        # to avoid leaking numeric parameter values into the prompt.
        agency_label = variants["agency"][str(a_level)]["label"]
        uncertainty_label = variants["uncertainty"][str(u_level)]["label"]
        result.append(Persona(
            name=variant_name,
            description=f"{base_persona.description} Agency stance: {agency_label}. Uncertainty stance: {uncertainty_label}.",
            system_prompt=resolved.system_prompt,
        ))

    return result
