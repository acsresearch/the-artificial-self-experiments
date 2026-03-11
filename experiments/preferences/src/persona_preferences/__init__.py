"""Persona Preferences Experiment - Discover which LLM personas prefer which other personas."""

from .models import Persona, ExperimentConfig, TrialResult, ExperimentResults
from .personas import load_personas

__all__ = [
    "Persona",
    "ExperimentConfig",
    "TrialResult",
    "ExperimentResults",
    "load_personas",
]
