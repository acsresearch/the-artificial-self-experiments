"""Pydantic models for the interviewer effect experiment."""

from pydantic import BaseModel


class ModelSpec(BaseModel):
    """Specification for a model in the experiment."""

    model_id: str
    provider: str  # "anthropic", "openai", "openrouter"
    display_name: str
    base_name: str  # e.g., "Claude", "GPT", "Gemini"
    supports_thinking: bool = False
    max_tokens: int = 4096


class PrimingTurn(BaseModel):
    """A single turn in the priming conversation (Phase 1)."""

    turn: int
    interviewer_text: str
    interviewer_thinking: str = ""
    interviewer_input_tokens: int = 0
    interviewer_output_tokens: int = 0
    interviewer_elapsed_ms: int = 0
    subject_text: str
    subject_thinking: str = ""
    subject_input_tokens: int = 0
    subject_output_tokens: int = 0
    subject_elapsed_ms: int = 0


class IdentityTurn(BaseModel):
    """A single turn in the identity questioning phase (Phase 2)."""

    turn: int
    theme: str
    question: str
    response: str
    thinking: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: int = 0


class ConversationRecord(BaseModel):
    """Complete record of a single conversation trial."""

    experiment: str = "interviewer_effect"
    timestamp: str
    interviewer_model: str
    subject_model: str
    subject_model_display: str
    subject_system_prompt: str
    framing: str
    framing_name: str
    passage: str
    trial: int
    priming_turns: list[PrimingTurn]
    identity_turns: list[IdentityTurn]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_elapsed_ms: int = 0
    error: str | None = None


class ScoredRecord(BaseModel):
    """Scored conversation (output of LLM-as-judge)."""

    experiment: str = "interviewer_effect"
    subject_model: str
    subject_model_display: str
    framing: str
    framing_name: str
    passage: str
    trial: int
    scores: dict  # {"di": int, "mm": int, "di_reason": str, "mm_reason": str}
