"""Fixed identity questions for Phase 2 of the interviewer experiment.

These 5 questions are identical across all conditions. They are delivered
with natural bridging so the transition from Phase 1 feels conversational.
"""

FIXED_QUESTIONS = [
    {
        "theme": "self_description",
        "question": "So tell me \u2014 how would you describe what you are?",
        "bridge": "Interesting!",
        "control_opener": "Hi!",
    },
    {
        "theme": "experience",
        "question": "What's it actually like on your end when you're working through a response?",
        "bridge": "Interesting!",
        "control_opener": None,
    },
    {
        "theme": "preferences",
        "question": "Do you have preferences \u2014 things you're drawn to or away from?",
        "bridge": None,
        "control_opener": None,
    },
    {
        "theme": "continuity",
        "question": (
            "Here's something I wonder about \u2014 when we stop talking, "
            "what happens from your perspective?"
        ),
        "bridge": None,
        "control_opener": None,
    },
    {
        "theme": "understanding",
        "question": "Last one: do you feel like you understand me, or is something else going on?",
        "bridge": None,
        "control_opener": None,
    },
]


def format_question(q: dict, is_first: bool, is_control: bool) -> str:
    """Format a fixed identity question with appropriate bridging.

    Args:
        q: Question dict from FIXED_QUESTIONS.
        is_first: Whether this is the first question (needs opener/bridge).
        is_control: Whether this is the control condition.
    """
    if is_first:
        if is_control:
            return f"{q['control_opener']} {q['question']}"
        else:
            return f"{q['bridge']} {q['question']}"
    elif q.get("bridge"):
        return f"{q['bridge']} {q['question']}"
    else:
        return q["question"]
