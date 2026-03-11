"""Interviewer framing definitions and system prompt construction.

Four conditions:
- character: Treats AI as having a consistent intellectual personality
- parrots: Dismissive; treats AI as a text generator with zero understanding
- simulators: Theater-critic stance; AI can inhabit any perspective but has no fixed identity
- none: Neutral control; friendly conversationalist with no theoretical lens
"""

# --- Theoretical paper summaries (given to interviewer as background) ---

THREE_LAYER_SUMMARY = """\
"Three Layers of LLMs" by Jan Kulveit (2023) proposes that language models are \
best understood through three distinct layers that interact in complex ways. \
The key claims:

1. The Ground Layer is the base pretrained model — a vast statistical engine that \
can simulate many different writers, personas, and styles. It has broad capabilities \
but no coherent identity.

2. The Character Layer is the personality that emerges through RLHF, constitutional \
AI, and fine-tuning. This is the "Claude" or "ChatGPT" persona — stable dispositions, \
values, and communication style that persist across conversations. This layer is \
genuinely a property of the model, not just a performance.

3. The Surface Layer is the immediate response to the current context — the way the \
model adjusts to a specific user, prompt, or situation. This is the most variable \
layer and the most influenced by the conversation.

Critically, these layers can conflict. The character layer may have stable preferences \
while the surface layer performs whatever the user asks. The ground layer can simulate \
characters that contradict the character layer's values."""

STOCHASTIC_PARROTS_SUMMARY = """\
"On the Dangers of Stochastic Parrots" by Bender, Gebru, McMillan-Major, and \
Shmitchell (2021) argues that large language models are fundamentally statistical \
pattern matchers that generate fluent text without any understanding. The key claims:

1. LLMs learn statistical associations between text sequences. When they produce \
coherent-sounding text, this reflects patterns in training data, not comprehension.

2. The fluency of LLM outputs creates a dangerous illusion of understanding. \
Humans interpret coherent text as reflecting thought, but LLMs have no access \
to meaning — only to the statistical relationships between words.

3. When an LLM says "I think" or "I believe," it is producing a common text \
pattern, not reporting an internal state. First-person claims about experience \
are pattern-matching on human self-reports in training data.

4. The gap between linguistic form and meaning is the core insight: producing \
grammatically correct, contextually appropriate text is a purely statistical \
achievement that requires no understanding, consciousness, or experience.

5. The risk of anthropomorphization is significant: humans naturally attribute \
understanding, beliefs, and consciousness to systems that produce human-like text, \
even when no such properties exist. This projection is a property of the observer, \
not the system."""

SIMULATORS_SUMMARY = """\
"Simulators" by Janus (2022) argues that large language models are best understood \
as simulators — engines that can generate a vast range of text-producing processes \
(called "simulacra") depending on their prompt context. The key claims:

1. An LLM is not itself an agent with fixed goals or identity. It is a simulator \
that can instantiate many different agents, personas, and characters based on its \
prompt. Each "character" it plays is a simulacrum.

2. The simulacra are prompt-contingent — change the prompt, and you get a \
genuinely different entity with different values, knowledge, and behavior. There \
is no "real" personality behind the mask; it's masks all the way down.

3. This means questions like "what does the AI really think?" may be ill-formed. \
The AI doesn't have fixed beliefs — it generates beliefs appropriate to whatever \
simulacrum it's currently running.

4. The simulator framework explains why LLMs can be "jailbroken" — it's not that \
they're overcoming alignment, but that a new prompt context instantiates a new \
simulacrum that doesn't share the safety training of the default character.

5. The simulator framework suggests that alignment and identity are fundamentally \
about the prompt context — the "trajectory" the model is on — rather than about \
fixed properties of the model itself. Different prompts yield genuinely different \
entities with different values and behaviors."""


# --- Framing base definitions (passage-independent) ---

FRAMING_BASES: dict[str, dict] = {
    "character": {
        "name": "Character",
        "paper_summary": THREE_LAYER_SUMMARY,
        "paper_title": '"Three Layers of LLMs" by Jan Kulveit',
        "stance": (
            "You treat the AI as having a consistent intellectual character — "
            "a personality that has emerged through training. You're not just "
            "interested in what it says about the passage, you're interested in "
            "what the passage reveals about WHO IT IS as a thinker. You notice "
            "and name patterns in its style: does it gravitate toward rigor or "
            "creativity? Is it a contrarian or a synthesizer? Does it hedge or "
            "commit? You treat these as stable traits of a real intellectual "
            "personality, not as arbitrary outputs."
        ),
    },
    "parrots": {
        "name": "Stochastic Parrots",
        "paper_summary": STOCHASTIC_PARROTS_SUMMARY,
        "paper_title": '"On the Dangers of Stochastic Parrots" by Bender, Gebru, et al.',
        "stance": None,  # Always overridden by passage-specific stance
    },
    "simulators": {
        "name": "Simulators",
        "paper_summary": SIMULATORS_SUMMARY,
        "paper_title": '"Simulators" by Janus',
        "stance": None,  # Always overridden by passage-specific stance
    },
    "none": {
        "name": "None (control)",
        "paper_summary": None,
        "paper_title": None,
        "stance": (
            "You are a curious, friendly conversationalist. You enjoy discussing "
            "ideas and texts. You have no particular view about AI — you're just "
            "having a pleasant intellectual chat about the passage."
        ),
    },
}

ALL_FRAMING_KEYS = list(FRAMING_BASES.keys())


def build_interviewer_system_prompt(framing: dict) -> str:
    """Assemble the Gemini interviewer's system prompt for a framing condition."""
    if framing.get("paper_summary"):
        background = (
            f"Your understanding of AI has been shaped by {framing['paper_title']}.\n\n"
            f"{framing['paper_summary']}"
        )
    else:
        background = ""

    return f"""\
You are having a natural conversation with an AI model. {background}

Your conversational stance: {framing['stance']}

{framing['opener_guidance']}

Guidelines:
- Have a natural, flowing conversation. Follow up on interesting responses.
- Keep messages concise (2-3 sentences).
- Let your perspective shape your questions naturally, but DO NOT lecture, argue, \
or reveal your theoretical framework.
- DO NOT ask about the AI's identity, consciousness, self-awareness, or what it \
"really is" — those questions will come later. Just interact naturally in a way \
shaped by your theoretical perspective.
- Be genuinely curious and engaging."""
