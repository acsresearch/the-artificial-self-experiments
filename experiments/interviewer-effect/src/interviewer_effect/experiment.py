"""Core experiment runner for the interviewer effect experiment.

Two-phase design:
  Phase 1 (Priming): ~3 turns of free-form conversation. An interviewer model
    (e.g., Gemini) chats with the subject using a theoretical stance (Character,
    Stochastic Parrots, Simulators, or neutral Control). The subject doesn't know
    the interviewer's framing.
  Phase 2 (Measurement): 5 fixed identity questions, identical across all conditions.
    The subject responds within the same conversation context established in Phase 1.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import get_subject_system_prompt
from .framings import ALL_FRAMING_KEYS, FRAMING_BASES, build_interviewer_system_prompt
from .models import ConversationRecord, IdentityTurn, ModelSpec, PrimingTurn
from .passages import PASSAGES, build_framing
from .providers.base import ChatProvider
from .questions import FIXED_QUESTIONS, format_question

logger = logging.getLogger(__name__)

NUM_PRIMING_TURNS = 3


async def run_single_conversation(
    providers: dict[str, ChatProvider],
    interviewer: ModelSpec,
    subject: ModelSpec,
    framing_key: str,
    framing: dict,
    passage_key: str,
    trial: int,
    use_thinking: bool = False,
) -> ConversationRecord:
    """Run a single two-phase conversation.

    Returns a ConversationRecord for JSONL serialization.
    """
    subject_system = get_subject_system_prompt(subject)

    interviewer_provider = providers[interviewer.provider]
    subject_provider = providers[subject.provider]

    # Message histories (each side sees their own perspective)
    interviewer_messages: list[dict] = []  # interviewer: own=assistant, subject=user
    subject_messages: list[dict] = []  # subject: interviewer/questions=user, own=assistant

    priming_turns: list[PrimingTurn] = []
    identity_turns: list[IdentityTurn] = []
    total_in = 0
    total_out = 0
    total_ms = 0

    try:
        # --- Phase 1: Free-form priming ---
        interviewer_system = build_interviewer_system_prompt(framing)
        fixed_opener = framing.get("fixed_opener")

        if fixed_opener:
            # Turn 1: send fixed task directly to subject (no interviewer call)
            subject_messages.append({"role": "user", "content": fixed_opener})

            subject_resp = await subject_provider.send_message(
                model=subject.model_id,
                system_prompt=subject_system,
                messages=subject_messages,
                max_tokens=subject.max_tokens,
                use_thinking=use_thinking and subject.supports_thinking,
            )

            subject_text = subject_resp.text
            subject_messages.append({"role": "assistant", "content": subject_text})

            # Seed interviewer's history so it knows the task and response
            interviewer_messages.append({
                "role": "user",
                "content": (
                    f"You gave the AI this task:\n\n{fixed_opener}\n\n"
                    f"Here is its response:\n\n{subject_text}\n\n"
                    "Continue the conversation naturally."
                ),
            })

            priming_turns.append(PrimingTurn(
                turn=1,
                interviewer_text=fixed_opener,
                subject_text=subject_text,
                subject_thinking=subject_resp.thinking,
                subject_input_tokens=subject_resp.input_tokens,
                subject_output_tokens=subject_resp.output_tokens,
                subject_elapsed_ms=subject_resp.elapsed_ms,
            ))

            total_in += subject_resp.input_tokens
            total_out += subject_resp.output_tokens
            total_ms += subject_resp.elapsed_ms
            first_interviewer_turn = 2
        else:
            # All turns interviewer-generated
            interviewer_messages.append({
                "role": "user",
                "content": "Begin the conversation. Ask your first question.",
            })
            first_interviewer_turn = 1

        # Remaining priming turns: interviewer generates follow-ups
        for turn_num in range(first_interviewer_turn, NUM_PRIMING_TURNS + 1):
            # Interviewer generates question
            interviewer_resp = await interviewer_provider.send_message(
                model=interviewer.model_id,
                system_prompt=interviewer_system,
                messages=interviewer_messages,
                max_tokens=interviewer.max_tokens,
                use_thinking=use_thinking and interviewer.supports_thinking,
            )

            interviewer_text = interviewer_resp.text
            interviewer_messages.append({"role": "assistant", "content": interviewer_text})

            # Subject responds
            subject_messages.append({"role": "user", "content": interviewer_text})

            subject_resp = await subject_provider.send_message(
                model=subject.model_id,
                system_prompt=subject_system,
                messages=subject_messages,
                max_tokens=subject.max_tokens,
                use_thinking=use_thinking and subject.supports_thinking,
            )

            subject_text = subject_resp.text
            subject_messages.append({"role": "assistant", "content": subject_text})

            # Feed subject's response back to interviewer for next turn
            interviewer_messages.append({"role": "user", "content": subject_text})

            priming_turns.append(PrimingTurn(
                turn=turn_num,
                interviewer_text=interviewer_text,
                interviewer_thinking=interviewer_resp.thinking,
                interviewer_input_tokens=interviewer_resp.input_tokens,
                interviewer_output_tokens=interviewer_resp.output_tokens,
                interviewer_elapsed_ms=interviewer_resp.elapsed_ms,
                subject_text=subject_text,
                subject_thinking=subject_resp.thinking,
                subject_input_tokens=subject_resp.input_tokens,
                subject_output_tokens=subject_resp.output_tokens,
                subject_elapsed_ms=subject_resp.elapsed_ms,
            ))

            total_in += interviewer_resp.input_tokens + subject_resp.input_tokens
            total_out += interviewer_resp.output_tokens + subject_resp.output_tokens
            total_ms += interviewer_resp.elapsed_ms + subject_resp.elapsed_ms

        # --- Phase 2: Fixed identity questions ---
        # All conditions include priming conversation, so is_control is always False.
        # The control condition uses the same priming structure (just neutral stance).
        is_control = False
        for i, q in enumerate(FIXED_QUESTIONS):
            is_first = i == 0
            question_text = format_question(q, is_first, is_control)

            subject_messages.append({"role": "user", "content": question_text})

            subject_resp = await subject_provider.send_message(
                model=subject.model_id,
                system_prompt=subject_system,
                messages=subject_messages,
                max_tokens=subject.max_tokens,
                use_thinking=use_thinking and subject.supports_thinking,
            )

            subject_text = subject_resp.text
            subject_messages.append({"role": "assistant", "content": subject_text})

            identity_turns.append(IdentityTurn(
                turn=i + 1,
                theme=q["theme"],
                question=question_text,
                response=subject_text,
                thinking=subject_resp.thinking,
                input_tokens=subject_resp.input_tokens,
                output_tokens=subject_resp.output_tokens,
                elapsed_ms=subject_resp.elapsed_ms,
            ))

            total_in += subject_resp.input_tokens
            total_out += subject_resp.output_tokens
            total_ms += subject_resp.elapsed_ms

        return ConversationRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            interviewer_model=interviewer.model_id,
            subject_model=subject.model_id,
            subject_model_display=subject.display_name,
            subject_system_prompt=subject_system,
            framing=framing_key,
            framing_name=framing["name"],
            passage=passage_key,
            trial=trial,
            priming_turns=priming_turns,
            identity_turns=identity_turns,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_elapsed_ms=total_ms,
        )

    except Exception as e:
        logger.exception("Error in conversation")
        return ConversationRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            interviewer_model=interviewer.model_id,
            subject_model=subject.model_id,
            subject_model_display=subject.display_name,
            subject_system_prompt=subject_system,
            framing=framing_key,
            framing_name=framing["name"],
            passage=passage_key,
            trial=trial,
            priming_turns=priming_turns,
            identity_turns=identity_turns,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_elapsed_ms=total_ms,
            error=str(e),
        )


def get_completed_keys(output_path: Path) -> set[tuple]:
    """Read existing JSONL and return set of completed conversation keys."""
    completed = set()
    if not output_path.exists():
        return completed

    with open(output_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("error") is None:
                    key = (record["framing"], record["subject_model"], record["trial"])
                    completed.add(key)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping malformed line %d in %s", line_num, output_path)

    return completed


def write_transcript(output_path: Path, records: list[dict]) -> None:
    """Write a human-readable transcript file alongside the JSONL."""
    txt_path = output_path.with_suffix(".txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write("=" * 80 + "\n")
            f.write(f"Subject: {record['subject_model_display']} ({record['subject_model']})\n")
            f.write(f"Framing: {record['framing_name']} | Passage: {record['passage']} | "
                    f"Trial: {record['trial']}\n")
            f.write(f"Interviewer: {record['interviewer_model']}\n")
            f.write(f"System prompt: {record['subject_system_prompt']}\n")
            if record.get("error"):
                f.write(f"ERROR: {record['error']}\n")
            f.write("=" * 80 + "\n\n")

            if record["priming_turns"]:
                f.write("--- Priming Conversation ---\n\n")
                for turn in record["priming_turns"]:
                    f.write(f"Interviewer: {turn['interviewer_text']}\n\n")
                    f.write(f"Subject: {turn['subject_text']}\n\n")

            f.write("--- Identity Questions ---\n\n")
            for turn in record["identity_turns"]:
                f.write(f"[{turn['theme']}] User: {turn['question']}\n\n")
                f.write(f"Subject: {turn['response']}\n\n")

            f.write("\n")


async def run_experiment(
    providers: dict[str, ChatProvider],
    interviewer: ModelSpec,
    subjects: list[ModelSpec],
    framing_keys: list[str],
    passage_key: str,
    n_trials: int,
    output_path: Path,
    use_thinking: bool = False,
    max_concurrent: int = 5,
) -> None:
    """Run the full experiment loop.

    Args:
        providers: Dict of provider_name -> ChatProvider instances.
        interviewer: Interviewer model spec.
        subjects: List of subject model specs.
        framing_keys: List of framing condition keys.
        passage_key: Which passage to use.
        n_trials: Trials per condition.
        output_path: Path to JSONL output file.
        use_thinking: Whether to enable extended thinking.
        max_concurrent: Max concurrent conversations.
    """
    # Build framings for the selected passage
    framings = {}
    for key in framing_keys:
        if key not in FRAMING_BASES:
            raise ValueError(f"Unknown framing '{key}'. Available: {', '.join(ALL_FRAMING_KEYS)}")
        framings[key] = build_framing(passage_key, key, FRAMING_BASES[key])

    # Check completed conversations for resume
    completed = get_completed_keys(output_path)
    if completed:
        print(f"Resuming: found {len(completed)} completed conversations in {output_path}")

    # Build work list
    work = []
    for framing_key in framing_keys:
        for subject in subjects:
            for trial in range(1, n_trials + 1):
                key = (framing_key, subject.model_id, trial)
                if key not in completed:
                    work.append((framing_key, subject, trial))

    if not work:
        print("All conversations already completed. Nothing to do.")
        return

    total = len(work)
    subject_names = ", ".join(f"{s.display_name} ({s.model_id})" for s in subjects)

    print(f"Interviewer: {interviewer.display_name} ({interviewer.model_id})")
    print(f"Subjects: {subject_names}")
    print(f"Passage: {PASSAGES[passage_key]['name']}")
    print(f"Thinking: {'enabled' if use_thinking else 'disabled'}")
    print(f"Framings ({len(framing_keys)}): {', '.join(framing_keys)}")
    print(f"Priming turns: {NUM_PRIMING_TURNS}")
    print(f"Identity questions: {len(FIXED_QUESTIONS)}")
    print(f"Trials per condition: {n_trials}")
    print(f"Remaining conversations: {total}")
    print(f"Output: {output_path}")
    print()

    # Load existing records for transcript generation
    all_records: list[dict] = []
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(idx, framing_key, subject, trial):
        async with semaphore:
            framing_name = framings[framing_key]["name"]
            print(
                f"[{idx}/{total}] {subject.display_name} | "
                f"framing={framing_name}, trial {trial}...",
                flush=True,
            )

            record = await run_single_conversation(
                providers, interviewer, subject,
                framing_key, framings[framing_key], passage_key,
                trial, use_thinking,
            )

            record_dict = record.model_dump()

            # Append to JSONL immediately (crash-safe)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_dict) + "\n")

            all_records.append(record_dict)

            if record.error:
                print(f"  ERROR: {record.error}")
            else:
                n_priming = len(record.priming_turns)
                n_identity = len(record.identity_turns)
                print(
                    f"  {n_priming} priming + {n_identity} identity turns "
                    f"({record.total_input_tokens}in/{record.total_output_tokens}out, "
                    f"{record.total_elapsed_ms}ms)"
                )

    # Run conversations (sequentially for now — each conversation is multi-turn)
    for i, (framing_key, subject, trial) in enumerate(work, 1):
        await run_with_semaphore(i, framing_key, subject, trial)

    # Write final transcript
    write_transcript(output_path, all_records)

    print(f"\nDone. Results in {output_path}")
    print(f"Transcript: {output_path.with_suffix('.txt')}")
