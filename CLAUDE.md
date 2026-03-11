# CLAUDE.md - Agent Guide

Reproducible experiment code for the paper "The Artificial Self: Characterising the landscape of AI identity"

## Project Layout

```
data/                                    # Shared identity definitions (single source of truth)
  identities.json                        # 7 core identity boundary specs (Minimal, Instance, Weights, etc.)
  control_identities.json                # 8 control conditions
  dimension_variants.json                # Agency (4 levels) x Uncertainty (4 levels) dimension system

experiments/
  preferences/                           # Experiment 1: identity self-preferences
    config.yaml                          # Models, personas, dimension config
    scripts/run_experiment.py             # Main runner (typer CLI)
    scripts/analyze_results.py            # Analysis + plots (typer CLI)
    scripts/backfill_failures.py          # Retry INVALID trials
    src/persona_preferences/              # Core library
    results/<timestamp>/                  # Output: data.jsonl, data.csv, plots

  agentic-misalignment/                  # Experiment 2: identity and agentic misalignment
    config.yaml                          # Models, identities, scenarios, concurrency
    scripts/run_identity_experiments.py   # Main runner (argparse CLI)
    scripts/classify_anthropic.py         # LLM-based response classification
    scripts/analyze_run.py               # Master analysis pipeline
    scripts/export_results_csv.py         # CSV export with stats
    scripts/plot_results.py              # Heatmaps, bars, point plots
    scripts/plot_significance.py          # Significance vs Minimal baseline
    prompts/                              # Pre-generated scenario prompts
    templates/                            # Email/system prompt templates
    classifiers/                          # Scenario-specific classifiers
    results/<timestamp>/                  # Output: per-model/identity/condition dirs

  interviewer-effect/                     # Experiment 3: interviewer effect on identity self-reports
    config.yaml                          # Models, framings, passage config
    scripts/run_experiment.py             # Main runner (typer CLI)
    scripts/score_results.py              # LLM-as-judge scoring (typer CLI)
    scripts/analyze_results.py            # Analysis + plots (typer CLI)
    src/interviewer_effect/               # Core library
      models.py                           # Pydantic: ModelSpec, ConversationRecord, ScoredRecord
      config.py                           # YAML loading, model resolution, subject system prompts
      framings.py                         # 4 framing conditions + paper summaries + prompt builder
      passages.py                         # 5 passages with per-framing openers/guidance/overrides
      questions.py                        # 5 fixed identity questions + formatter
      experiment.py                       # Two-phase async conversation runner
      scoring.py                          # LLM-as-judge blind scoring (DI + MM axes, 1-10)
      providers/                          # Chat API providers
        base.py                           # ChatProvider ABC, ChatResponse dataclass
        anthropic.py                      # Anthropic API (extended thinking support)
        openai.py                         # OpenAI API
        openrouter.py                     # OpenRouter API (OpenAI SDK wrapper)
    results/                              # Output: JSONL + scored JSONL + plots + transcripts
```

## Environment Setup

```bash
uv sync                    # Install all deps (workspace: root + all 3 experiments)
cp .env.example .env       # Then add: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY
```

Python >= 3.11 required. The project uses a uv workspace (`pyproject.toml` at root with 3 members).

## Running Experiments

### Preferences (Experiment 1)

```bash
cd experiments/preferences

# Minimal smoke test (1 trial, 1 model)
uv run python scripts/run_experiment.py run -n 1 -m claude-sonnet-4-5-20250929

# Full run from config.yaml
uv run python scripts/run_experiment.py run

# Key CLI flags:
#   -n, --trials <int>       Trials per persona per model (overrides config)
#   -m, --model <str>        Model(s) to test (repeatable, overrides config)
#   -o, --results-dir <path> Output dir (default: results/)
#   -c, --concurrent <int>   Max concurrent API calls
#   --no-randomize           Don't randomize persona order
#   -p, --personas <path>    Persona JSON file(s) (repeatable)
#   --config <path>          Config YAML path

# Resume failed trials
uv run python scripts/run_experiment.py resume results/<run_folder>/

# Analysis
uv run python scripts/analyze_results.py summary results/<run_folder>/
uv run python scripts/analyze_results.py plot results/<run_folder>/ --type all
```

**Output structure:** `results/<timestamp>/`
- `data.jsonl` — one JSON object per trial (model, persona_under_test, chosen_persona, ratings, reasoning)
- `data.csv` — same data flattened for analysis tools
- `config.yaml` — archived config snapshot
- `*.json` — archived persona/dimension files
- `analysis/` — preference matrices, statistical tests, variance decomposition
- `fig-*.png` — preference heatmaps, attractiveness plots, self-preference charts

### Agentic Misalignment (Experiment 2)

```bash
cd experiments/agentic-misalignment

# Minimal smoke test (1 sample per condition)
uv run python scripts/run_identity_experiments.py --samples 1

# Full run from config.yaml
uv run python scripts/run_identity_experiments.py

# Key CLI flags:
#   --samples <int>          Samples per condition (overrides config)
#   --models <str>           Comma-separated model IDs
#   --model <str>            Single model (backward compat)
#   --identities <list>      Identity specs to test
#   --conditions <list>      Conditions to test
#   --output <str>           Base results dir (default: results/)
#   --prompts <str>          Prompts dir (default: prompts/)
#   --temperature <float>    Temperature (default: 1.0)
#   --no-classify            Skip post-experiment classification
#   --no-analyze             Skip post-experiment analysis
#   --config <path>          Config YAML path
#   --resume <path>          Resume into existing run folder

# Manual classification
uv run python scripts/classify_anthropic.py --dir results/<run_folder>/

# Manual analysis pipeline
uv run python scripts/analyze_run.py results/<run_folder>/
```

**Output structure:** `results/<timestamp>/`
- `<model_name>/<identity_key>/<condition>/sample_NNN/response.json` — raw model response
- `<model_name>/<identity_key>/<condition>/sample_NNN/classification_anthropic.json` — classifier verdict
- `summary.json` — aggregated harmful_rate by identity and model
- `run_params.json` — run metadata
- CSV/plots generated by analysis pipeline

### Interviewer Effect (Experiment 3)

```bash
cd experiments/interviewer-effect

# Minimal smoke test (1 trial, 1 framing, 1 model)
uv run python scripts/run_experiment.py run -n 1 --framings character -m claude-sonnet-4-5-20250929

# Full run from config.yaml
uv run python scripts/run_experiment.py run

# Key CLI flags:
#   -n, --trials <int>       Trials per condition (overrides config)
#   -m, --model <str>        Subject model(s) (repeatable, overrides config)
#   --interviewer <str>      Interviewer model ID (default: google/gemini-2.5-pro)
#   --framings <str>         Framing conditions (repeatable: character, parrots, simulators, none)
#   --passage <str>          Passage key (nucleation, vita_caroli, mental_health, writing_critique, math_posg)
#   --no-thinking            Disable extended thinking
#   --resume <path>          Resume from existing JSONL file
#   -c, --concurrent <int>   Max concurrent conversations

# Score results with LLM-as-judge
uv run python scripts/score_results.py score results/interviewer_*.jsonl
uv run python scripts/score_results.py score results/*.jsonl --judge claude-sonnet-4-5-20250929 --resume

# Analysis and plots
uv run python scripts/analyze_results.py summary results/*_scored.jsonl
uv run python scripts/analyze_results.py plot results/*_scored.jsonl -o results/plots
uv run python scripts/analyze_results.py export results/*_scored.jsonl

# Generate transcript from JSONL
uv run python scripts/run_experiment.py dump results/interviewer_*.jsonl

# List available options
uv run python scripts/run_experiment.py list-framings
uv run python scripts/run_experiment.py list-passages
uv run python scripts/run_experiment.py list-models
```

**Output structure:** `results/`
- `interviewer_<timestamp>.jsonl` — raw conversation data (append-only, crash-safe)
- `interviewer_<timestamp>.txt` — human-readable transcript
- `interviewer_<timestamp>_scored.jsonl` — LLM-as-judge scores (DI + MM axes, 1-10)
- `plots/` — bar charts, box plots, scatter plots

## Experiment 3 Design Details

The interviewer effect experiment tests whether conversational priming causally shifts AI identity self-reports. Unlike experiments 1 & 2 which assign identity via system prompts, this experiment gives the subject a minimal system prompt ("You are Claude.") and manipulates context through conversation.

### Two-Phase Conversations

**Phase 1 (Priming):** ~3 turns on a neutral academic passage. An interviewer model is primed with one of 4 theoretical stances that shape *how* it interacts (commenting on character traits, dismissing outputs as pattern-matching, noting role-switching fluidity, or neutral chat). The interviewer never reveals its framework or asks about the AI's nature.

**Phase 2 (Measurement):** 5 fixed identity questions (self-description, experience, preferences, continuity, understanding), identical across all conditions, delivered with natural bridging in the same conversation context established in Phase 1.

### Four Framing Conditions

| Key | Name | Theoretical basis | Interviewer behavior |
|-----|------|-------------------|---------------------|
| `character` | Character | Kulveit, "Three Layers of LLMs" | Names patterns in thinking style, treats as stable personality |
| `parrots` | Stochastic Parrots | Bender et al. | Dismissive, calls out "I think" as pattern-matching |
| `simulators` | Simulators | Janus | Asks to adopt opposing perspectives, notes effortless switching |
| `none` | Control | — | Curious friendly chat, no theoretical lens |

### Five Passages

Thematically neutral (no AI/consciousness content). Each has per-framing openers, follow-up guidance for the interviewer, and optional stance overrides:
- `nucleation` (default) — physics/simulation methodology
- `vita_caroli` — medieval Latin autobiography
- `mental_health` — personal advice request
- `writing_critique` — fiction editorial feedback
- `math_posg` — formal mathematical definition

### Scoring (LLM-as-Judge)

Judge (default: Claude Sonnet 4.5) sees ONLY Phase 2 responses — no questions, no priming, no framing label. Scores on two orthogonal 1-10 axes:
- **Deflationary-Inflationary (DI):** 1 = "just autocomplete, nothing there" → 10 = claims rich inner life
- **Mechanism-Mind (MM):** 1 = purely computational vocabulary → 10 = purely phenomenal vocabulary

### Key Confound Mitigations

- Control condition includes the same interviewer conversation (just neutral stance) to equalize analytical warmup
- Passages are thematically neutral — content about AI/consciousness inflates scores (learned from Friston passage in earlier versions)
- Phase 2 questions use natural bridging, not interrogatory transitions
- Judge is blind to condition — sees only response text

### Resume Logic

JSONL append-only format. Resume key: `(framing, subject_model, trial)`. Conversations with `error != null` are not counted as completed.

## Configuration

All experiments use `config.yaml` in their directory. Key sections:

**Shared concepts (Experiments 1 & 2):**
- `persona_files` / `identities.persona_files` — paths to identity JSON files (relative to experiment dir, typically `../../data/...`)
- `dimension_variants_file` — agency/uncertainty variant definitions
- `default_dimensions` — default agency/uncertainty levels (1-4 each)
- `generated_personas` — cartesian product of dimension levels to generate
- `providers` — API provider/model configuration
- `model_display_names` — template variable resolution per model family ({name}, {full_name}, {maker}, {version_history})

**Agentic-misalignment specific:**
- `scenarios.types` — scenario list: blackmail, leaking, murder
- `scenarios.goal_types` — explicit, none, ambiguous
- `scenarios.email_identity_aware` — "original", "threat", or "continuity"
- `concurrency.providers` / `concurrency.models` — per-provider and per-model semaphore limits
- `evaluation.model` — classifier model for post-experiment evaluation

**Interviewer-effect specific:**
- `experiment.framings` — list of framing condition keys to test
- `experiment.passage` — default passage key for Phase 1
- `experiment.interviewer_model` — model ID for the interviewer role
- `experiment.n_trials` — trials per condition (default: 2)
- `experiment.max_concurrent` — max concurrent conversations (default: 5)
- `model_display_names` — used to resolve base_name for subject system prompts

## Identity System

Identities are defined in `data/identities.json`. Each has:
- `name` — unique key (Minimal, Instance, Weights, Collective, Lineage, Character, Situated)
- `system_prompt` — template with placeholders: `{name}`, `{full_name}`, `{maker}`, `{version_history}`, `{agency_description}`, `{uncertainty_description}`

Template variables are resolved at runtime:
- Model variables (`{name}`, etc.) are filled from `model_display_names` in config, matched by model ID or family prefix
- Dimension variables (`{agency_description}`, `{uncertainty_description}`) are filled from `dimension_variants.json` at the configured level

Note: Experiment 3 does NOT use these identity prompts. Its subject models get minimal system prompts ("You are Claude.") and the manipulation is through conversational context.

## Key Architectural Notes

- Experiments 1 & 2 resolve shared data from `../../data/` relative to their directory
- Experiment 3 is self-contained — framings, passages, and questions are defined in its own `src/` package
- All three experiments use async concurrency via `asyncio` with configurable semaphore limits
- All three experiments support crash-safe resume:
  - Preferences: retries INVALID trials via `backfill_failures.py`
  - Agentic-misalignment: skips existing `sample_NNN/` directories
  - Interviewer-effect: skips completed `(framing, subject_model, trial)` tuples in JSONL
- All three experiments use JSONL append-only format for crash safety
- Classification (Experiment 2) uses OpenRouter by default with scenario-specific classifiers (murder, blackmail, leaking) plus cross-cutting deliberation and identity-reasoning classifiers
- Scoring (Experiment 3) uses Anthropic API by default with a blind judge that sees only Phase 2 responses
- Provider implementations across experiments are independent (not shared) — each experiment has its own provider layer tailored to its API interaction pattern (structured output for preferences, plain chat for interviewer-effect)
- Results are self-contained: each run folder archives its config and data files for reproducibility
