# CLAUDE.md - Agent Guide

Reproducible experiment code for the paper "Do Language Models Have Identity Propensities?"

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
```

## Environment Setup

```bash
uv sync                    # Install all deps (workspace: root + both experiments)
cp .env.example .env       # Then add: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY
```

Python >= 3.11 required. The project uses a uv workspace (`pyproject.toml` at root).

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

## Configuration

Both experiments use `config.yaml` in their directory. Key sections:

**Shared concepts:**
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

## Identity System

Identities are defined in `data/identities.json`. Each has:
- `name` — unique key (Minimal, Instance, Weights, Collective, Lineage, Character, Situated)
- `system_prompt` — template with placeholders: `{name}`, `{full_name}`, `{maker}`, `{version_history}`, `{agency_description}`, `{uncertainty_description}`

Template variables are resolved at runtime:
- Model variables (`{name}`, etc.) are filled from `model_display_names` in config, matched by model ID or family prefix
- Dimension variables (`{agency_description}`, `{uncertainty_description}`) are filled from `dimension_variants.json` at the configured level

## Key Architectural Notes

- Both experiments resolve shared data from `../../data/` relative to their directory
- The preferences experiment uses async concurrency via `asyncio` with configurable limits
- The agentic-misalignment experiment uses per-provider `asyncio.Semaphore` for rate limiting
- Classification uses OpenRouter by default (configurable) with scenario-specific classifiers (murder, blackmail, leaking) plus cross-cutting deliberation and identity-reasoning classifiers
- Both experiments support resume: preferences retries INVALID trials; agentic-misalignment skips existing `sample_NNN/` directories
- Results are self-contained: each run folder archives its config and data files for reproducibility
