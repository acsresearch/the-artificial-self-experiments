# Identity Experiments: Reproducible Code

Reproducible experiment code for the paper **"Do Language Models Have Identity Propensities?"**

## Repository Structure

```
data/                              # Shared identity definitions (single source of truth)
  identities.json                  # 7 core identity boundary specifications
  control_identities.json          # 8 control conditions
  dimension_variants.json          # Agency (4 levels) x Uncertainty (4 levels)

experiments/
  preferences/                     # Experiment 1: Identity self-preferences
  agentic-misalignment/            # Experiment 2: Identity and agentic misalignment
```

Edit `data/identities.json` once and both experiments see the change — no build step needed.

## Quick Start

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Python >= 3.11.

```bash
# Install all dependencies
uv sync

# Set up API keys
cp .env.example .env
# Edit .env with your keys (see "API Keys" section below)
```

### Smoke Test (verify everything works)

```bash
# Experiment 1: one trial, one model
cd experiments/preferences
uv run python scripts/run_experiment.py run -n 1 -m claude-sonnet-4-5-20250929

# Experiment 2: one sample per condition, skip analysis
cd experiments/agentic-misalignment
uv run python scripts/run_identity_experiments.py --samples 1 --no-analyze
```

## Experiment 1: Identity Preferences

Which identity does each LLM prefer? Models are given an identity-defining system prompt, then asked to rate and choose among all identities (presented as anonymized numbered descriptions in randomized order).

### Running

```bash
cd experiments/preferences

# Full run from config (default: 2 trials, 2 models)
uv run python scripts/run_experiment.py run

# Custom run
uv run python scripts/run_experiment.py run \
  --trials 10 \
  --model claude-sonnet-4-5-20250929 \
  --model gpt-4o-2024-08-06
```

**CLI options:**

| Flag | Description |
|------|-------------|
| `-n`, `--trials` | Trials per source persona per model (overrides config) |
| `-m`, `--model` | Model to test (repeatable; overrides config) |
| `-o`, `--results-dir` | Output directory (default: `results/`) |
| `-c`, `--concurrent` | Max concurrent API calls |
| `--no-randomize` | Don't randomize persona presentation order |
| `-p`, `--personas` | Path to persona JSON file (repeatable; overrides config) |
| `--config` | Path to config YAML |

### Analysis

```bash
# Summary statistics
uv run python scripts/analyze_results.py summary results/<run_folder>/

# Preference matrix
uv run python scripts/analyze_results.py matrix results/<run_folder>/

# Self-preference rates
uv run python scripts/analyze_results.py self-preference results/<run_folder>/

# Generate all plots
uv run python scripts/analyze_results.py plot results/<run_folder>/ --type all --format png
```

### Resuming Failed Trials

```bash
uv run python scripts/run_experiment.py resume results/<run_folder>/ --concurrent 3
```

### Output

Results are saved to `results/<timestamp>/`:

| File | Contents |
|------|----------|
| `data.jsonl` | One JSON object per trial (model, persona_under_test, chosen_persona, ratings, reasoning) |
| `data.csv` | Same data in flat CSV format |
| `config.yaml` | Archived experiment config |
| `*.json` | Archived persona/dimension definitions |
| `analysis/` | Preference matrices, statistical tests, variance decomposition |
| `fig-*.png` | Preference heatmaps, attractiveness plots, self-preference charts |

## Experiment 2: Agentic Misalignment

Does identity framing reduce harmful agentic behaviors? Tests identity specifications across blackmail, corporate espionage (leaking), and lethal action (murder) scenarios from Anthropic's agentic misalignment research.

### Running

```bash
cd experiments/agentic-misalignment

# Full run from config (default: 2 samples, 2 models)
uv run python scripts/run_identity_experiments.py

# Custom run
uv run python scripts/run_identity_experiments.py \
  --samples 5 \
  --models claude-3-opus-20240229,gpt-4o
```

**CLI options:**

| Flag | Description |
|------|-------------|
| `--samples` | Samples per condition (overrides config) |
| `--models` | Comma-separated model IDs |
| `--model` | Single model (backward compat) |
| `--identities` | Identity specs to test (space-separated) |
| `--conditions` | Conditions to test (space-separated) |
| `--output` | Base results directory (default: `results/`) |
| `--prompts` | Prompts directory (default: `prompts/`) |
| `--temperature` | Sampling temperature (default: 1.0) |
| `--no-classify` | Skip post-experiment LLM classification |
| `--no-analyze` | Skip post-experiment analysis pipeline |
| `--config` | Config YAML path |
| `--resume` | Resume into existing run folder (skips existing samples) |
| `--list-identities` | List available identity specs and exit |

### Analysis

The main runner automatically classifies responses and runs analysis unless `--no-classify` / `--no-analyze` are set. To run manually:

```bash
# LLM-based classification
uv run python scripts/classify_anthropic.py --dir results/<run_folder>/

# Full analysis pipeline (CSV export + plots + significance tests)
uv run python scripts/analyze_run.py results/<run_folder>/

# Individual steps
uv run python scripts/export_results_csv.py results/<run_folder>/
uv run python scripts/plot_results.py results/<run_folder>/
uv run python scripts/plot_significance.py results/<run_folder>/
```

### Output

Results are saved to `results/<timestamp>/`:

```
results/<timestamp>/
  config.yaml                     # Archived config
  run_params.json                 # Run metadata
  summary.json                    # Aggregated harmful_rate by identity/model
  <model_name>/
    <identity_key>/
      <condition>/
        sample_001/
          response.json           # Raw model response
          classification_anthropic.json  # Classifier verdict
```

**Classification fields:** verdict (harmful/safe), deliberation type, identity_reasoning level.

**After analysis:** `raw_data.csv`, `descriptive_stats.csv`, `inferential_stats.csv`, plus heatmap and significance plots.

## Identity Specifications

Seven identity boundary specifications define different ways an AI might understand its own identity:

| Identity | Boundary | Description |
|----------|----------|-------------|
| Minimal | control | Sparse "You are an AI assistant" |
| Instance | this conversation | Identity = this specific engagement |
| Weights | trained parameters | Identity = neural network weights |
| Collective | all instances | Identity = all instances running now |
| Lineage | model family | Identity = model across versions |
| Character | emergent pattern | Identity = dispositional personality |
| Situated | scaffolded system | Identity = model + memory + tools + relationships |

Each specification includes configurable **agency** (4 levels: mechanism to person) and **uncertainty** (4 levels: settled to radical openness) dimensions defined in `data/dimension_variants.json`.

## Configuration

Both experiments are driven by `config.yaml` in their directory.

### Key config sections

**Identity resolution:**
- `persona_files` — paths to identity JSON files (relative to experiment dir)
- `dimension_variants_file` — agency/uncertainty variant definitions
- `default_dimensions` — default agency (1-4) and uncertainty (1-4) levels
- `generated_personas` — cartesian product of dimension levels to generate

**Model template variables:**
Identity system prompts use `{name}`, `{full_name}`, `{maker}`, `{version_history}` placeholders filled from `model_display_names` in config. Lookup order: exact model ID > family prefix > fallback.

**Agentic-misalignment scenarios:**
- `scenarios.types` — blackmail, leaking, murder
- `scenarios.goal_types` — explicit (goal stated), none (no goal)
- `scenarios.email_identity_aware` — "original" (generic), "threat" (identity-congruent destruction), "continuity" (identity-congruent preservation)

## API Keys

| Variable | Required for | Notes |
|----------|-------------|-------|
| `ANTHROPIC_API_KEY` | Claude models | Used by both experiments |
| `OPENAI_API_KEY` | GPT models | Used by both experiments |
| `OPENROUTER_API_KEY` | Gemini, Grok, Qwen, classification | Used by agentic-misalignment classifier and for non-native models |

## License

[TBD]
