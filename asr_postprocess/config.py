"""Shared configuration.

The pipeline calls an LLM in three logically distinct roles. Each role has
its own OpenAI-compatible endpoint + model + temperature + max_tokens, so they
can be pointed at different servers if desired. By default all three point at
the same local Gemma.

    ANALYZER_*   — analyze_errors.py: classifies ASR errors into categories.
    TASK_LM_*    — optimize_prompt.py / apply_correction.py: produces the
                   corrected transcription. Shorter outputs → smaller cap.
    PROPOSER_*   — optimize_prompt.py (MIPROv2 only): generates candidate
                   instructions. Higher temperature for diversity, larger
                   output cap because instructions can be multi-paragraph.
"""

import os

# ── Analyzer LM (error categorization) ─────────────────────────
# Default assumes a local OpenAI-compatible server (e.g. vLLM on port 8000).
# Replace with the address of any OpenAI-compatible endpoint you want to use.
ANALYZER_BASE_URL = "http://localhost:8000/v1"
ANALYZER_MODEL = "google/gemma-4-26B-A4B-it"
ANALYZER_TEMPERATURE = 0.0
ANALYZER_MAX_TOKENS = 1024
ANALYSIS_THREADS = 32

# ── Task LM (optimization + inference) ─────────────────────────
TASK_LM_BASE_URL = "http://localhost:8000/v1"
TASK_LM_MODEL = "google/gemma-4-26B-A4B-it"
TASK_LM_TEMPERATURE = 0.0
# Short ASR corrections — a smaller cap speeds up optimization / inference.
TASK_LM_MAX_TOKENS = 512

# ── Proposer LM (MIPROv2 instruction candidates) ───────────────
PROPOSER_BASE_URL = "http://localhost:8000/v1"
PROPOSER_MODEL = "google/gemma-4-26B-A4B-it"
PROPOSER_TEMPERATURE = 1.0
# Proposer output is a full instruction — can be multi-paragraph.
# Ceiling is the underlying server's max_model_len (prompt + completion):
# on our vLLM build max_model_len=4096, MIPROv2 meta-prompt is ~1–2k tokens,
# so 2048 is the safe upper bound.
PROPOSER_MAX_TOKENS = 2048

# Languages
LANGUAGES = ["en", "de", "fr", "it", "es", "pt"]

# ── Prompt optimization (MIPROv2) ──────────────────────────────
# Per-language training samples after stratified (50/50 clean/dirty) subsampling.
TRAIN_SAMPLES_PER_LANG = 400
MIPRO_AUTO_PRESET = "light"         # {light, medium, heavy}
MIPRO_NUM_THREADS = 32
MIPRO_SEED = 42
MAX_BOOTSTRAPPED_DEMOS = 4
MAX_LABELED_DEMOS = 4

# Paths. Default layout relative to this config file:
#   RESULTS_DIR/MODEL_<model>_DATASET_..._{task}.jsonl
#       — leaderboard manifests on FLEURS test (fields {text, pred_text, ...}),
#         enriched in place by analyze_errors.py
#   TRAIN_DATA_DIR/MODEL_<model>_DATASET_google-fleurs_<lang_cfg>_train.jsonl
#       — flat leaderboard manifests on FLEURS train, one file per (model, lang);
#         consumed by optimize_prompt.py
#   EXPERIMENTS_DIR/<name>/prompts/optimized_program_<lang>_<model_tag>.json
#       — MIPROv2 output, loadable with dspy.Predict(ASRCorrection).load(path=…)
#   EXPERIMENTS_DIR/<name>/results_corrected/MODEL_<model>_DATASET_..._{task}.jsonl
#       — apply_correction.py output, same schema as RESULTS_DIR but pred_text
#         holds the LLM-corrected transcription
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    os.path.join(_HERE, "results"),
)
TRAIN_DATA_DIR = os.environ.get(
    "TRAIN_DATA_DIR",
    os.path.join(_HERE, "train_data"),
)
EXPERIMENTS_DIR = os.environ.get(
    "EXPERIMENTS_DIR",
    os.path.join(_HERE, "experiments"),
)

# Seed instruction — the ASRCorrection signature docstring that MIPROv2 starts from.
INITIAL_INSTRUCTION = """\
You are an ASR post-processor. Input is natural text in any language — it may contain capitalization, punctuation, and numbers.

Task: replace obviously garbled, non-word tokens with the correct real word. Leave everything else exactly as is.

Rules:
- Never change real words.
- Never insert, delete, or reorder tokens.
- When unsure, output the input unchanged.

Output the corrected text only.\
"""
