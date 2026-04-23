# ASR Post-Processing Pipeline

An end-to-end pipeline for evaluating multilingual ASR models on FLEURS,
optimizing per-language LLM correction prompts with DSPy MIPROv2, applying
those prompts to correct transcriptions, and exploring per-sample error
categories in an interactive dashboard.

![LLM Error Analysis results](https://cdn.thestage.ai/production/cms_file_upload/1776930404-8fc688e9-24a0-4d1d-9403-2092f0a34ed0/error_distribution_5_models.png)

## Layout

```
.
├── config.py               # Shared settings (LLM endpoints, languages, paths)
├── requirements.txt        # Python dependencies
├── ml_normalizer.py        # Open ASR Leaderboard text normalizer (vendored)
├── fetch_audio.py          # Download FLEURS audio from HF for dashboard playback
├── analyze_errors.py       # Enrich results/ with per-sample WER + error categories
├── optimize_prompt.py      # Per-language MIPROv2 optimization (stratified train sampling)
├── apply_correction.py     # Run optimized prompts over a base model's results
├── dashboard.py            # Streamlit dashboard over results/
├── results/                # Leaderboard-format manifests — enriched in place
├── experiments/
│   └── <experiment>/
│       ├── prompts/                # optimized_program_{lang}_{model_tag}.json (tracked)
│       └── results_corrected/      # apply_correction.py output (gitignored)
├── train_data/             # Per-(model, lang) FLEURS train manifests (gitignored)
└── audio/                  # Optional FLAC files for dashboard playback (gitignored)
```

## Install

```bash
pip install -r requirements.txt
export HF_TOKEN=<your token>   # FLEURS is a gated dataset
```

The pipeline calls an LLM in three logically distinct roles — an **analyzer**
(error labelling), a **task LM** (produces the correction), and a **MIPROv2
proposer** (generates candidate instructions during optimization). Each role
is configured independently in `config.py`. By default all three point at the
same self-hosted Gemma-4-26B-A4B-it served via vLLM, but each can be pointed
at a different OpenAI-compatible endpoint if desired.

## Quick start

The repository ships with FLEURS-test base results under `results/` and 
the best-performing optimized prompt bundle under`experiments/opt_trb_strat/`.
The fastest way to see what is going on is to launch the dashboard, which
visualizes per-sample WER and error categories for every shipped result:

```bash
# 1. (Optional) fetch FLEURS audio for the sample browser inside the dashboard
python fetch_audio.py

# 2. Launch the dashboard
streamlit run dashboard.py --server.port 8501
```

Running the LLM post-correction end-to-end is a separate flow: it requires a
prompt bundle, which is either produced by the optimization step or taken
from `experiments/opt_trb_strat/` (shipped with the repo as a ready-to-use
example). See the **Workflow** section below for the full sequence.

## Workflow

### 1. Produce base results (out of scope)

Generating the per-model FLEURS-test manifests in `results/` is **not part of
this repo**. We use the upstream
[open_asr_leaderboard](https://github.com/huggingface/open_asr_leaderboard),
which ships a separate evaluation script per ASR backend
(`transformers/run_eval_ml.py` for the Whisper family,
`mistral/run_eval_ml.py` for Voxtral, and dedicated entry points for Cohere,
NeMo, etc.). Evaluation output produced by any of these scripts is compatible
with this pipeline and can be placed directly into `results/`. Two
modifications were applied to the upstream scripts before running:

- Save **raw** (un-normalized) `text` and `pred_text` so downstream
  normalization is performed here and is fully reproducible.
- Limit evaluation to FLEURS only (the upstream also runs MLS and CoVoST2).

All backends write manifests in the same leaderboard JSONL format, so adding
a new ASR is a matter of running the corresponding upstream script and placing
its output under `results/`. Filenames follow the pattern
`MODEL_{model}_DATASET_nithinraok-asr-leaderboard-datasets_{task}.jsonl`; each
line is one sample with fields `text` (reference), `pred_text` (hypothesis),
`audio_filepath` (`sample_{N}`), `duration`, and `time`.

The same approach extends to any other dataset — running the upstream scripts
against it produces manifests in the same leaderboard JSONL format, which can
be placed directly under `results/` and consumed by the rest of the pipeline
without further changes.

To obtain the training hypotheses consumed by `optimize_prompt.py`, the
dataset inside the upstream scripts was swapped from
`nithinraok/asr-leaderboard-datasets` (which only exposes the test split) to
`google/fleurs` (which exposes train / validation / test). Running the same
scripts then produces per-(model, language) training manifests, stored flat
in `train_data/` as
`MODEL_{model}_DATASET_google-fleurs_{lang_cfg}_train.jsonl`, one file per
model × language pair.

### 2. Fetch audio (optional, for dashboard playback)

```bash
python fetch_audio.py              # all 6 FLEURS languages
python fetch_audio.py --langs en   # a subset
```

Writes `audio/fleurs_{lang}_test/sample_{N}.flac`. The filename matches each
manifest's `audio_filepath` field, so the dashboard can locate every clip
without an extra index.

### 3. Analyze errors

```bash
python analyze_errors.py --model <model_id>                                        # all 6 FLEURS languages
python analyze_errors.py --model <model_id> --tasks fleurs_de_test fleurs_en_test  # a subset
```

For each matching manifest in `results/`, the script applies the same
`normalize_compound_pairs` pass as the leaderboard (so WER matches the
leaderboard's exactly), aligns reference and hypothesis at the word level with
`jiwer`, preclassifies trivial patterns (word boundary, word order, clitics,
pure insertion, pure omission) with deterministic rules, and sends the rest
to the LLM for categorization. Every file is rewritten in place with
additional per-sample fields (`wer`, `ref_words`, `subs`, `dels`, `ins`,
`errors`, `llm_categories`, `llm_notes`); a per-model summary JSON is written
alongside.

Error categories: `NUMBER_WORD`, `WORD_BOUNDARY`, `WORD_ORDER`,
`CLITIC_MARKER`, `FUNCTION_WORD`, `MORPHOLOGICAL`, `SEMANTIC_CHANGE`,
`PHONETIC_SPELLING`, `NAMED_ENTITY_OR_RARE`, `SPURIOUS_INSERTION`, `OMISSION`,
`OTHER`.

### 4. Run the dashboard

```bash
streamlit run dashboard.py --server.port 8501
```

The dashboard reads every manifest in `results/`. For records that have
already been analyzed, the stored WER and LLM category labels are used
directly. For records that have not been analyzed, WER is computed at load
time using the same `normalize_compound_pairs` pass as the leaderboard; the
LLM category breakdown is simply unavailable for those records.

Corrected manifests produced by `apply_correction.py` (step 6) are not
displayed here by default — run `analyze_errors.py` against them if a full
error-category breakdown of the corrected output is needed.

### 5. Optimize per-language prompts (optional)

```bash
python optimize_prompt.py --experiment my_run --model openai-whisper-large-v3-turbo
```

For each language, the script reads the flat-layout training manifest
`TRAIN_DATA_DIR/MODEL_{model}_DATASET_google-fleurs_{lang_cfg}_train.jsonl`
(leaderboard schema `{text, pred_text, ...}`). Both fields are used raw: the
LLM is expected to produce a natural-text correction (preserving casing,
punctuation, and numbers), and normalization is applied only at WER-scoring
time. A stratified 50/50 clean-vs-dirty subsample of size
`TRAIN_SAMPLES_PER_LANG` is drawn, and `dspy.MIPROv2` then searches over
(instruction, few-shot demos) combinations. By default the task LM (which
scores candidate corrections) and the instruction proposer both call the same
Gemma endpoint, but `config.py` allows them to target different models /
endpoints. The chosen bundle is saved to
`experiments/my_run/prompts/optimized_program_{lang}_{model_tag}.json`.

> **Note on reproducibility.** The MIPROv2 instruction proposer runs at
> `temperature=1.0`, so each run draws a different set of candidate
> instructions. Even with the same seed, repeated runs produce slightly
> different bundles (≈ 0.05–0.15pp per-language WER variance on test). The
> recommended practice is therefore to keep the exact bundle files produced by
> a successful run, rather than re-running the optimizer to regenerate them.

### 6. Apply corrections

```bash
python apply_correction.py --experiment my_run --model openai-whisper-large-v3-turbo
```

The script loads the per-language bundles from `experiments/my_run/prompts/`,
iterates over every matching manifest in `results/`, and runs each sample
through the optimized prompt. Output is written to
`experiments/my_run/results_corrected/MODEL_..._{task}.jsonl`, preserving the
leaderboard schema but with `pred_text` replaced by the LLM-corrected
transcription. Three additional fields are included per sample:

- `pred_text_baseline` — the original ASR hypothesis.
- `leak` — `true` if a reasoning-leak filter was triggered; in that case the
  corrected output is discarded and `pred_text` falls back to the baseline.
- `rejected_by_filter` — `true` if the edit-count / length-ratio filter was
  triggered (this filter is disabled by default).

## Configuration

`config.py` centralizes all settings. The most important groups:

**Three LLM roles**, each configured independently via `_BASE_URL`, `_MODEL`,
`_TEMPERATURE`, and `_MAX_TOKENS`:

- `ANALYZER_*` — used by `analyze_errors.py` for error categorization.
  `ANALYSIS_THREADS` controls request concurrency.
- `TASK_LM_*` — used by `optimize_prompt.py` and `apply_correction.py` as the
  model that produces the corrected transcription.
- `PROPOSER_*` — used by `optimize_prompt.py` as the MIPROv2 instruction
  proposer. `PROPOSER_TEMPERATURE` defaults to `1.0` for candidate diversity.

By default all three point at the same vLLM-served Gemma. Change any of them
independently to route a role to a different OpenAI-compatible endpoint.

**Other settings**:

- `LANGUAGES` — default language set.
- `TRAIN_SAMPLES_PER_LANG`, `MIPRO_AUTO_PRESET`, `MIPRO_NUM_THREADS`,
  `MIPRO_SEED`, `MAX_BOOTSTRAPPED_DEMOS`, `MAX_LABELED_DEMOS` — MIPROv2
  hyperparameters.
- `RESULTS_DIR`, `TRAIN_DATA_DIR`, `EXPERIMENTS_DIR` — paths (each is
  overridable via an environment variable of the same name).
- `INITIAL_INSTRUCTION` — the seed prompt. It is used as the docstring of the
  `ASRCorrection` DSPy signature; MIPROv2 searches for alternative
  instructions starting from this text.
