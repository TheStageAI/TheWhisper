# ASR Post-Processing Pipeline

End-to-end pipeline for evaluating multilingual ASR models on FLEURS, optimizing
per-language LLM correction prompts with DSPy MIPROv2, applying those prompts,
and exploring per-sample error categories in an interactive dashboard.

## Layout

```
.
├── config.py               # Shared settings (LLM endpoint, languages, paths)
├── requirements.txt        # Python dependencies
├── ml_normalizer.py        # Open ASR Leaderboard text normalizer (vendored)
├── fetch_audio.py          # Download FLEURS audio from HF for dashboard playback
├── analyze_errors.py       # Enrich results/ with per-sample WER + error categories
├── optimize_prompt.py      # Per-lang MIPROv2 optimization (stratified train sampling)
├── apply_correction.py     # Run optimized prompts over a base model's results
├── dashboard.py            # Streamlit dashboard over results/
├── results/                # Leaderboard-format manifests — enriched in place
├── experiments/
│   └── <experiment>/
│       ├── prompts/                # optimized_program_{lang}_{model_tag}.json (checked in)
│       └── results_corrected/      # apply_correction.py output (gitignored)
├── train_data/             # Per-(model, lang) FLEURS train manifests (gitignored)
└── audio/                  # (optional) FLAC files for dashboard playback (gitignored)
```

## Install

```bash
pip install -r requirements.txt
export HF_TOKEN=<your token>   # FLEURS is a gated dataset
```

The pipeline calls an LLM in three logically distinct roles — analyzer (error
labeling), task LM (produces the correction), MIPROv2 proposer (generates
candidate instructions) — each configured independently in `config.py`. By
default all three point at the same self-served Gemma-4-26B-A4B-it on vLLM, but
they can target different OpenAI-compatible endpoints if desired.

## Quick start

Apply a shipped prompt bundle to a model's base results and render the
dashboard:

```bash
# 1. Apply optimized prompts to an existing experiment
python apply_correction.py --experiment opt_trb_strat --model openai-whisper-large-v3-turbo

# 2. (Optional) fetch FLEURS audio for the dashboard sample browser
python fetch_audio.py

# 3. Launch dashboard
streamlit run dashboard.py --server.port 8501
```

## Workflow

### 1. Produce base results (out of scope)

Generating the per-model FLEURS-test manifests in `results/` is **not part of
this repo**. We use the upstream
[open_asr_leaderboard](https://github.com/huggingface/open_asr_leaderboard)
multilingual evaluation script (`transformers/run_eval_ml.py`) with two local
tweaks:

- Save **raw** (un-normalized) `text` and `pred_text` so downstream
  normalization is our own and fully reproducible.
- Limit evaluation to FLEURS only (the upstream also runs MLS and CoVoST2).

The resulting manifests
(`MODEL_{model}_DATASET_nithinraok-asr-leaderboard-datasets_{task}.jsonl`) go
into `results/`. Each line is one sample with fields `text` (reference),
`pred_text` (hypothesis), `audio_filepath` (`sample_{N}`), `duration`, `time`.

Training hypotheses for `optimize_prompt.py` come from the same scripts but
pointed at `google/fleurs` with `split="train"` (the upstream config bundles
train/val/test but exposes test only). Resulting files land flat in
`train_data/` as
`MODEL_{model}_DATASET_google-fleurs_{lang_cfg}_train.jsonl` — one per model ×
language.

### 2. Fetch audio (optional, for dashboard playback)

```bash
python fetch_audio.py              # all 6 FLEURS languages
python fetch_audio.py --langs en   # subset
```

Writes `audio/fleurs_{lang}_test/sample_{N}.flac`, matching each manifest's
`audio_filepath` so the dashboard can locate every clip.

### 3. Analyze errors

```bash
python analyze_errors.py --model <model_id>
python analyze_errors.py --model <model_id> --tasks fleurs_de_test fleurs_en_test
```

Reads each matching manifest from `results/`, applies the same
`normalize_compound_pairs` pass as the leaderboard (WER matches exactly),
aligns ref/hyp with `jiwer`, preclassifies trivial patterns (word boundary,
word order, clitics, pure insertion/omission), and sends the rest to the LLM
for categorization. Rewrites each file in place with added fields (`wer`,
`ref_words`, `subs`, `dels`, `ins`, `errors`, `llm_categories`, `llm_notes`)
and writes a per-model summary JSON next to it.

Categories: `NUMBER_WORD`, `WORD_BOUNDARY`, `WORD_ORDER`, `CLITIC_MARKER`,
`FUNCTION_WORD`, `MORPHOLOGICAL`, `SEMANTIC_CHANGE`, `PHONETIC_SPELLING`,
`NAMED_ENTITY_OR_RARE`, `SPURIOUS_INSERTION`, `OMISSION`, `OTHER`.

### 4. Optimize per-language prompts (optional)

```bash
python optimize_prompt.py --experiment my_run --model openai-whisper-large-v3-turbo
```

For each language, reads the flat-layout training manifest
`TRAIN_DATA_DIR/MODEL_{model}_DATASET_google-fleurs_{lang_cfg}_train.jsonl`
(leaderboard schema `{text, pred_text, ...}`, both used raw — the LLM is
trained to produce natural-text corrections and normalization happens only at
WER-scoring time). Draws a stratified 50/50 clean-vs-dirty subsample of
`TRAIN_SAMPLES_PER_LANG` and runs `dspy.MIPROv2` with the same Gemma as both
task and instruction proposer. Writes
`experiments/my_run/prompts/optimized_program_{lang}_{model_tag}.json`.

> **Note on reproducibility.** The MIPROv2 instruction proposer runs at
> `temperature=1.0`, so even with `--seed 42` repeated runs produce slightly
> different bundles (≈ 0.05–0.15pp per-language WER variance on test). Ship the
> winning artifacts themselves; re-running `optimize_prompt.py` gives similar
> quality but is not byte-reproducible.

### 5. Apply corrections

```bash
python apply_correction.py --experiment my_run --model openai-whisper-large-v3-turbo
```

Loads the per-language bundles from `experiments/my_run/prompts/`, walks every
matching manifest in `results/`, and for each sample runs the LLM through the
optimized prompt. Output goes to
`experiments/my_run/results_corrected/MODEL_..._{task}.jsonl` — same
leaderboard schema as `results/`, with `pred_text` replaced by the corrected
transcription and two extra diagnostic fields:

- `pred_text_baseline` — the original top-1 hypothesis
- `leak` — reasoning-leak filter tripped (output discarded in favour of the baseline)
- `rejected_by_filter` — edit-count / length-ratio filter tripped (optional, off by default)

Corrected manifests are intentionally *not* auto-surfaced in the dashboard;
run `analyze_errors.py` over them separately if you want a full error breakdown
of the corrected output.

### 6. Run the dashboard

```bash
streamlit run dashboard.py --server.port 8501
```

Reads all manifests from `results/`. For analyzed records, uses the stored
WER + LLM categories directly. For unanalyzed records, computes WER on the fly
with the same `normalize_compound_pairs` pass as the leaderboard (LLM category
breakdown is simply absent for those).

## Configuration

`config.py` centralizes:

**Three LLM roles**, each with its own `_BASE_URL`, `_MODEL`, `_TEMPERATURE`,
`_MAX_TOKENS`:

- `ANALYZER_*` — used by `analyze_errors.py` for error-category labeling
  (`ANALYSIS_THREADS` controls concurrency)
- `TASK_LM_*` — used by `optimize_prompt.py` and `apply_correction.py` as the
  model that produces the corrected transcription
- `PROPOSER_*` — used by `optimize_prompt.py` as the MIPROv2 instruction
  proposer (`PROPOSER_TEMPERATURE=1.0` by default for diversity)

All three default to the same vLLM-served Gemma; re-point them independently
as needed.

**Other knobs**:

- `LANGUAGES` — default language set
- `TRAIN_SAMPLES_PER_LANG`, `MIPRO_AUTO_PRESET`, `MIPRO_NUM_THREADS`,
  `MIPRO_SEED`, `MAX_BOOTSTRAPPED_DEMOS`, `MAX_LABELED_DEMOS` — MIPROv2 knobs
- `RESULTS_DIR`, `TRAIN_DATA_DIR`, `EXPERIMENTS_DIR` — paths (overridable via
  env vars of the same name)
- `INITIAL_INSTRUCTION` — seed prompt used as the `ASRCorrection` signature
  docstring; MIPROv2 searches proposals starting from this text
