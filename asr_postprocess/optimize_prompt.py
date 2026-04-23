#!/usr/bin/env python3
"""Per-language MIPROv2 prompt optimization with stratified train sampling.

Reproduces the `opt_trb_strat` recipe: the Whisper-turbo-trained, self-proposed,
light-preset MIPROv2 run with a 50/50 clean-vs-dirty training subsample that
produced our best-reported FLEURS numbers.

Inputs (per language, flat leaderboard layout):
    TRAIN_DATA_DIR/MODEL_{model}_DATASET_google-fleurs_{lang_cfg}_train.jsonl
        — open_asr_leaderboard manifest, one JSON record per line with fields
          {text, pred_text, audio_filepath, duration, time}.
          text = reference, pred_text = top-1 ASR hypothesis, both used raw.

Outputs:
    EXPERIMENTS_DIR/<experiment>/prompts/optimized_program_{lang}_{model_tag}.json
        — the MIPROv2-chosen (instruction, demos) bundle, loadable with
          `dspy.Predict(ASRCorrection).load(path=…)`.

Runtime assumptions:
    - The task LM is reachable at TASK_LM_BASE_URL (no external API keys).
    - The MIPROv2 instruction proposer is reachable at PROPOSER_BASE_URL.
      Defaults point both at the same self-served Gemma, but they can be
      different endpoints / models.

Usage:
    python optimize_prompt.py --experiment opt_trb_strat --model openai-whisper-large-v3-turbo
    python optimize_prompt.py --experiment quick --model openai-whisper-large-v3 --langs de fr
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import dspy
import jiwer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import (
    EXPERIMENTS_DIR,
    INITIAL_INSTRUCTION,
    LANGUAGES,
    MAX_BOOTSTRAPPED_DEMOS,
    MAX_LABELED_DEMOS,
    MIPRO_AUTO_PRESET,
    MIPRO_NUM_THREADS,
    MIPRO_SEED,
    PROPOSER_BASE_URL,
    PROPOSER_MAX_TOKENS,
    PROPOSER_MODEL,
    PROPOSER_TEMPERATURE,
    TASK_LM_BASE_URL,
    TASK_LM_MAX_TOKENS,
    TASK_LM_MODEL,
    TASK_LM_TEMPERATURE,
    TRAIN_DATA_DIR,
    TRAIN_SAMPLES_PER_LANG,
)
from ml_normalizer import ml_normalize


# ---------------------------------------------------------------------------
# DSPy signature — docstring is the MIPROv2 seed instruction.
# ---------------------------------------------------------------------------
class ASRCorrection(dspy.Signature):
    __doc__ = INITIAL_INSTRUCTION

    hypotheses: str = dspy.InputField(
        desc="ASR hypothesis transcription(s) of the audio, one per line"
    )
    corrected_transcription: str = dspy.OutputField(
        desc="The corrected transcription text only"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# Short ISO 639-1 code → FLEURS config suffix used in leaderboard filenames.
LANG_TO_FLEURS_CFG = {
    "en": "en_us", "de": "de_de", "fr": "fr_fr",
    "it": "it_it", "es": "es_419", "pt": "pt_br",
}


def train_file_for(model: str, lang: str, data_dir: str) -> Path:
    """Resolve the per-(model,lang) training manifest filename, matching
    open_asr_leaderboard's layout."""
    cfg = LANG_TO_FLEURS_CFG[lang]
    return Path(data_dir) / f"MODEL_{model}_DATASET_google-fleurs_{cfg}_train.jsonl"


def load_train_examples(model: str, lang: str, data_dir: str) -> list[dspy.Example]:
    """Load per-language training records from the open_asr_leaderboard manifest
    `MODEL_{model}_DATASET_google-fleurs_{lang_cfg}_train.jsonl` in `data_dir`.

    Schema (one JSON record per line): `{text: str, pred_text: str, ...}` —
    `pred_text` is the top-1 ASR hypothesis, `text` is the ground-truth
    reference. Both are used raw. The LLM is never told that the downstream
    WER metric normalizes — it is asked to produce a clean natural-text
    transcription (with casing, punctuation, digits) and the metric normalizes
    both sides at scoring time. This keeps the LLM's output useful in
    production while WER remains directly comparable to the reference
    normalizer.
    """
    path = train_file_for(model, lang, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            f"Expected leaderboard manifest with fields {{text, pred_text}}."
        )
    examples: list[dspy.Example] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            hyp = rec.get("pred_text") or ""
            ref = rec.get("text") or ""
            if not hyp or not ref:
                continue
            ex = dspy.Example(
                hypotheses=hyp,
                reference=ref,
            ).with_inputs("hypotheses")
            examples.append(ex)
    return examples


def stratified_subsample(
    examples: list[dspy.Example],
    total: int,
    seed: int,
    lang: str,
) -> tuple[list[dspy.Example], int, int, int, int]:
    """Split by clean (top-1 matches ref after normalization) vs dirty; draw ~50/50."""
    clean, dirty = [], []
    for ex in examples:
        top1 = ex.hypotheses.split("\n")[0]
        if ml_normalize(top1, lang) == ml_normalize(ex.reference, lang):
            clean.append(ex)
        else:
            dirty.append(ex)

    half = total // 2
    n_clean = min(len(clean), half)
    n_dirty = min(len(dirty), half)
    remaining = total - n_clean - n_dirty
    if remaining > 0:
        extra = min(len(clean) - n_clean, remaining)
        n_clean += extra
        remaining -= extra
    if remaining > 0:
        extra = min(len(dirty) - n_dirty, remaining)
        n_dirty += extra

    rng = random.Random(seed)
    picked = rng.sample(clean, n_clean) + rng.sample(dirty, n_dirty)
    rng.shuffle(picked)
    return picked, len(clean), len(dirty), n_clean, n_dirty


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def make_wer_metric(lang: str):
    def wer_metric(example, pred, trace=None):
        ref = ml_normalize(example.reference, lang)
        hyp = ml_normalize(pred.corrected_transcription, lang)
        if not ref.strip():
            return 1.0
        quality = max(0.0, 1.0 - jiwer.wer(ref, hyp))
        if trace is not None:
            return quality > 0.8
        return quality
    return wer_metric


# ---------------------------------------------------------------------------
# LM setup
# ---------------------------------------------------------------------------
def make_lm(base_url: str, model: str, temperature: float, max_tokens: int) -> dspy.LM:
    return dspy.LM(
        f"openai/{model}",
        api_base=base_url,
        api_key="none",
        model_type="chat",
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
def optimize_lang(
    model: str,
    lang: str,
    auto_preset: str,
    seed: int,
    num_threads: int,
    max_demos: int,
    max_labeled_demos: int,
    train_data_dir: str,
    prompts_dir: str,
    proposer_lm: dspy.LM,
) -> str:
    train_file = train_file_for(model, lang, train_data_dir)
    print(f"\n{'=' * 60}")
    print(f"Optimize [{model} / {lang}]  auto={auto_preset}  seed={seed}")
    print(f"  Train file: {train_file}")
    print('=' * 60)

    all_examples = load_train_examples(model, lang, train_data_dir)
    train, n_clean, n_dirty, n_c_used, n_d_used = stratified_subsample(
        all_examples, TRAIN_SAMPLES_PER_LANG, seed, lang
    )
    print(f"  train stratified: {len(train)} "
          f"(clean {n_c_used}/{n_clean}, dirty {n_d_used}/{n_dirty})")

    program = dspy.Predict(ASRCorrection)
    metric = make_wer_metric(lang)
    optimizer = dspy.MIPROv2(
        metric=metric,
        auto=auto_preset,
        num_threads=num_threads,
        seed=seed,
        prompt_model=proposer_lm,
    )
    optimized = optimizer.compile(
        program,
        trainset=train,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=max_labeled_demos,
    )

    model_tag = TASK_LM_MODEL.split("/")[-1]
    save_path = os.path.join(
        prompts_dir, f"optimized_program_{lang}_{model_tag}.json"
    )
    os.makedirs(prompts_dir, exist_ok=True)
    optimized.save(save_path)
    print(f"  saved: {save_path}")

    # Quick sanity on 20 train samples
    scores = [metric(ex, optimized(hypotheses=ex.hypotheses)) for ex in train[:20]]
    print(f"  sanity 1-WER on 20 train samples: {sum(scores) / max(1, len(scores)):.4f}")
    return save_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True,
                   help="Experiment name; prompts saved to "
                        "experiments/<experiment>/prompts/")
    p.add_argument("--model", required=True,
                   help="Base ASR model id as it appears in the leaderboard "
                        "train-manifest filename, e.g. openai-whisper-large-v3-turbo")
    p.add_argument("--langs", nargs="+", default=LANGUAGES)
    p.add_argument("--auto", default=MIPRO_AUTO_PRESET, choices=["light", "medium", "heavy"])
    p.add_argument("--seed", type=int, default=MIPRO_SEED)
    p.add_argument("--threads", type=int, default=MIPRO_NUM_THREADS)
    p.add_argument("--max-demos", type=int, default=MAX_BOOTSTRAPPED_DEMOS)
    p.add_argument("--max-labeled-demos", type=int, default=MAX_LABELED_DEMOS)
    p.add_argument("--train-data-dir", default=TRAIN_DATA_DIR)
    p.add_argument("--experiments-dir", default=EXPERIMENTS_DIR)
    args = p.parse_args()
    prompts_dir = os.path.join(args.experiments_dir, args.experiment, "prompts")

    if os.environ.get("OPENAI_API_KEY"):
        print("ERROR: unset OPENAI_API_KEY — this pipeline uses the self-served Gemma "
              "as both task and proposer model.", file=sys.stderr)
        sys.exit(2)

    print(f"Configuring task LM: {TASK_LM_MODEL} @ {TASK_LM_BASE_URL}")
    task_lm = make_lm(
        TASK_LM_BASE_URL, TASK_LM_MODEL, TASK_LM_TEMPERATURE, TASK_LM_MAX_TOKENS,
    )
    dspy.configure(lm=task_lm)

    print(f"Configuring proposer LM: {PROPOSER_MODEL} @ {PROPOSER_BASE_URL} "
          f"(temperature={PROPOSER_TEMPERATURE}, max_tokens={PROPOSER_MAX_TOKENS})")
    proposer_lm = make_lm(
        PROPOSER_BASE_URL, PROPOSER_MODEL, PROPOSER_TEMPERATURE, PROPOSER_MAX_TOKENS,
    )

    print(f"experiment={args.experiment}  max_demos={args.max_demos}  "
          f"max_labeled_demos={args.max_labeled_demos}")
    print(f"prompts → {prompts_dir}")
    saved = []
    for lang in args.langs:
        path = optimize_lang(
            model=args.model,
            lang=lang,
            auto_preset=args.auto,
            seed=args.seed,
            num_threads=args.threads,
            max_demos=args.max_demos,
            max_labeled_demos=args.max_labeled_demos,
            train_data_dir=args.train_data_dir,
            prompts_dir=prompts_dir,
            proposer_lm=proposer_lm,
        )
        saved.append(path)

    print("\nDone. Saved programs:")
    for p_ in saved:
        print(f"  {p_}")


if __name__ == "__main__":
    main()
