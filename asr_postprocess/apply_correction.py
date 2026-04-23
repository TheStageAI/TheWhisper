#!/usr/bin/env python3
"""Apply an optimized prompt bundle to a base model's results, producing
LLM-corrected transcriptions in the same leaderboard-manifest format.

Inputs:
    results/MODEL_<model>_DATASET_..._{task}.jsonl
        — base leaderboard manifests
    experiments/<experiment>/prompts/optimized_program_{lang}_{model_tag}.json
        — MIPROv2 bundles produced by optimize_prompt.py

Outputs:
    experiments/<experiment>/results_corrected/MODEL_<model>_DATASET_..._{task}.jsonl
        — one-for-one with the input manifest; `pred_text` is replaced with the
          LLM-corrected transcription. On leak-filter trip the corrected output
          is discarded and `pred_text` stays equal to the baseline.

Per-sample output schema extends the leaderboard format with:
    pred_text_baseline — the original top-1 ASR hypothesis
    leak               — bool, reasoning-leak filter triggered
    rejected_by_filter — bool, edit-count / length-ratio filter triggered

Usage:
    python apply_correction.py --experiment opt_trb_strat --model openai-whisper-large-v3-turbo
    python apply_correction.py --experiment opt_trb_strat --model openai-whisper-large-v3-turbo \
                               --tasks fleurs_de_test fleurs_fr_test --threads 64
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dspy
import jiwer
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import (
    EXPERIMENTS_DIR,
    RESULTS_DIR,
    TASK_LM_BASE_URL,
    TASK_LM_MAX_TOKENS,
    TASK_LM_MODEL,
    TASK_LM_TEMPERATURE,
)
from analyze_errors import normalize_compound_pair
from ml_normalizer import ml_normalize
from optimize_prompt import ASRCorrection  # re-uses the same signature

FNAME_RE = re.compile(
    r"^MODEL_(?P<model>.+?)_DATASET_.+?_(?P<task>.+?)\.jsonl$"
)

# Strings in LLM output that signal a reasoning leak (thinking-channel spill
# or meta-commentary). Fallback to the baseline hypothesis when any is seen.
LEAK_MARKERS = (
    "Wait,", "Let's apply", "Let me re", "Looking at", "Actually,",
    "$\\rightarrow$", "rightarrow", "### ", "**Step",
)


# ---------------------------------------------------------------------------
# LM setup
# ---------------------------------------------------------------------------
def configure_lm() -> dspy.LM:
    lm = dspy.LM(
        f"openai/{TASK_LM_MODEL}",
        api_base=TASK_LM_BASE_URL,
        api_key="none",
        model_type="chat",
        temperature=TASK_LM_TEMPERATURE,
        max_tokens=TASK_LM_MAX_TOKENS,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    dspy.configure(lm=lm)
    return lm


# ---------------------------------------------------------------------------
# Correction with leak filter
# ---------------------------------------------------------------------------
def _word_edit_count(a: str, b: str) -> int:
    out = jiwer.process_words(a, b)
    return out.substitutions + out.deletions + out.insertions


def safe_correct(
    program: dspy.Predict,
    hypotheses_text: str,
    baseline: str,
    max_edits: int | None = None,
    max_len_ratio: float | None = None,
) -> tuple[str, bool, bool]:
    """Run `program(hypotheses=…)`, guard output. Return (text, leak, rejected)."""
    try:
        pred = program(hypotheses=hypotheses_text)
        out = pred.corrected_transcription
    except Exception:
        return baseline, False, False

    input_words = len(hypotheses_text.split())
    output_words = len(out.split())
    if output_words > input_words * 2 or any(m in out for m in LEAK_MARKERS):
        return baseline, True, False

    if max_len_ratio is not None:
        base_len = max(1, len(baseline.split()))
        if abs(output_words - base_len) / base_len > max_len_ratio:
            return baseline, False, True
    if max_edits is not None:
        if _word_edit_count(baseline, out) > max_edits:
            return baseline, False, True

    return out, False, False


# ---------------------------------------------------------------------------
# Per-task processing
# ---------------------------------------------------------------------------
def detect_lang(task: str) -> str:
    for part in task.split("_"):
        if len(part) == 2 and part.isalpha():
            return part
    return "en"


def process_task(
    src_path: Path,
    out_path: Path,
    program: dspy.Predict,
    lang: str,
    threads: int,
) -> tuple[int, float, float, int, int]:
    with src_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]

    def correct_one(idx_row):
        idx, row = idx_row
        baseline = row.get("pred_text", "")
        corrected, leak, rejected = safe_correct(
            program, hypotheses_text=baseline, baseline=baseline,
        )
        return idx, corrected, leak, rejected

    results: list[tuple[int, str, bool, bool]] = []
    if threads <= 1:
        for idx, row in enumerate(records):
            results.append(correct_one((idx, row)))
    else:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(correct_one, (i, r)) for i, r in enumerate(records)]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=src_path.stem[:24]):
                results.append(fut.result())
    results.sort(key=lambda x: x[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    leaks = rejected_ct = 0
    with out_path.open("w") as f:
        for (idx, corrected, leak, rejected), row in zip(results, records):
            baseline = row.get("pred_text", "")
            new = {k: v for k, v in row.items()
                   if k not in {"wer", "ref_words", "subs", "dels", "ins",
                                "llm_categories", "llm_notes", "errors", "idx"}}
            new["pred_text"] = corrected
            new["pred_text_baseline"] = baseline
            new["leak"] = leak
            new["rejected_by_filter"] = rejected
            f.write(json.dumps(new, ensure_ascii=False) + "\n")
            leaks += int(leak)
            rejected_ct += int(rejected)

    # Corpus WER using the same normalization pipeline as analyze_errors.py /
    # dashboard.py: ml_normalize + normalize_compound_pair. Directly comparable
    # to the authoritative per-sample WER the analysis step records.
    def corpus_wer(refs: list[str], hyps: list[str]) -> float:
        pairs: list[tuple[str, str]] = []
        for r, h in zip(refs, hyps):
            rn = ml_normalize(r, lang)
            hn = ml_normalize(h, lang)
            rn, hn = normalize_compound_pair(rn, hn)
            if rn.strip():
                pairs.append((rn, hn))
        if not pairs:
            return 0.0
        rs, hs = zip(*pairs)
        return jiwer.wer(list(rs), list(hs))

    refs = [r.get("text", "") for r in records]
    base_hyps = [r.get("pred_text", "") for r in records]
    corr_hyps = [c for _, c, _, _ in results]
    wer_b = corpus_wer(refs, base_hyps)
    wer_a = corpus_wer(refs, corr_hyps)
    return len(records), wer_b, wer_a, leaks, rejected_ct


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True,
                   help="Experiment name; prompts live under "
                        "experiments/<experiment>/prompts/, outputs go to "
                        "experiments/<experiment>/results_corrected/")
    p.add_argument("--model", required=True,
                   help="Base model id as in MODEL_<model>_DATASET_... filename")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Only these tasks (e.g. fleurs_de_test)")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--results-dir", default=RESULTS_DIR)
    p.add_argument("--experiments-dir", default=EXPERIMENTS_DIR)
    args = p.parse_args()

    if os.environ.get("OPENAI_API_KEY"):
        print("ERROR: unset OPENAI_API_KEY — this script uses the self-served "
              "LLM at TASK_LM_BASE_URL, not OpenAI.", file=sys.stderr)
        sys.exit(2)

    results_dir = Path(args.results_dir)
    prompts_dir = Path(args.experiments_dir) / args.experiment / "prompts"
    out_dir = Path(args.experiments_dir) / args.experiment / "results_corrected"

    if not prompts_dir.exists():
        print(f"ERROR: prompts dir not found: {prompts_dir}", file=sys.stderr)
        sys.exit(1)

    # Find matching task files for this model
    files = []
    for fpath in sorted(results_dir.glob("MODEL_*.jsonl")):
        m = FNAME_RE.match(fpath.name)
        if not m or m["model"] != args.model:
            continue
        task = m["task"]
        if args.tasks and task not in args.tasks:
            continue
        files.append((fpath, task))

    if not files:
        print(f"ERROR: no manifests in {results_dir} for model {args.model}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Configuring task LM: {TASK_LM_MODEL} @ {TASK_LM_BASE_URL}")
    configure_lm()

    model_tag = TASK_LM_MODEL.split("/")[-1]
    per_lang_programs: dict[str, dspy.Predict] = {}

    def get_program(lang: str) -> dspy.Predict:
        if lang in per_lang_programs:
            return per_lang_programs[lang]
        prog_path = prompts_dir / f"optimized_program_{lang}_{model_tag}.json"
        if not prog_path.exists():
            print(f"ERROR: missing prompt bundle for {lang}: {prog_path}",
                  file=sys.stderr)
            sys.exit(1)
        prog = dspy.Predict(ASRCorrection)
        prog.load(path=str(prog_path))
        per_lang_programs[lang] = prog
        print(f"  Loaded {prog_path.name}")
        return prog

    print(f"Applying {args.experiment} corrections to {len(files)} file(s) "
          f"of model {args.model}")
    print(f"Writing to: {out_dir}\n")
    summary = {}
    for fpath, task in files:
        lang = detect_lang(task)
        prog = get_program(lang)
        out_path = out_dir / fpath.name
        n, wer_b, wer_a, leaks, rej = process_task(
            fpath, out_path, prog, lang, args.threads,
        )
        rel = 100.0 * (wer_b - wer_a) / max(1e-9, wer_b)
        print(f"  {task}: n={n}  WER {100*wer_b:.2f}% → {100*wer_a:.2f}%  "
              f"(Δ {rel:+.2f}% rel)  leaks={leaks}  rejected={rej}")
        summary[task] = {
            "n": n, "base_wer": wer_b, "post_wer": wer_a,
            "rel_improvement_pct": rel, "leaks": leaks, "rejected": rej,
            "out_path": str(out_path),
        }

    summary_path = out_dir / f"summary_{args.model}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
