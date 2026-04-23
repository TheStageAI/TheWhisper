#!/usr/bin/env python3
"""Classify ASR errors per sample using jiwer alignment + LLM.

For each result JSONL in results/ (filtered by --model and optionally
--tasks), computes per-sample WER, extracts mismatch spans via jiwer,
preclassifies simple patterns (word boundary, word order, clitics,
pure insertion/omission), and sends the rest to an LLM for categorization.

Rewrites each file in place, preserving leaderboard fields (text, pred_text)
and appending per-sample analysis fields (wer, ref_words, subs, dels, ins,
errors, llm_categories, llm_notes).

Usage:
    python analyze_errors.py --model TheStageAI-thewhisper-large-v3-turbo
    python analyze_errors.py --model TheStageAI-thewhisper-large-v3-turbo --tasks fleurs_en_test fleurs_de_test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import jiwer
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ANALYZER_BASE_URL,
    ANALYZER_MODEL,
    ANALYZER_MAX_TOKENS,
    ANALYZER_TEMPERATURE,
    ANALYSIS_THREADS,
    RESULTS_DIR as _RESULTS_DIR,
)
from ml_normalizer import ml_normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(_RESULTS_DIR)

FNAME_RE = re.compile(
    r"^MODEL_(?P<model>.+?)_DATASET_.+?_(?P<task>.+?)\.jsonl$"
)

# ---------------------------------------------------------------------------
# Error categories
# ---------------------------------------------------------------------------
CATEGORIES = [
    "NUMBER_WORD", "WORD_BOUNDARY", "WORD_ORDER", "CLITIC_MARKER",
    "FUNCTION_WORD", "MORPHOLOGICAL", "SEMANTIC_CHANGE", "PHONETIC_SPELLING",
    "NAMED_ENTITY_OR_RARE", "SPURIOUS_INSERTION", "OMISSION", "OTHER",
]

CLITIC_TOKENS = {
    "en": {"s", "t", "d", "ll", "ve", "re", "m"},
    "fr": {"l", "d", "j", "m", "n", "c", "s", "t", "qu", "lorsqu", "jusqu", "puisqu", "quoiqu"},
    "it": {"l", "d", "un", "all", "dell", "sull", "dall", "nell", "quell", "gl"},
    "de": set(), "es": set(), "pt": set(),
}
CLITIC_FULL_FORMS = {
    "en": {"s": {"is", "has", "us"}, "t": {"not"}, "d": {"had", "would"},
            "ll": {"will", "shall"}, "ve": {"have"}, "re": {"are"}, "m": {"am"}},
    "fr": {"l": {"le", "la"}, "d": {"de"}, "j": {"je"}, "m": {"me"}, "n": {"ne"},
            "c": {"ce"}, "s": {"se", "si"}, "t": {"te"}, "qu": {"que", "qui"},
            "lorsqu": {"lorsque"}, "jusqu": {"jusque"}, "puisqu": {"puisque"}, "quoiqu": {"quoique"}},
    "it": {"l": {"lo", "la"}, "d": {"di", "da"}, "un": {"una", "uno"},
            "all": {"alla", "allo", "alle"}, "dell": {"della", "dello", "delle"},
            "sull": {"sulla", "sullo", "sulle"}, "dall": {"dalla", "dallo", "dalle"},
            "nell": {"nella", "nello", "nelle"}, "quell": {"quella", "quello", "quelle"}, "gl": {"gli"}},
    "de": {}, "es": {}, "pt": {},
}

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = '''You are analyzing ASR errors. You will receive:
- REFERENCE: the ground-truth sentence
- HYPOTHESIS: the ASR output for the same audio
- MISMATCHES: a pre-aligned list of places where HYPOTHESIS differs from REFERENCE.
  Each mismatch has an index, a REF span (words from the reference, may be empty)
  and a HYP span (words from the hypothesis, may be empty).

Both texts are pre-normalized: lowercase, no punctuation, digits spelled as words.
Do NOT report casing, punctuation, or digit-form differences.

Classify each mismatch into exactly ONE category. Pick the FIRST matching one from
this priority list and stop:

1. NUMBER_WORD — the spans differ only in how the same numeric value is spelled.
2. CLITIC_MARKER — possessive / contraction / elision clitic that became a standalone
   short token after apostrophe stripping.
3. FUNCTION_WORD — short grammatical word: article, preposition, conjunction,
   auxiliary verb, copula, pronoun. Never for content words.
4. MORPHOLOGICAL — two inflected forms of the SAME dictionary lemma (number, gender,
   case, tense, person). Stem must match. NOT for different words that share letters.
5. SEMANTIC_CHANGE — both sides are valid, real, dictionary words with clearly
   different meanings. The ASR mis-heard a real word as another real word.
6. PHONETIC_SPELLING — hypothesis word is NOT a valid dictionary word (garbled,
   misspelled, truncated), OR it is valid but the reader sees it as "same underlying
   word written with a small error" (typo, diacritic drop, UK/US spelling).
7. NAMED_ENTITY_OR_RARE — substitution involving a proper noun, technical term,
   or rare/loan word, where categories 1-6 do not apply.
8. SPURIOUS_INSERTION — HYP span non-empty, REF span empty (extra word, not reorder).
9. OMISSION — REF span non-empty, HYP span empty (missing word, not reorder).
10. OTHER — any genuine mismatch not fitting the above.

Rules:
- Classify every listed mismatch in order.
- Each mismatch gets exactly one category — the first matching in the priority list.
- Output ONLY valid JSON, no commentary, no code fences.

Output format:
{{"classifications": [
  {{"index": 0, "category": "<CATEGORY>"}},
  {{"index": 1, "category": "<CATEGORY>"}}
]}}

REFERENCE:
{ref}

HYPOTHESIS:
{hyp}

MISMATCHES:
{mismatches_block}
'''

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
_session = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def llm_chat(prompt: str) -> str:
    r = _get_session().post(
        ANALYZER_BASE_URL.rstrip("/") + "/chat/completions",
        json={
            "model": ANALYZER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": ANALYZER_TEMPERATURE,
            "max_tokens": ANALYZER_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------
def extract_chunks(ref: str, hyp: str):
    out = jiwer.process_words(ref, hyp)
    ref_words = ref.split()
    hyp_words = hyp.split()
    chunks = []
    for c in out.alignments[0]:
        chunks.append({
            "type": c.type,
            "ref_words": ref_words[c.ref_start_idx:c.ref_end_idx],
            "hyp_words": hyp_words[c.hyp_start_idx:c.hyp_end_idx],
            "ref_idx": (c.ref_start_idx, c.ref_end_idx),
            "hyp_idx": (c.hyp_start_idx, c.hyp_end_idx),
        })
    return chunks


def merge_adjacent(chunks: list[dict], max_gap: int = 0) -> list[dict]:
    merged = []
    i, n = 0, len(chunks)
    while i < n:
        c = chunks[i]
        if c["type"] == "equal":
            i += 1
            continue
        cur = {
            "ref_words": list(c["ref_words"]),
            "hyp_words": list(c["hyp_words"]),
            "ref_start": c["ref_idx"][0],
            "ref_end": c["ref_idx"][1],
            "hyp_start": c["hyp_idx"][0],
            "hyp_end": c["hyp_idx"][1],
        }
        j = i + 1
        while j < n:
            cj = chunks[j]
            if cj["type"] == "equal":
                if len(cj["ref_words"]) <= max_gap and (j + 1) < n and chunks[j + 1]["type"] != "equal":
                    cur["ref_words"].extend(cj["ref_words"])
                    cur["hyp_words"].extend(cj["ref_words"])  # bridge words are same in ref/hyp
                    cur["ref_end"] = cj["ref_idx"][1]
                    cur["hyp_end"] = cj["hyp_idx"][1]
                    j += 1
                    continue
                else:
                    break
            else:
                cur["ref_words"].extend(cj["ref_words"])
                cur["hyp_words"].extend(cj["hyp_words"])
                cur["ref_end"] = cj["ref_idx"][1]
                cur["hyp_end"] = cj["hyp_idx"][1]
                j += 1
        merged.append(cur)
        i = j
    return merged


def preclassify(m: dict, lang: str) -> str | None:
    rw = [w for w in m["ref_words"] if w]
    hw = [w for w in m["hyp_words"] if w]
    if not rw and not hw:
        return None
    if "".join(rw) == "".join(hw) and rw != hw:
        return "WORD_BOUNDARY"
    if rw and hw and rw != hw and sorted(rw) == sorted(hw) and len(rw) >= 2:
        return "WORD_ORDER"
    clitics = CLITIC_TOKENS.get(lang, set())
    full_forms = CLITIC_FULL_FORMS.get(lang, {})
    if not rw and len(hw) == 1 and hw[0] in clitics:
        return "CLITIC_MARKER"
    if not hw and len(rw) == 1 and rw[0] in clitics:
        return "CLITIC_MARKER"
    if len(rw) == 1 and len(hw) == 1:
        r, h = rw[0], hw[0]
        if r in clitics and h in full_forms.get(r, set()):
            return "CLITIC_MARKER"
        if h in clitics and r in full_forms.get(h, set()):
            return "CLITIC_MARKER"
    if not rw and hw:
        return "SPURIOUS_INSERTION"
    if rw and not hw:
        return "OMISSION"
    return None


def span_sdi(rw: list[str], hw: list[str]) -> dict:
    rw = [w for w in rw if w]
    hw = [w for w in hw if w]
    if not rw and not hw:
        return {"S": 0, "D": 0, "I": 0}
    if not rw:
        return {"S": 0, "D": 0, "I": len(hw)}
    if not hw:
        return {"S": 0, "D": len(rw), "I": 0}
    out = jiwer.process_words(" ".join(rw), " ".join(hw))
    return {"S": out.substitutions, "D": out.deletions, "I": out.insertions}


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------
def make_mismatches_block(items: list[dict]) -> str:
    lines = []
    for i, m in enumerate(items):
        r = " ".join(m["ref_words"])
        h = " ".join(m["hyp_words"])
        lines.append(f'[{i}] REF: "{r}"  HYP: "{h}"')
    return "\n".join(lines)


def parse_response(text: str, n_expected: int) -> dict[int, str]:
    match = re.search(r'\{.*"classifications".*\}', text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError:
        return {}
    out = {}
    for item in obj.get("classifications", []):
        idx = item.get("index")
        cat = item.get("category", "").strip().upper()
        if isinstance(idx, int) and 0 <= idx < n_expected and cat in CATEGORIES:
            out[idx] = cat
    return out


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
def detect_lang(task: str) -> str:
    """Extract language code from task name like fleurs_de_test."""
    parts = task.split("_")
    for p in parts:
        if len(p) == 2 and p.isalpha():
            return p
    return "en"


def normalize_compound_pair(ref: str, hyp: str) -> tuple[str, str]:
    """Mirror open_asr_leaderboard's normalize_compound_pairs on a single pair:
    collapse word-boundary mismatches where the concatenated words match
    (e.g. 'data base' vs 'database'). Required to match leaderboard WER."""
    rw, pw = ref.split(), hyp.split()
    sm = SequenceMatcher(None, rw, pw)
    new_r, new_p = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            new_r.extend(rw[i1:i2])
            new_p.extend(pw[j1:j2])
        else:
            rc = "".join(rw[i1:i2])
            pc = "".join(pw[j1:j2])
            if rc == pc:
                new_r.append(rc)
                new_p.append(pc)
            else:
                new_r.extend(rw[i1:i2])
                new_p.extend(pw[j1:j2])
    return " ".join(new_r), " ".join(new_p)


def process_file(fpath: Path, model: str, task: str, threads: int) -> dict:
    lang = detect_lang(task)
    with fpath.open() as f:
        raw = [json.loads(line) for line in f if line.strip()]

    # Compute per-sample metrics and prepare LLM work.
    # Leaderboard fields (text, pred_text) are preserved verbatim; analysis
    # fields (wer, ref_words, subs, dels, ins, errors, llm_*) plus the
    # normalized versions (text_norm, pred_text_norm) are appended so that
    # the dashboard doesn't need to redo normalization on load.
    work = []
    all_records = []
    for i, row in enumerate(raw):
        ref_n = ml_normalize(row.get("text", ""), lang)
        hyp_n = ml_normalize(row.get("pred_text", ""), lang)
        ref_n, hyp_n = normalize_compound_pair(ref_n, hyp_n)
        if not ref_n.strip():
            all_records.append({**row, "idx": i, "wer": 0.0, "ref_words": 0,
                                "subs": 0, "dels": 0, "ins": 0,
                                "text_norm": ref_n, "pred_text_norm": hyp_n,
                                "llm_categories": [], "llm_notes": "", "errors": []})
            continue
        out = jiwer.process_words(ref_n, hyp_n)
        rw = out.hits + out.substitutions + out.deletions
        s, d, ins_ = out.substitutions, out.deletions, out.insertions
        wer = (s + d + ins_) / max(1, rw)

        rec = {**row,
               "idx": i, "wer": wer, "ref_words": rw,
               "subs": s, "dels": d, "ins": ins_,
               "text_norm": ref_n, "pred_text_norm": hyp_n,
               "llm_categories": [], "llm_notes": "", "errors": []}
        all_records.append(rec)

        if wer <= 0:
            continue

        chunks = extract_chunks(ref_n, hyp_n)
        merged = merge_adjacent(chunks)
        if not merged:
            continue

        preclass = [preclassify(m, lang) for m in merged]
        to_llm = [j for j, c in enumerate(preclass) if c is None]
        work.append({
            "rec_idx": i, "ref": ref_n, "hyp": hyp_n,
            "merged": merged, "preclass": preclass, "to_llm": to_llm,
        })

    # LLM calls in parallel
    llm_results: dict[int, dict[int, str]] = {}

    def do_llm(wi: int):
        w = work[wi]
        if not w["to_llm"]:
            return wi, {}
        sub = [w["merged"][j] for j in w["to_llm"]]
        prompt = PROMPT_TEMPLATE.format(
            lang=lang, ref=w["ref"], hyp=w["hyp"],
            mismatches_block=make_mismatches_block(sub),
        )
        try:
            text = llm_chat(prompt)
            mapping_sub = parse_response(text, len(sub))
        except Exception as e:
            print(f"  LLM error on sample {w['rec_idx']}: {e}", file=sys.stderr)
            mapping_sub = {}
        return wi, {w["to_llm"][k]: v for k, v in mapping_sub.items()}

    if work:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(do_llm, wi) for wi in range(len(work))]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{task}"):
                wi, mapping = fut.result()
                llm_results[wi] = mapping

    # Assemble errors into records
    cat_counts: Counter = Counter()
    for wi, w in enumerate(work):
        rec = all_records[w["rec_idx"]]
        ref_word_count = max(rec["ref_words"], 1)
        errors = []
        cats_for_sample = []
        for j, m in enumerate(w["merged"]):
            cat = w["preclass"][j]
            if cat is None:
                cat = llm_results.get(wi, {}).get(j, "UNCLASSIFIED")
            sdi = span_sdi(m["ref_words"], m["hyp_words"])
            word_contrib = sdi["S"] + sdi["D"] + sdi["I"]
            errors.append({
                "category": cat,
                "ref_words": m["ref_words"],
                "hyp_words": m["hyp_words"],
                "ref_indices": [m["ref_start"], m["ref_end"]],
                "hyp_indices": [m["hyp_start"], m["hyp_end"]],
                "sdi": sdi,
                "word_contribution": word_contrib,
                "wer_contribution": round(word_contrib / ref_word_count, 6),
            })
            cats_for_sample.append(cat)
            cat_counts[cat] += 1
        rec["errors"] = errors
        rec["llm_categories"] = sorted(set(cats_for_sample))
        rec["llm_notes"] = ", ".join(
            f"{e['category']}: '{' '.join(e['ref_words'])}' → '{' '.join(e['hyp_words'])}'"
            for e in errors
        )

    # Save in place (leaderboard format + appended analysis fields)
    with fpath.open("w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_rw = sum(r["ref_words"] for r in all_records)
    total_err = sum(r["subs"] + r["dels"] + r["ins"] for r in all_records)
    print(f"  {task}: {len(raw)} samples, WER {100 * total_err / max(1, total_rw):.2f}%, "
          f"{sum(cat_counts.values())} mismatches classified")
    return {"task": task, "categories": dict(cat_counts)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Classify ASR errors per sample")
    p.add_argument("--model", required=True, help="Model key as in filename (e.g. TheStageAI-thewhisper-large-v3-turbo)")
    p.add_argument("--tasks", nargs="*", default=None, help="Only these tasks (e.g. fleurs_en_test)")
    p.add_argument("--threads", type=int, default=ANALYSIS_THREADS)
    args = p.parse_args()

    files = []
    for fpath in sorted(RESULTS_DIR.glob("MODEL_*.jsonl")):
        m = FNAME_RE.match(fpath.name)
        if not m:
            continue
        if m["model"] != args.model:
            continue
        task = m["task"]
        if args.tasks and task not in args.tasks:
            continue
        files.append((fpath, m["model"], task))

    if not files:
        print(f"No files found for model '{args.model}' in {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(files)} file(s) for model {args.model}")
    summary = {}
    for fpath, model, task in files:
        stats = process_file(fpath, model, task, args.threads)
        summary[task] = stats

    # Save summary
    summary_path = RESULTS_DIR / f"summary_{args.model}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
