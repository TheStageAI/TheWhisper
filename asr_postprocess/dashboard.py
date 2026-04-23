"""ASR Evaluation Dashboard — Streamlit app.

Reads JSONL result manifests from ``results/`` (leaderboard format),
computes per-sample WER, and renders interactive charts + example browser.

Run:
    streamlit run dashboard.py
"""
from __future__ import annotations

import html as html_mod
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from jiwer import process_words

from analyze_errors import detect_lang, normalize_compound_pair
from ml_normalizer import ml_normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
AUDIO_DIR = HERE / "audio"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FNAME_RE = re.compile(
    r"^MODEL_(?P<model>.+?)_DATASET_.+?_(?P<task>.+?)\.jsonl$"
)

LLM_CATEGORY_ORDER = [
    "NUMBER_WORD", "WORD_BOUNDARY", "WORD_ORDER", "CLITIC_MARKER",
    "FUNCTION_WORD", "MORPHOLOGICAL", "SEMANTIC_CHANGE", "PHONETIC_SPELLING",
    "NAMED_ENTITY_OR_RARE", "SPURIOUS_INSERTION", "OMISSION", "OTHER",
]
LLM_CATEGORY_LABELS = {
    "NUMBER_WORD": "number/word",
    "WORD_BOUNDARY": "word boundary",
    "WORD_ORDER": "word order",
    "CLITIC_MARKER": "clitic/contraction",
    "FUNCTION_WORD": "function word",
    "MORPHOLOGICAL": "morphological",
    "SEMANTIC_CHANGE": "semantic change",
    "PHONETIC_SPELLING": "phonetic/spelling",
    "NAMED_ENTITY_OR_RARE": "named entity/rare",
    "SPURIOUS_INSERTION": "insertion",
    "OMISSION": "omission",
    "OTHER": "other",
}
LLM_CATEGORY_COLORS = {
    "NUMBER_WORD": "#a855f7",
    "WORD_BOUNDARY": "#0891b2",
    "WORD_ORDER": "#92400e",
    "CLITIC_MARKER": "#7c3aed",
    "FUNCTION_WORD": "#65a30d",
    "MORPHOLOGICAL": "#16a34a",
    "SEMANTIC_CHANGE": "#2563eb",
    "PHONETIC_SPELLING": "#0ea5e9",
    "NAMED_ENTITY_OR_RARE": "#eab308",
    "SPURIOUS_INSERTION": "#f97316",
    "OMISSION": "#dc2626",
    "OTHER": "#6b7280",
}

OP_BASE_COLORS = {
    "substitutions": "#16a34a",
    "deletions": "#dc2626",
    "insertions": "#2563eb",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cat_label(cat: str) -> str:
    return LLM_CATEGORY_LABELS.get(cat, cat)


def _blend(hex_color: str, amount: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _op_color(op: str, idx: int, count: int) -> str:
    base = OP_BASE_COLORS[op]
    if count <= 1:
        return base
    return _blend(base, 0.12 + 0.45 * idx / max(1, count - 1))


def _fmt_sdi(s: int, d: int, i: int) -> str:
    return f"S {s} / D {d} / I {i}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    idx: int
    ref: str
    hyp: str
    wer: float = 0.0
    ref_words: int = 0
    subs: int = 0
    dels: int = 0
    ins: int = 0
    llm_categories: list[str] = field(default_factory=list)
    llm_notes: str = ""
    errors: list[dict] = field(default_factory=list)


@dataclass
class EvalResult:
    model: str
    task: str
    samples: list[Sample]
    wer: float = 0.0
    ref_words: int = 0
    subs: int = 0
    dels: int = 0
    ins: int = 0


def _normalized_pair(raw_text: str, raw_pred: str, lang: str) -> tuple[str, str]:
    ref = ml_normalize(raw_text, lang)
    hyp = ml_normalize(raw_pred, lang)
    if ref.strip():
        ref, hyp = normalize_compound_pair(ref, hyp)
    return ref, hyp


def _get_normalized(rec: dict, lang: str) -> tuple[str, str]:
    """Prefer pre-normalized fields stored by analyze_errors.py; fall back to
    computing them on the fly for un-analyzed records."""
    if "text_norm" in rec and "pred_text_norm" in rec:
        return rec["text_norm"], rec["pred_text_norm"]
    return _normalized_pair(rec.get("text", ""), rec.get("pred_text", ""), lang)


def _compute_sample_metrics(ref: str, hyp: str) -> dict:
    if not ref.strip():
        return {"ref": ref, "hyp": hyp, "wer": 0.0, "ref_words": 0, "subs": 0, "dels": 0, "ins": 0}
    out = process_words(ref, hyp)
    rw = out.hits + out.substitutions + out.deletions
    return {
        "ref": ref, "hyp": hyp,
        "wer": (out.substitutions + out.deletions + out.insertions) / max(1, rw),
        "ref_words": rw,
        "subs": out.substitutions,
        "dels": out.deletions,
        "ins": out.insertions,
    }


@st.cache_data(show_spinner="Loading results…")
def load_all_results(results_dir: str, cache_key: tuple) -> list[EvalResult]:
    """Read leaderboard manifests from `results_dir`. Records may be either
    untouched (plain text/pred_text) or enriched by analyze_errors.py with
    `wer`, `subs`, `dels`, `ins`, `errors`, `llm_categories`, `llm_notes`,
    plus pre-normalized `text_norm` / `pred_text_norm`. The dashboard uses
    the normalized fields directly when present (fast), falling back to
    on-the-fly normalization for un-analyzed records.

    `cache_key` (unused inside) is just the tuple of file mtimes so Streamlit
    invalidates the cache whenever the underlying results change.
    """
    del cache_key  # marker only
    base = Path(results_dir)
    results: list[EvalResult] = []

    for fpath in sorted(base.glob("MODEL_*.jsonl")):
        m = FNAME_RE.match(fpath.name)
        if not m:
            continue
        model, task = m["model"], m["task"]
        lang = detect_lang(task)
        with fpath.open() as f:
            records = [json.loads(line) for line in f if line.strip()]

        samples: list[Sample] = []
        for i, r in enumerate(records):
            ref_n, hyp_n = _get_normalized(r, lang)
            if "wer" in r:
                samples.append(Sample(
                    idx=r.get("idx", i), ref=ref_n, hyp=hyp_n,
                    wer=r["wer"], ref_words=r.get("ref_words", 0),
                    subs=r.get("subs", 0), dels=r.get("dels", 0), ins=r.get("ins", 0),
                    llm_categories=r.get("llm_categories", []),
                    llm_notes=r.get("llm_notes", ""),
                    errors=r.get("errors", []),
                ))
            else:
                samples.append(Sample(idx=i, **_compute_sample_metrics(ref_n, hyp_n)))

        total_rw = sum(s.ref_words for s in samples)
        total_s = sum(s.subs for s in samples)
        total_d = sum(s.dels for s in samples)
        total_i = sum(s.ins for s in samples)
        corpus_wer = (total_s + total_d + total_i) / max(1, total_rw)

        results.append(EvalResult(
            model=model, task=task, samples=samples,
            wer=corpus_wer, ref_words=total_rw,
            subs=total_s, dels=total_d, ins=total_i,
        ))
    return results


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def _inject_css():
    st.markdown("""<style>
    .aligned-seq { display:flex; flex-wrap:wrap; align-items:flex-start; gap:0.35rem 0.5rem; }
    .aligned-tok { display:inline-flex; flex-direction:column; align-items:center; }
    .aligned-top, .aligned-bottom { white-space:nowrap; min-height:1.3rem; }
    .aligned-top-plain { color:inherit; }
    .aligned-top-sub, .aligned-bottom-sub { color:#067647; font-weight:600; }
    .aligned-top-ins { color:#0b63b6; font-weight:600; }
    .aligned-bottom-del { color:#b42318; font-weight:600; }
    .aligned-empty { visibility:hidden; }
    .detail-card { border:1px solid rgba(128,128,128,0.18); border-radius:0.75rem;
                   padding:0.85rem 1rem; margin-bottom:0.75rem;
                   background:rgba(127,127,127,0.08); color:inherit; }
    .mini-meta { color:inherit; opacity:0.65; font-size:0.92rem; margin-bottom:0.6rem; }
    .diff-line { line-height:1.9; word-wrap:break-word; }
    .llm-marker { width:0.72rem; height:0.72rem; border-radius:0.18rem;
                  display:inline-block; border:1px solid rgba(0,0,0,0.08); margin-right:0.15rem; }
    .llm-badge { display:inline-flex; align-items:center; border-radius:999px;
                 padding:0.13rem 0.44rem; font-size:0.67rem; font-weight:600; color:white; margin:0.08rem; }
    .hyp-text { line-height:2.2; font-size:0.95rem; }
    .err-wrong { background:#fecaca; color:#991b1b; text-decoration:line-through;
                 border-radius:0.2rem; padding:0.1rem 0.2rem; }
    .err-correct { background:#bbf7d0; color:#166534;
                   border-radius:0.2rem; padding:0.1rem 0.2rem; }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        height:auto !important; white-space:normal !important; line-height:1.3 !important;
        padding:0.25rem 0.4rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] span[data-baseweb="tag"] > span {
        white-space:normal !important; overflow:visible !important; text-overflow:unset !important;
    }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] > div > div,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div > div {
        max-height:40vh !important; height:auto !important; overflow-y:auto !important;
    }
    /* Compact tables so both WER summary and Sample Browser fit the viewport
       without affecting chart fonts. */
    [data-testid="stDataFrame"] {
        font-size: 0.80rem !important;
    }
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrame"] [role="columnheader"] {
        font-size: 0.80rem !important;
    }
    </style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Diff rendering
# ---------------------------------------------------------------------------
def _fmt_tok(top: str, bottom: str, top_cls: str, bot_cls: str) -> str:
    t = html_mod.escape(top) if top else "&nbsp;"
    b = html_mod.escape(bottom) if bottom else "&nbsp;"
    tc = f"{top_cls}{' aligned-empty' if not top else ''}"
    bc = f"{bot_cls}{' aligned-empty' if not bottom else ''}"
    return (f"<span class='aligned-tok'>"
            f"<span class='aligned-top {tc}'>{t}</span>"
            f"<span class='aligned-bottom {bc}'>{b}</span></span>")


def render_error_spans(hyp: str, ref: str, errors: list[dict]) -> str:
    """Render hypothesis: wrong words struck-through on red, correct ref words on green after."""
    hyp_words = hyp.split()
    ref_words = ref.split()
    if not hyp_words and not ref_words:
        return "<span class='hyp-text'><i>(empty)</i></span>"

    # Sort errors by hyp_start position for sequential processing
    sorted_errors = sorted(errors, key=lambda e: (e.get("hyp_indices", [0])[0], e.get("ref_indices", [0])[0]))

    # Build output by walking hyp word positions
    # Track which hyp positions are consumed by errors
    consumed_hyp: set[int] = set()
    # error_at_hyp_pos[pos] = list of errors starting at that hyp position
    error_at_hyp: dict[int, list[dict]] = {}
    for err in sorted_errors:
        hi = err.get("hyp_indices", [0, 0])
        if len(hi) == 2:
            start = hi[0]
            error_at_hyp.setdefault(start, []).append(err)
            for wi in range(hi[0], hi[1]):
                consumed_hyp.add(wi)

    parts: list[str] = []
    i = 0
    while i <= len(hyp_words):
        # Check for errors starting at position i (including omissions where hyp start==end)
        if i in error_at_hyp:
            for err in error_at_hyp[i]:
                hi = err.get("hyp_indices", [i, i])
                ri = err.get("ref_indices", [])
                # Red: wrong/extra hyp words
                hyp_span = hyp_words[hi[0]:hi[1]] if len(hi) == 2 else []
                if hyp_span:
                    red = " ".join(html_mod.escape(w) for w in hyp_span)
                    parts.append(f"<span class='err-wrong'>{red}</span>")
                # Green: correct ref words
                ref_span = ref_words[ri[0]:ri[1]] if len(ri) == 2 else []
                if ref_span:
                    green = " ".join(html_mod.escape(w) for w in ref_span)
                    parts.append(f"<span class='err-correct'>{green}</span>")
            # Advance past the longest hyp span
            max_end = max((err.get("hyp_indices", [i, i])[1] for err in error_at_hyp[i]), default=i)
            i = max(max_end, i + 1) if max_end > i else i + 1
        elif i < len(hyp_words) and i not in consumed_hyp:
            parts.append(html_mod.escape(hyp_words[i]))
            i += 1
        else:
            i += 1

    return "<span class='hyp-text'>" + " ".join(parts) + "</span>"


def render_diff(ref: str, hyp: str) -> str:
    ref_toks, hyp_toks = ref.split(), hyp.split()
    if not ref_toks and not hyp_toks:
        return ""
    if not ref_toks:
        return "<div class='aligned-seq'>" + "".join(
            _fmt_tok(t, "", "aligned-top-ins", "") for t in hyp_toks
        ) + "</div>"
    if not hyp_toks:
        return "<div class='aligned-seq'>" + "".join(
            _fmt_tok("", t, "", "aligned-bottom-del") for t in ref_toks
        ) + "</div>"

    out = process_words(" ".join(ref_toks), " ".join(hyp_toks))
    parts: list[str] = []
    for chunk in (out.alignments[0] if out.alignments else []):
        rs = ref_toks[chunk.ref_start_idx:chunk.ref_end_idx]
        hs = hyp_toks[chunk.hyp_start_idx:chunk.hyp_end_idx]
        if chunk.type == "equal":
            parts.extend(_fmt_tok(t, "", "aligned-top-plain", "") for t in hs)
        elif chunk.type == "insert":
            parts.extend(_fmt_tok(t, "", "aligned-top-ins", "") for t in hs)
        elif chunk.type == "delete":
            parts.extend(_fmt_tok("", t, "", "aligned-bottom-del") for t in rs)
        else:
            for j in range(max(len(rs), len(hs))):
                h = hs[j] if j < len(hs) else ""
                r = rs[j] if j < len(rs) else ""
                tc = "aligned-top-sub" if h else "aligned-top-plain"
                bc = "aligned-bottom-sub" if r else "aligned-bottom-del"
                if not h:
                    tc = ""
                    bc = "aligned-bottom-del"
                if not r:
                    tc = "aligned-top-ins"
                    bc = ""
                parts.append(_fmt_tok(h, r, tc, bc))
    return "<div class='aligned-seq'>" + "".join(parts) + "</div>"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def render_wer_chart(df: pd.DataFrame, models: list[str], tasks: list[str]):
    st.subheader("WER by Model × Task (S/D/I)")
    chart = go.Figure()
    for mi, model in enumerate(models):
        mdf = df[df["model"] == model].set_index("task")
        for op, pct_col, label in [
            ("substitutions", "sub_pct", "sub"),
            ("deletions", "del_pct", "del"),
            ("insertions", "ins_pct", "ins"),
        ]:
            y = [float(mdf.at[t, pct_col]) if t in mdf.index else 0 for t in tasks]
            custom = [
                (float(mdf.at[t, "wer_pct"]) if t in mdf.index else 0,
                 int(mdf.at[t, "subs"]) if t in mdf.index else 0,
                 int(mdf.at[t, "dels"]) if t in mdf.index else 0,
                 int(mdf.at[t, "ins"]) if t in mdf.index else 0)
                for t in tasks
            ]
            chart.add_bar(
                x=[tasks, [model] * len(tasks)],
                y=y,
                name=f"{model} · {label}",
                marker_color=_op_color(op, mi, len(models)),
                customdata=custom,
                hovertemplate=(
                    "<b>%{x[0]}</b> — " + model + "<br>"
                    f"{label}: " + "%{y:.2f}%<br>"
                    "WER: %{customdata[0]:.2f}%<br>"
                    "S/D/I: %{customdata[1]}/%{customdata[2]}/%{customdata[3]}"
                    "<extra></extra>"
                ),
            )
    chart.update_layout(
        barmode="stack", xaxis_title="Task", yaxis_title="WER (%)",
        height=480, showlegend=False,
        margin=dict(l=8, r=8, t=10, b=8),
    )
    st.plotly_chart(chart, use_container_width=True)


def render_llm_chart(df: pd.DataFrame, models: list[str], tasks: list[str]):
    st.subheader("WER by LLM Error Categories")

    # Collect present categories across all rows
    present_cats = []
    for cat in LLM_CATEGORY_ORDER:
        for _, row in df.iterrows():
            cats = row["llm_cats"]
            if isinstance(cats, dict) and cats.get(cat, 0) > 0:
                present_cats.append(cat)
                break

    if not present_cats:
        st.info("No LLM error analysis data available. Run `analyze_errors.py` to populate this chart.")
        return

    # Show which models have LLM data
    models_with_llm = set()
    for _, row in df.iterrows():
        cats = row["llm_cats"]
        if isinstance(cats, dict) and any(v > 0 for v in cats.values()):
            models_with_llm.add(row["model"])
    models_without = set(models) - models_with_llm
    if models_without:
        st.caption(f"LLM analysis available for: {', '.join(sorted(models_with_llm))}. "
                   f"Missing for: {', '.join(sorted(models_without))}.")
    chart = go.Figure()
    for cat in present_cats:
        x_outer, x_inner, y_vals = [], [], []
        for task in tasks:
            for model in models:
                rdf = df[(df["task"] == task) & (df["model"] == model)]
                if rdf.empty:
                    pct = 0.0
                else:
                    row = rdf.iloc[0]
                    cnt = row["llm_cats"].get(cat, 0) if isinstance(row["llm_cats"], dict) else 0
                    pct = cnt / max(1, int(row["ref_words"])) * 100
                x_outer.append(task)
                x_inner.append(model)
                y_vals.append(pct)
        chart.add_bar(
            x=[x_outer, x_inner], y=y_vals,
            name=_cat_label(cat),
            marker_color=LLM_CATEGORY_COLORS.get(cat, "#6b7280"),
        )
    chart.update_layout(
        barmode="stack", xaxis_title="Task",
        yaxis_title="LLM error units / ref words (%)",
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        margin=dict(l=8, r=8, t=140, b=8),
    )
    st.plotly_chart(chart, use_container_width=True)


# ---------------------------------------------------------------------------
# Example browser
# ---------------------------------------------------------------------------
def render_examples(all_results: list[EvalResult], sel_models: list[str], sel_tasks: list[str]):
    st.subheader("Sample Browser")
    if not sel_tasks or not sel_models:
        st.info("Select models and tasks above.")
        return

    # Collect all LLM categories present
    all_cats: set[str] = set()
    for r in all_results:
        if r.model in sel_models and r.task in sel_tasks:
            for s in r.samples:
                all_cats.update(s.llm_categories)
    avail_cats = [c for c in LLM_CATEGORY_ORDER if c in all_cats]

    filter_cats = st.multiselect("Filter by LLM category", avail_cats,
                                 format_func=_cat_label, key="filter_cats",
                                 placeholder="Choose categories")

    # Build rows
    rows: list[dict] = []
    for task in sel_tasks:
        task_results = {r.model: r for r in all_results if r.model in sel_models and r.task == task}
        if not task_results:
            continue
        first = next(iter(task_results.values()))
        for i, sample in enumerate(first.samples):
            row: dict = {"task": task, "idx": i, "ref": sample.ref}
            any_match = False
            max_wer = 0.0
            for model in sel_models:
                r = task_results.get(model)
                if r is None or i >= len(r.samples):
                    continue
                s = r.samples[i]
                pfx = f"{model}__"
                row[f"{pfx}hyp"] = s.hyp
                row[f"{pfx}wer"] = s.wer * 100
                row[f"{pfx}sdi"] = _fmt_sdi(s.subs, s.dels, s.ins)
                row[f"{pfx}cats"] = s.llm_categories
                row[f"{pfx}notes"] = s.llm_notes
                row[f"{pfx}errors"] = s.errors
                max_wer = max(max_wer, s.wer * 100)
                if filter_cats and set(filter_cats) & set(s.llm_categories):
                    any_match = True
            row["max_wer"] = max_wer
            if filter_cats and not any_match:
                continue
            rows.append(row)

    if not rows:
        st.info("No samples match filters.")
        return

    ex_df = pd.DataFrame(rows).sort_values("max_wer", ascending=False).reset_index(drop=True)

    # Table columns — WER and S/D/I per model only; ref/hyp are in the detail view below
    tcols = ["task", "idx", "max_wer"]
    ccfg: dict = {
        "task": st.column_config.TextColumn("Task", width="small"),
        "idx": st.column_config.NumberColumn("#", width="small"),
        "max_wer": st.column_config.NumberColumn("Max WER%", format="%.1f", width="small"),
    }
    for model in sel_models:
        pfx = f"{model}__"
        tcols.extend([f"{pfx}wer", f"{pfx}sdi"])
        short = model.split("-")[0] if len(model) > 20 else model
        ccfg[f"{pfx}wer"] = st.column_config.NumberColumn(f"{short} WER%", format="%.1f", width="small")
        ccfg[f"{pfx}sdi"] = st.column_config.TextColumn(f"{short} S/D/I", width="small")

    valid_tcols = [c for c in tcols if c in ex_df.columns]
    event = st.dataframe(
        ex_df[valid_tcols], use_container_width=True, hide_index=True,
        height=240, column_config=ccfg,
        on_select="rerun", selection_mode="single-row",
    )

    # Detail view
    selected_rows = []
    try:
        sel = getattr(event, "selection", None)
        if sel is not None:
            selected_rows = list(getattr(sel, "rows", []) or sel.get("rows", []))
    except Exception:
        pass

    if not selected_rows:
        st.info("Click a row above for detailed view.")
        return

    row = ex_df.iloc[int(selected_rows[0])]
    st.markdown(f"**{row['task']} / sample {int(row['idx'])}**")

    audio_path = AUDIO_DIR / str(row["task"]) / f"sample_{int(row['idx'])}.flac"
    if audio_path.exists():
        st.audio(str(audio_path))

    # Ground truth card — full width
    st.markdown(
        f"<div class='detail-card'>"
        f"<div class='mini-meta'><b>Ground truth</b></div>"
        f"<div class='hyp-text'>{html_mod.escape(str(row['ref']))}</div></div>",
        unsafe_allow_html=True,
    )

    # Model cards — full width, one per model
    for model in sel_models:
        pfx = f"{model}__"
        hyp = row.get(f"{pfx}hyp", "")
        if pd.isna(hyp):
            continue
        wer_val = row.get(f"{pfx}wer", 0)
        sdi = row.get(f"{pfx}sdi", "")
        cats = row.get(f"{pfx}cats", [])
        errors = row.get(f"{pfx}errors", [])
        hyp_html = render_error_spans(str(hyp), str(row["ref"]), errors if isinstance(errors, list) else [])
        cat_html = ""
        if cats:
            cat_html = "<div style='margin-bottom:0.4rem'>" + "".join(
                f"<span class='llm-badge' style='background:{LLM_CATEGORY_COLORS.get(c, '#6b7280')}'>"
                f"{html_mod.escape(_cat_label(c))}</span>"
                for c in cats
            ) + "</div>"
        st.markdown(
            f"<div class='detail-card'>"
            f"<div class='mini-meta'><b>{html_mod.escape(model)}</b> | WER {wer_val:.1f}% | {sdi}</div>"
            f"{cat_html}"
            f"<div>{hyp_html}</div></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# WER summary table
# ---------------------------------------------------------------------------
def render_wer_table(df: pd.DataFrame, models: list[str], tasks: list[str]):
    rows = []
    per_model_wers: dict[str, list[float]] = {m: [] for m in models}
    for task in tasks:
        row: dict = {"Task": task}
        for model in models:
            rdf = df[(df["task"] == task) & (df["model"] == model)]
            if rdf.empty:
                row[model] = "-"
            else:
                r = rdf.iloc[0]
                row[model] = f"{r['wer_pct']:.2f}%  ({_fmt_sdi(int(r['subs']), int(r['dels']), int(r['ins']))})"
                per_model_wers[model].append(float(r["wer_pct"]))
        rows.append(row)

    avg_row: dict = {"Task": "AVG"}
    for model in models:
        vals = per_model_wers[model]
        avg_row[model] = f"{sum(vals) / len(vals):.2f}%" if vals else "-"
    rows.append(avg_row)

    tdf = pd.DataFrame(rows)
    st.dataframe(tdf, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="ASR Dashboard", layout="wide")
    _inject_css()
    st.title("ASR Evaluation Dashboard")

    cache_key = tuple(sorted(
        (p.name, p.stat().st_mtime) for p in RESULTS_DIR.glob("MODEL_*.jsonl")
    ))
    all_results = load_all_results(str(RESULTS_DIR), cache_key)
    if not all_results:
        st.error(f"No result files found in `{RESULTS_DIR}`")
        return

    all_models = sorted({r.model for r in all_results})
    all_tasks = sorted({r.task for r in all_results})

    st.sidebar.header("Filters")
    sel_models = st.sidebar.multiselect("Models", all_models, default=all_models)
    sel_tasks = st.sidebar.multiselect("Tasks", all_tasks, default=all_tasks)

    if not sel_models or not sel_tasks:
        st.info("Select at least one model and one task.")
        return

    # Build summary dataframe
    summary_rows = []
    for r in all_results:
        if r.model not in sel_models or r.task not in sel_tasks:
            continue
        # Aggregate LLM categories by word-level contribution (S+D+I per error span);
        # summing these across all categories equals the task's WER numerator.
        llm_cats: dict[str, int] = {}
        for s in r.samples:
            for e in s.errors:
                sdi = e.get("sdi") or {}
                units = int(sdi.get("S", 0)) + int(sdi.get("D", 0)) + int(sdi.get("I", 0))
                if units:
                    cat = e.get("category", "OTHER")
                    llm_cats[cat] = llm_cats.get(cat, 0) + units
        summary_rows.append({
            "model": r.model, "task": r.task,
            "wer_pct": r.wer * 100, "ref_words": r.ref_words,
            "subs": r.subs, "dels": r.dels, "ins": r.ins,
            "sub_pct": r.subs / max(1, r.ref_words) * 100,
            "del_pct": r.dels / max(1, r.ref_words) * 100,
            "ins_pct": r.ins / max(1, r.ref_words) * 100,
            "llm_cats": llm_cats,
        })

    if not summary_rows:
        st.info("No data for selected filters.")
        return

    sdf = pd.DataFrame(summary_rows)
    tasks_ordered = [t for t in all_tasks if t in sel_tasks]
    models_ordered = [m for m in all_models if m in sel_models]

    render_wer_chart(sdf, models_ordered, tasks_ordered)
    render_wer_table(sdf, models_ordered, tasks_ordered)
    render_llm_chart(sdf, models_ordered, tasks_ordered)
    render_examples(all_results, sel_models, tasks_ordered)


if __name__ == "__main__":
    main()
