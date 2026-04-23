#!/usr/bin/env python3
"""Fetch FLEURS audio from HF and save locally for the dashboard.

Mirrors how `open_asr_leaderboard/transformers/run_eval_ml.py` loads the data
(`nithinraok/asr-leaderboard-datasets`, config `fleurs_{lang}`, split `test`,
resampled to 16 kHz) so sample order — and therefore `sample_{N}` naming —
matches the `audio_filepath` already stored in results/.

Writes one file per sample to:
    audio/fleurs_{lang}_test/sample_{N}.flac

Usage:
    python fetch_audio.py                 # all 6 FLEURS languages
    python fetch_audio.py --langs en de   # subset
    python fetch_audio.py --overwrite     # re-download existing files
"""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

DATASET = "nithinraok/asr-leaderboard-datasets"
LANGUAGES = ["en", "de", "fr", "it", "es", "pt"]
SAMPLE_RATE = 16_000

AUDIO_DIR = Path(__file__).resolve().parent / "audio"


def fetch_lang(lang: str, overwrite: bool) -> None:
    task = f"fleurs_{lang}_test"
    out_dir = AUDIO_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET, f"fleurs_{lang}", split="test", token=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    for idx, row in enumerate(tqdm(ds, desc=task)):
        fpath = out_dir / f"sample_{idx}.flac"
        if fpath.exists() and not overwrite:
            continue
        sf.write(fpath, row["audio"]["array"], SAMPLE_RATE, format="FLAC")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=LANGUAGES, choices=LANGUAGES)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    for lang in args.langs:
        fetch_lang(lang, args.overwrite)


if __name__ == "__main__":
    main()
