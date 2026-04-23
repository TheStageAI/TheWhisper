"""Multilingual text normalizer used by Open ASR Leaderboard.

Vendored from open_asr_leaderboard/normalizer (Apache-2.0, OpenAI & HuggingFace):
- https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/normalizer.py
- https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/data_utils.py

Used to keep our offline WER computations byte-identical to the leaderboard's.
Kept as a small self-contained module to avoid depending on the leaderboard repo
(or on the similarly-named `whisper_normalizer` pypi package, whose classes are
not a drop-in match).
"""
from __future__ import annotations

import re
import unicodedata

import num2words

ADDITIONAL_DIACRITICS = {
    "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
    "ß": "ss", "ẞ": "SS", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "th", "ł": "l", "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep: str = "") -> str:
    def replace_character(char: str) -> str:
        if char in keep:
            return char
        if char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]
        if unicodedata.category(char) == "Mn":
            return ""
        if unicodedata.category(char)[0] in "MSP":
            return " "
        return char

    return "".join(replace_character(c) for c in unicodedata.normalize("NFKD", s))


def remove_symbols(s: str) -> str:
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    )


class BasicMultilingualTextNormalizer:
    def __init__(self, remove_diacritics: bool = True):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols

    def __call__(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
        s = re.sub(r"\(([^)]+?)\)", "", s)
        s = self.clean(s).lower()
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s


class MultilingualNormalizer(BasicMultilingualTextNormalizer):
    """Adds digit-to-word conversion on top of BasicMultilingualTextNormalizer."""

    def _normalize_numbers(self, text: str, lang: str) -> str:
        text = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)

        def _replace(m: re.Match) -> str:
            try:
                return num2words.num2words(int(m.group()), lang=lang)
            except Exception:
                return m.group()

        return re.sub(r"\d+", _replace, text)

    def __call__(self, s: str, lang: str | None = None) -> str:
        s = super().__call__(s)
        if lang is not None:
            s = self._normalize_numbers(s, lang)
        return s


ml_normalize = MultilingualNormalizer(remove_diacritics=False)
