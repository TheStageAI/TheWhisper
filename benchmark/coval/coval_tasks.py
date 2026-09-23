"""Coval STT benchmark tasks (https://benchmarks.coval.ai/stt).

Manifests are read from the Coval GitHub repo, audio is downloaded from Coval's public
bucket into memory and checked against the manifest sha256.

Truncation modes:
  manifest    cut at speech_end_offset_ms from the manifest (what Coval runs)
  none        no cut, full clip
  clean-twin  cut at the offset of the matching stt-wildasr-clean clip; for noisegap the
              offset is shifted by the inserted silence (duration difference)
"""

import hashlib
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset

from data_utils import DatasetConfig

# Coval repo commit the manifests are read from
COVAL_COMMIT = "67550d367b9f6a47dbb9e49f96236e2c461b6d08"
MANIFEST_URL = ("https://raw.githubusercontent.com/coval-ai/benchmarks/{commit}"
                "/runner/src/coval_bench/datasets/manifests/{name}.json")
AUDIO_URL = "https://storage.googleapis.com/coval-benchmarks-datasets/{name}/{path}"

# stt-v1 is not included
DATASETS = ["stt-v3", "stt-wildasr-accent", "stt-wildasr-clean", "stt-wildasr-clipping",
            "stt-wildasr-farfield", "stt-wildasr-noisegap", "stt-wildasr-phonecodec",
            "stt-wildasr-reverb"]
CLEAN_TWIN = "stt-wildasr-clean"
INHERIT_OFFSET = {"stt-wildasr-reverb", "stt-wildasr-farfield",
                  "stt-wildasr-clipping", "stt-wildasr-phonecodec"}
SHIFTED_OFFSET = {"stt-wildasr-noisegap"}
TRIM_MODES = ("manifest", "none", "clean-twin")


def _fetch(url: str, attempts: int = 5) -> bytes:
    for attempt in range(attempts - 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError:
            time.sleep(2 ** attempt)
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


@lru_cache(maxsize=None)
def _manifest(name: str) -> dict:
    return json.loads(_fetch(MANIFEST_URL.format(commit=COVAL_COMMIT, name=name)))


def _offsets(name: str, mode: str) -> Dict[str, float]:
    """Cut offset in ms per clip path; empty dict means no cut."""
    if mode == "none":
        return {}
    items = _manifest(name)["items"]
    own = {i["path"]: i["speech_end_offset_ms"] for i in items
           if i.get("speech_end_offset_ms") is not None}
    # file names repeat across datasets, so only the degraded wildasr sets use the clean offsets
    if mode == "manifest" or name not in INHERIT_OFFSET | SHIFTED_OFFSET:
        return own
    clean = {i["path"]: i for i in _manifest(CLEAN_TWIN)["items"]}
    out = {}
    for item in items:
        source = clean[item["path"]]
        if source.get("speech_end_offset_ms") is None:
            continue
        offset = source["speech_end_offset_ms"]
        if name in SHIFTED_OFFSET:
            offset += (item["duration_sec"] - source["duration_sec"]) * 1000.0
        out[item["path"]] = offset
    return out


def _download(name: str, item: dict) -> bytes:
    data = _fetch(AUDIO_URL.format(name=name, path=item["path"]))
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise SystemExit(f"{name}/{item['path']}: sha256 does not match the manifest")
    return data


def _build(name: str, mode: str) -> Dataset:
    items = _manifest(name)["items"]
    with ThreadPoolExecutor(max_workers=16) as pool:
        files = list(pool.map(partial(_download, name), items))
    cut_at = _offsets(name, mode)

    audio, text, trimmed = [], [], 0
    for item, data in zip(items, files):
        wave, rate = sf.read(io.BytesIO(data), dtype="float32")
        offset = cut_at.get(item["path"])
        if offset is not None:
            keep = int(round(offset / 1000.0 * rate))
            if 0 < keep < len(wave):
                wave = wave[:keep]
                trimmed += 1
        audio.append({"array": np.asarray(wave, dtype=np.float32), "sampling_rate": rate})
        text.append(item["transcript"])
    print(f"{name}: {len(audio)} clips downloaded, sha256 verified, {trimmed} truncated ({mode})")
    return Dataset.from_dict({"audio": audio, "text": text}).cast_column(
        "audio", Audio(sampling_rate=16000))


def coval_tasks(
    trim: str = "manifest",
    min_duration_s: Optional[float] = None,
    max_duration_s: Optional[float] = None,
    max_samples: Optional[int] = None,
) -> List[DatasetConfig]:
    if trim not in TRIM_MODES:
        raise SystemExit(f"unknown truncation mode {trim!r}; choose from {TRIM_MODES}")
    return [
        DatasetConfig(
            dataset_name=name,
            config_name=None,
            split="test",
            language="en",
            text_column="text",
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            max_samples=max_samples,
            task_name=name,
            builder=partial(_build, name, trim),
        )
        for name in DATASETS
    ]
