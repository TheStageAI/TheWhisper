"""Coval STT benchmark tasks (https://benchmarks.coval.ai/stt).

Manifests are read from the Coval GitHub repo, audio is downloaded from Coval's public
bucket into memory and checked against the manifest sha256.

Truncation modes:
  manifest    cut at speech_end_offset_ms from the manifest (what Coval runs)
  none        no cut, full clip
"""

import hashlib
import http.client
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from typing import List, Optional

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
TRIM_MODES = ("manifest", "none")


def _fetch(url: str, attempts: int = 5) -> bytes:
    for attempt in range(attempts - 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except (OSError, http.client.HTTPException):
            time.sleep(2 ** attempt)
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


@lru_cache(maxsize=None)
def _manifest(name: str) -> dict:
    return json.loads(_fetch(MANIFEST_URL.format(commit=COVAL_COMMIT, name=name)))


def _download(name: str, item: dict) -> bytes:
    data = _fetch(AUDIO_URL.format(name=name, path=item["path"]))
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise SystemExit(f"{name}/{item['path']}: sha256 does not match the manifest")
    return data


def _build(name: str, mode: str) -> Dataset:
    items = _manifest(name)["items"]
    with ThreadPoolExecutor(max_workers=16) as pool:
        files = list(pool.map(partial(_download, name), items))

    audio, text, trimmed = [], [], 0
    for item, data in zip(items, files):
        wave, rate = sf.read(io.BytesIO(data), dtype="float32")
        offset = item.get("speech_end_offset_ms") if mode == "manifest" else None
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
