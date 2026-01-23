import os
import logging
import json
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from datasets import Audio, Dataset, DatasetDict, load_dataset

import re
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer

from data_utils import DatasetConfig, load_hf_dataset


logger = logging.getLogger(__name__)
# Configure a basic logger if none configured by the host application
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")


NORMALIZERS_MAP = {
    'en': EnglishTextNormalizer(),
    'other': BasicTextNormalizer(remove_diacritics=True),
}


def normalize_texts(texts: Iterable[str], language: Optional[str]) -> List[str]:
    """
    Normalize text for the given language.
    """
    normalizer = NORMALIZERS_MAP.get(language, NORMALIZERS_MAP['other'])
    return [normalizer(text) for text in texts]


def _load_metric(name: str):
    from evaluate import load as _load

    return _load(name)


def compute_text_metrics(
    predictions: List[str], references: List[str], language: Optional[str]
) -> Dict[str, float]:
    """Compute corpus-level WER and CER."""
    norm_preds = normalize_texts(predictions, language)
    norm_refs = normalize_texts(references, language)

    wer_metric = _load_metric("wer")
    cer_metric = _load_metric("cer")

    wer_val = float(wer_metric.compute(predictions=norm_preds, references=norm_refs))
    cer_val = float(cer_metric.compute(predictions=norm_preds, references=norm_refs))

    return {
        "wer": wer_val,
        "cer": cer_val,
    }


def _derive_task_name(cfg: DatasetConfig) -> str:
    if cfg.task_name:
        return cfg.task_name
    comp = [cfg.dataset_name]
    if cfg.config_name:
        comp.append(cfg.config_name)
    comp.append(cfg.split)
    return "/".join(comp)


def evaluate_dataset(
    cfg: DatasetConfig,
    asr_generator: Any,
    generate_kwargs: Optional[Dict[str, Any]] = None,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Evaluate a single dataset split using the provided generation function.
    Returns a dict of metrics.
    
    Args:
        cfg: Dataset configuration.
        asr_generator: ASR generation function/pipeline.
        generate_kwargs: Extra kwargs passed to the generate function.
        batch_size: Batch size for generation.
    """
    task_name = _derive_task_name(cfg)
    logger.info("Starting evaluation for task: %s", task_name)
    ds = load_hf_dataset(cfg)
    
    generate_kwargs['language'] = cfg.language

    text_col = cfg.text_column

    # Prepare batches of audio arrays and references
    audio_list: List[Any] = []
    refs: List[str] = []
    durations: List[float] = []

    for ex in ds:
        audio = ex["audio"]
        audio_list.append(audio["array"])  # numpy array
        refs.append(ex.get(text_col, ex.get("text", "")))
        durations.append(float(audio["array"].shape[0]) / float(audio["sampling_rate"]))

    # Generation
    bs = max(1, int(batch_size))
    preds_raw: List[Any] = []

    start = time.time()
    total = len(audio_list)
    if total == 0:
        logger.warning("Task %s has 0 samples after filtering.", task_name)
        return {}
    total_batches = (total + bs - 1) // bs if total > 0 else 0
    log_every = max(1, total_batches // 10) if total_batches > 0 else 1
    for batch_idx, i in enumerate(range(0, total, bs)):
        batch_audio = audio_list[i : i + bs]
        out = asr_generator(batch_audio, generate_kwargs=generate_kwargs)
        if isinstance(out, list):
            preds_raw.extend(out)
        else:
            preds_raw.append(out)
        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == total_batches:
            logger.info(
                "Generated %d/%d batches (%.0f%%) for task %s",
                batch_idx + 1,
                total_batches,
                ((batch_idx + 1) / max(1, total_batches)) * 100.0,
                task_name,
            )
    gen_time = time.time() - start

    # Extract texts
    pred_texts: List[str] = []
    for item in preds_raw:
        if isinstance(item, str):
            pred_texts.append(item)
        elif isinstance(item, dict):
            pred_texts.append(item.get("text", ""))
        else:
            pred_texts.append(str(item))

    # Compute corpus-level metrics
    metrics = compute_text_metrics(pred_texts, refs, cfg.language)

    # RTFx (Real-Time Factor)
    total_audio_s = sum(durations) if len(durations) > 0 else 0.0
    # Save dataset duration in hours
    metrics["dataset_duration_hours"] = (total_audio_s / 3600.0) if total_audio_s > 0 else 0.0
    if total_audio_s > 0 and gen_time > 0:
        metrics["rtfx"] = total_audio_s / gen_time

    logger.info(
        "Finished task %s: samples=%d, gen_time=%.2fs, audio=%.2fs, RTFx=%.3f",
        task_name,
        len(refs),
        gen_time,
        total_audio_s,
        metrics.get("rtfx", float("nan")),
    )
    logger.info(
        "Metrics %s: WER=%.2f%%, CER=%.2f%%", 
        task_name,
        metrics.get("wer", 0.0) * 100.0,
        metrics.get("cer", 0.0) * 100.0,
    )

    return metrics


def evaluate_whisper(
    asr_generator,
    benchmark_tasks: List[DatasetConfig],
    generate_kwargs: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    batch_size: int = 64,
) -> Dict[str, Dict[str, float]]:
    """
    High-level evaluation entrypoint.

    Args:
        asr_generator: A HF pipeline.
        benchmark_tasks: List of DatasetConfig objects.
        generate_kwargs: Extra kwargs passed to the generate function/pipeline.
        save_path: Optional JSON file path to dump the aggregated results.
        print_high_wer: If True, print prediction-reference pairs with WER > wer_threshold.
        wer_threshold: WER threshold for printing high WER pairs (default 0.5 = 50%).

    Returns:
        Dict mapping task_name -> metrics dict.
    """
    logger.info("Starting benchmark with %d tasks", len(benchmark_tasks))
    results: Dict[str, Dict[str, float]] = {}
    for task in benchmark_tasks:
        name = _derive_task_name(task)
        logger.info("Running task: %s", name)
        metrics = evaluate_dataset(
            task, asr_generator, 
            generate_kwargs, 
            batch_size=batch_size,
        )
        results[name] = {
            k: (float(v) if v is not None else None) 
            for k, v in metrics.items()
        }  # type: ignore

    # Calculate mean WER, CER, and RTFx across all benchmark tasks
    wer_values = [m["wer"] for m in results.values() if "wer" in m and m["wer"] is not None]
    cer_values = [m["cer"] for m in results.values() if "cer" in m and m["cer"] is not None]
    rtfx_values = [m["rtfx"] for m in results.values() if "rtfx" in m and m["rtfx"] is not None]
    
    if wer_values or cer_values or rtfx_values:
        mean_metrics: Dict[str, float] = {}
        if wer_values:
            mean_metrics["wer"] = sum(wer_values) / len(wer_values)
        if cer_values:
            mean_metrics["cer"] = sum(cer_values) / len(cer_values)
        if rtfx_values:
            mean_metrics["rtfx"] = sum(rtfx_values) / len(rtfx_values)
        results["mean"] = mean_metrics
        logger.info(
            "Mean metrics: WER=%.2f%%, CER=%.2f%%, RTFx=%.2f",
            mean_metrics.get("wer", 0.0) * 100.0,
            mean_metrics.get("cer", 0.0) * 100.0,
            mean_metrics.get("rtfx", 0.0),
        )

    if save_path:
        # Load existing results if file exists
        existing_results = {}
        if os.path.exists(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                logger.info("Loaded existing results from %s", save_path)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Could not load existing results from %s: %s", save_path, e)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Update existing results with new values
        existing_results.update(results)
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)
        logger.info("Saved results to %s", save_path)

    return results
