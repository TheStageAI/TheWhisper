import numpy as np
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk

logger = logging.getLogger(__name__)

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")


@dataclass
class DatasetConfig:
    """
    Configuration for a single HF dataset split.
    """

    dataset_name: str
    config_name: Optional[str]
    split: str
    language: Optional[str] = None
    text_column: Optional[str] = 'text'
    sampling_rate: int = 16000
    max_samples: Optional[int] = None
    min_duration_s: Optional[float] = 0.
    max_duration_s: Optional[float] = 30.
    filters: Optional[Dict[str, Iterable[Any]]] = None  # column -> allowed values
    streaming: bool = False
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    task_name: Optional[str] = None  # key for results; defaults to derived id
    cache_dir: Optional[str] = None
    signal_to_noise_ratio: Optional[float] = None
    load_from_disk: bool = False


# -------------------------------
# Dataset loading and preparation
# -------------------------------

def add_noise_with_snr(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add noise to `audio` at the desired SNR (in dB).
    
    Args:
        audio: Input audio signal
        noise: Noise signal (will be adjusted to match audio length and power)
        snr_db: Signal-to-noise ratio in decibels
        
    Returns:
        Noisy audio signal
    """
    # Ensure noise matches audio length by looping or truncating
    if len(noise) < len(audio):
        # Loop noise to match audio length
        repeats = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, repeats)[:len(audio)]
    elif len(noise) > len(audio):
        # Truncate noise to match audio length (random start)
        max_start = len(noise) - len(audio)
        start = np.random.randint(0, max_start + 1)
        noise = noise[start:start + len(audio)]
    
    # Calculate signal and noise power
    sig_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0 or sig_power == 0:
        # Noise or signal is silent, return original audio
        return audio
    
    # Convert SNR dB to linear ratio
    snr_linear = 10 ** (snr_db / 10)
    
    # Compute required noise power and scaling factor
    target_noise_power = sig_power / snr_linear
    scale = np.sqrt(target_noise_power / noise_power)
    
    # Scale noise and add to audio
    noisy_audio = audio + noise * scale
    return noisy_audio


def preprocess_dataset(
    ds: Union[Dataset, DatasetDict], cfg: DatasetConfig
) -> Union[Dataset, DatasetDict]:
    if isinstance(ds, DatasetDict):
        # We only care about the requested split. Assume the caller loaded the specific split.
        raise ValueError("Expected a single Dataset, got DatasetDict. Load with split=...")

    ds = ds.cast_column("audio", Audio(sampling_rate=cfg.sampling_rate))

    num_before = len(ds)

    # Apply categorical filters if provided
    if cfg.filters:
        for col, allowed in cfg.filters.items():
            allowed_set = set(allowed)
            ds = ds.filter(lambda x: x.get(col) in allowed_set)

    # Filter by duration in seconds if requested
    if cfg.min_duration_s is not None or cfg.max_duration_s is not None:
        min_s = cfg.min_duration_s if cfg.min_duration_s is not None else 0.0
        max_s = cfg.max_duration_s if cfg.max_duration_s is not None else float("inf")

        def _dur_ok(ex: Dict[str, Any]) -> bool:
            audio = ex.get("audio")
            if audio is None:
                return False
            dur = float(audio["array"].shape[0]) / float(audio["sampling_rate"])
            return (dur >= min_s) and (dur <= max_s)

        ds = ds.filter(_dur_ok)

    # Truncate sample count if requested
    if cfg.max_samples is not None and cfg.max_samples > 0:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))

    if cfg.signal_to_noise_ratio is not None:
        # Load MUSAN dataset for noise samples
        logger.info("Loading MUSAN dataset for noise augmentation with SNR=%s dB", cfg.signal_to_noise_ratio)
        musan_ds = load_dataset(
            "FluidInference/musan", 
            split="train", 
            cache_dir=cfg.cache_dir,
            token=os.environ.get("HF_TOKEN", None),
        )
        # Materialize noise samples (load up to 1000 samples)
        noise_samples = []
        for i, ex in enumerate(musan_ds):
            noise_samples.append(ex["audio"]["array"])
            if i >= 999:
                break
        logger.info("Loaded %d noise samples from MUSAN", len(noise_samples))
        
        def _add_musan_noise(x):
            audio = x["audio"]["array"]
            # Randomly select a noise sample
            noise = noise_samples[np.random.randint(0, len(noise_samples))]
            noisy_audio = add_noise_with_snr(audio, noise, cfg.signal_to_noise_ratio)
            return {
                "audio": {
                    "array": noisy_audio,
                    "sampling_rate": x["audio"]["sampling_rate"]
                }
            }
        
        ds = ds.map(_add_musan_noise)
        ds = ds.cast_column("audio", Audio(sampling_rate=cfg.sampling_rate))

    num_after = len(ds)
    logger.info(
        "Prepared dataset samples: %d -> %d (filters: %s, min_dur: %s, max_dur: %s)",
        num_before,
        num_after,
        list(cfg.filters.keys()) if cfg.filters else None,
        cfg.min_duration_s,
        cfg.max_duration_s,
    )

    return ds


def load_hf_dataset(cfg: DatasetConfig) -> Dataset:
    """
    Load and prepare a Hugging Face dataset per the config.
    """
    logger.info(
        "Loading dataset %s/%s split=%s streaming=%s",
        cfg.dataset_name,
        cfg.config_name,
        cfg.split,
        cfg.streaming,
    )
    if cfg.load_from_disk:
        ds = load_from_disk(
            cfg.dataset_name, 
            cfg.config_name, 
            **cfg.dataset_kwargs
        )
    else:
        ds = load_dataset(
            cfg.dataset_name, 
            cfg.config_name, 
            split=cfg.split, 
            streaming=cfg.streaming, 
            cache_dir=cfg.cache_dir, 
            token=os.environ.get("HF_TOKEN", None),
            trust_remote_code=True,
            **cfg.dataset_kwargs
        )
        if cfg.streaming:
            # Convert to a finite list if max_samples specified; otherwise, leave as iterable
            if cfg.max_samples is not None and cfg.max_samples > 0:
                # Materialize up to max_samples
                items = []
                for i, ex in enumerate(ds):
                    items.append(ex)
                    if i + 1 >= cfg.max_samples:
                        break
                ds = Dataset.from_list(items)
            else:
                # For streaming without a sample cap, force materialization to allow transforms/filters
                ds = Dataset.from_list(list(ds))

    ds = preprocess_dataset(ds, cfg)
    logger.info(
        "Loaded dataset %s/%s split=%s: %d samples",
        cfg.dataset_name,
        cfg.config_name,
        cfg.split,
        len(ds),
    )
    return ds


def open_asr_en_tasks(
    max_samples: Optional[int] = None, 
    min_duration_s: Optional[float] = None, 
    max_duration_s: Optional[float] = None,
    split: Optional[str] = "test",
    cache_dir: Optional[str] = None,
    signal_to_noise_ratio: Optional[float] = None,
) -> List[DatasetConfig]:
    """
    Open ASR tasks from the ESB datasets.
    """
    return [
        DatasetConfig(
            dataset_name="hf-audio/esb-datasets-test-only-sorted",
            config_name="voxpopuli",
            split=split,
            language='en',
            text_column="text",
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            max_samples=max_samples,
            task_name="voxpopuli_test",
            cache_dir=cache_dir,
            signal_to_noise_ratio=signal_to_noise_ratio,
        ),
        DatasetConfig(
            dataset_name="hf-audio/esb-datasets-test-only-sorted",
            config_name="ami",
            split=split,
            language='en',
            text_column="text",
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            max_samples=max_samples,
            task_name="ami_test",
            cache_dir=cache_dir,
            signal_to_noise_ratio=signal_to_noise_ratio,
        ),
        DatasetConfig(
            dataset_name="hf-audio/esb-datasets-test-only-sorted",
            config_name="earnings22",
            split=split,
            language='en',
            text_column="text",
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            max_samples=max_samples,
            task_name="earnings22_test",
            cache_dir=cache_dir,
            signal_to_noise_ratio=signal_to_noise_ratio,
        ),
        DatasetConfig(
            dataset_name="hf-audio/esb-datasets-test-only-sorted",
            config_name="gigaspeech",
            split=split,
            language='en',
            text_column="text",
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            max_samples=max_samples,
            task_name="gigaspeech_test",
            cache_dir=cache_dir,
            signal_to_noise_ratio=signal_to_noise_ratio,
        ),
        DatasetConfig(
            dataset_name="hf-audio/esb-datasets-test-only-sorted",
            config_name="librispeech",
            split=split + ".clean",
            language='en',
            text_column="text",
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            max_samples=max_samples,
            task_name="librispeech_clean_test",
            cache_dir=cache_dir,
            signal_to_noise_ratio=signal_to_noise_ratio,
        ),
        DatasetConfig(
            dataset_name="hf-audio/esb-datasets-test-only-sorted",
            config_name="librispeech",
            split=split + ".other",
            language='en',
            text_column="text",
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            max_samples=max_samples,
            task_name="librispeech_other_test",
            cache_dir=cache_dir,
            signal_to_noise_ratio=signal_to_noise_ratio,
        ),
        DatasetConfig(
            dataset_name="hf-audio/esb-datasets-test-only-sorted",
            config_name="tedlium",
            split=split,
            language='en',
            text_column="text",
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            max_samples=max_samples,
            task_name="tedlium_test",
            cache_dir=cache_dir,
            signal_to_noise_ratio=signal_to_noise_ratio,
        ),
        DatasetConfig(
            dataset_name="hf-audio/esb-datasets-test-only-sorted",
            config_name="spgispeech",
            split=split,
            language='en',
            text_column="text",
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            max_samples=max_samples,
            task_name="spgispeech_test",
            cache_dir=cache_dir,
            signal_to_noise_ratio=signal_to_noise_ratio,
        )
    ]


def open_asr_multilingual_tasks(
    max_samples: Optional[int] = None,
    min_duration_s: Optional[float] = None,
    max_duration_s: Optional[float] = None,
    split: Optional[str] = "test",
    cache_dir: Optional[str] = None,
    signal_to_noise_ratio: Optional[float] = None,
) -> List[DatasetConfig]:
    
    mls_langs = ["french", "italian", "spanish", "portuguese", "german"]
    mls_configs = [
        DatasetConfig(
            dataset_name="facebook/multilingual_librispeech",
            config_name=language,
            split=split,
            language=language,
            text_column="transcript",
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            max_samples=max_samples,
            task_name=f"mls_{language}_test",
            cache_dir=cache_dir,
            signal_to_noise_ratio=signal_to_noise_ratio,
        ) for language in mls_langs
    ]

    fleurs_langs = ["fr_fr", "it_it", "es_419", "pt_br", "de_de"]
    fleurs_configs = [
        DatasetConfig(
            dataset_name="google/fleurs",
            config_name=language,
            split=split,
            language=language.split('_')[0],
            text_column="transcription",
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            max_samples=max_samples,
            task_name=f"fleurs_{language.split('_')[0]}_test",
            cache_dir=cache_dir,
            signal_to_noise_ratio=signal_to_noise_ratio,
        ) for language in fleurs_langs
    ]

    covost_langs = ["fr_en", "it_en", "es_en", "pt_en", "de_en"]
    covost_configs = [
        DatasetConfig(
            dataset_name="fixie-ai/covost2",
            config_name=lang,
            split=split,
            language=lang.split('_')[0],
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            max_samples=max_samples,
            task_name=f"covost2_{lang.split('_')[0]}_{split}",
            cache_dir=cache_dir,
            signal_to_noise_ratio=signal_to_noise_ratio,
            text_column="sentence",
        ) for lang in covost_langs
    ]

    return mls_configs + covost_configs + fleurs_configs
