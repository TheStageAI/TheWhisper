import torch
import os
import numpy as np
from typing import List, Dict, Optional, Any
from transformers import SequenceFeatureExtractor, PreTrainedTokenizer
from transformers.utils import logging as hf_logging
import zlib
import io
import wave
import httpx
from abc import ABC, abstractmethod

from .local_agreement import LocalAgreement
from ..vad import batched_vad

import logging

logger = logging.getLogger(__name__)
hf_logging.set_verbosity_error()
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

torch.set_grad_enabled(False)


def _compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


# ======================
# Backend interface
# ======================


class TranscriptionBackend(ABC):
    """
    Strategy interface for turning an audio buffer into a list of tokens
    of the form {"text": str, "start": float, "end": float}.
    `start` / `end` are absolute times in seconds.
    """

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        buffer_start_time: float,
        sample_rate: int,
    ) -> List[Dict[str, Any]]: ...


class RemoteAPIBackend(TranscriptionBackend):
    """
    Backend that sends audio as WAV over HTTP to a remote ASR service.
    """

    def __init__(
        self,
        api_url: str,
        auth_token: str = "",
        model_name: str = "",
        lang_id: str = "",
        request_timeout_s: float = 60.0,
        bytes_per_sample: int = 2,
    ):
        if not api_url:
            raise ValueError("api_url must be provided for RemoteAPIBackend")

        self.api_url = api_url
        self.auth_token = auth_token
        self.model_name = model_name
        self.lang_id = lang_id
        self.request_timeout_s = request_timeout_s
        self.bytes_per_sample = bytes_per_sample

    def _audio_to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """
        Convert mono numpy audio (float32/float64 or int16) into WAV bytes
        compatible with the FastAPI gateway (16-bit PCM).
        """
        if audio.dtype != np.int16:
            audio_float = audio.astype(np.float32)
            audio_float = np.clip(audio_float, -1.0, 1.0)
            audio_int16 = (audio_float * 32767.0).astype(np.int16)
        else:
            audio_int16 = audio

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.bytes_per_sample)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        wav_buf.seek(0)
        return wav_buf.getvalue()

    def _parse_response(self, data: Dict[str, Any]) -> str:
        if "transcription" in data:
            return data["transcription"]
        if "text" in data and isinstance(data["text"], str):
            return data["text"]
        if "segments" in data and isinstance(data["segments"], list):
            return " ".join(seg.get("text", "") for seg in data["segments"])
        return ""

    def transcribe(
        self,
        audio: np.ndarray,
        buffer_start_time: float,
        sample_rate: int,
    ) -> List[Dict[str, Any]]:
        wav_bytes = self._audio_to_wav_bytes(audio, sample_rate)

        files = {
            "file": ("chunk.wav", wav_bytes, "audio/wav"),
        }

        headers: Dict[str, str] = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if self.lang_id:
            headers["X-Lang-Id"] = self.lang_id
        if self.model_name:
            headers["X-Model-Name"] = self.model_name

        resp = httpx.post(
            self.api_url,
            headers=headers,
            files=files,
            timeout=self.request_timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        text = self._parse_response(data) or ""

        if not text.strip():
            return []

        # Optional spam / gibberish filter
        if _compression_ratio(text) > 2.2:
            return []

        audio_duration: float = len(audio) / sample_rate
        words = text.strip().split()
        if not words or audio_duration <= 0:
            return []

        num_words = len(words)
        per_word = audio_duration / num_words
        max_word_duration: float = 1.0

        tokens: List[Dict[str, Any]] = []
        for i, w in enumerate(words):
            rel_start = i * per_word
            rel_end = (i + 1) * per_word
            duration = min(rel_end - rel_start, max_word_duration)

            token_start = buffer_start_time + rel_start
            token_end = token_start + duration

            tokens.append(
                {
                    "text": w + (" " if i < num_words - 1 else ""),
                    "start": token_start,
                    "end": token_end,
                }
            )

        return tokens


class LocalWhisperBackend(TranscriptionBackend):
    """
    Backend that wraps the original local ASRPipeline implementation.
    """

    def __init__(
        self,
        model: str,
        model_size: str = "S",
        chunk_length_s: int = 10,
        platform: str = "apple",
        torch_dtype: torch.dtype = torch.float16,
        language: str = "en",
        feature_extractor: Optional[SequenceFeatureExtractor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        if platform == "apple":
            from ..apple import ASRPipeline

            device = "cpu"
        elif platform == "nvidia":
            from ..nvidia import ASRPipeline

            device = "cuda"
        else:
            raise ValueError(f"Invalid platform: {platform}")

        self.chunk_length_s = chunk_length_s
        self.device = device
        self.language = language

        self.asr_pipeline = ASRPipeline(
            model,
            model_size=model_size,
            chunk_length_s=chunk_length_s,
            torch_dtype=torch_dtype,
            device=device,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

        self.chunk_length_s: float = chunk_length_s
        self.window_size: float = chunk_length_s - 2
        self.sample_rate: int = 16000
        self.language: str = language
        
        special_tokens: str = f"<|startoftranscript|><|{language}|><|transcribe|>"
        self.encoded_special_tokens: torch.Tensor = self.asr_pipeline.tokenizer(
            special_tokens, return_tensors="pt", add_special_tokens=False
        ).input_ids

    def transcribe(
        self,
        audio: np.ndarray,
        buffer_start_time: float,
        sample_rate: int,
    ) -> List[Dict[str, Any]]:
        
        audio_duration: float = len(audio) / sample_rate

        audio_duration: float = len(audio) / self.sample_rate
        max_new_tokens = 64

        generate_kwargs: Dict[str, Any] = {
            'use_cache': True,
            'num_beams': 1,
            'do_sample': False,
            'max_new_tokens': max_new_tokens,
            "language": self.language,
        }
        
        result: Dict[str, Any] = self.asr_pipeline(
            audio,
            return_timestamps='word',
            generate_kwargs=generate_kwargs,
            chunk_length_s=self.chunk_length_s,
        )
        
        if _compression_ratio(result['text']) > 2.2:
            return []
        
        generated_tokens: List[Dict[str, Any]] = []
        max_word_duration: float = 1.0

        for token in result['chunks']:
            if token['timestamp'][1] is None:
                if audio_duration - token['timestamp'][0] < max_word_duration:
                    token['timestamp'] = (token['timestamp'][0], audio_duration)
                else:
                    token['timestamp'] = (token['timestamp'][0], token['timestamp'][0] + max_word_duration)
            generated_tokens.append({
                'text': token['text'],
                'start': token['timestamp'][0] + buffer_start_time,
                'end': token['timestamp'][1] + buffer_start_time
            })
        
        return generated_tokens


# ======================
# StreamingPipeline
# ======================


class StreamingPipeline:
    """
    High-level streaming wrapper around a TranscriptionBackend.
    Maintains buffers, (optional) VAD, and local agreement logic.
    """

    def __init__(
        self,
        model: str,
        model_size: str = "S",
        chunk_length_s: int = 10,
        use_vad: bool = False,
        agreement_history_size: int = 5,
        agreement_majority_threshold: int = 3,
        platform: str = "apple",
        torch_dtype: torch.dtype = torch.float16,
        language: str = "en",
        feature_extractor: Optional[SequenceFeatureExtractor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        # New: injectable backend (strategy). If None, we fall back to constructing
        # a local or remote backend based on the flags below.
        backend: Optional[TranscriptionBackend] = None,
        # Optional convenience flags to auto-construct a RemoteAPIBackend
        use_remote_api: bool = False,
        api_url: Optional[str] = None,
        api_auth_token: Optional[str] = None,
        api_model_name: Optional[str] = None,
        api_lang_id: Optional[str] = None,
        request_timeout_s: Optional[float] = None,
        bytes_per_sample: int = 2,
        sample_rate: int = 16000,
    ):
        self.sample_rate: int = sample_rate
        self.chunk_length_s: float = chunk_length_s
        self.window_size: float = chunk_length_s - 2

        # Choose backend
        if backend is not None:
            self.backend = backend
        else:
            if use_remote_api:
                # Default remote backend based on environment / kwargs
                resolved_api_url = api_url or os.getenv("TRITON_URL", "")
                if not resolved_api_url:
                    raise ValueError(
                        "TRITON_URL / api_url must be set when use_remote_api=True"
                    )

                auth_token = (
                    api_auth_token
                    if api_auth_token is not None
                    else os.getenv("TRITON_AUTH_TOKEN", "")
                )
                model_name = (
                    api_model_name
                    if api_model_name is not None
                    else os.getenv("TRITON_MODEL_NAME", "")
                )
                lang_id = (
                    api_lang_id
                    if api_lang_id is not None
                    else os.getenv("TRITON_LANG_ID", "")
                )
                timeout_val = (
                    request_timeout_s
                    if request_timeout_s is not None
                    else float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
                )

                self.backend = RemoteAPIBackend(
                    api_url=resolved_api_url,
                    auth_token=auth_token,
                    model_name=model_name,
                    lang_id=lang_id,
                    request_timeout_s=timeout_val,
                    bytes_per_sample=bytes_per_sample,
                )
            else:
                # Default local backend (original behaviour)
                self.backend = LocalWhisperBackend(
                    model=model,
                    model_size=model_size,
                    chunk_length_s=chunk_length_s,
                    platform=platform,
                    torch_dtype=torch_dtype,
                    language=language,
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer,
                )

        self.no_speech_streak: int = 0
        self.speech_threshold: float = 0.5

        if use_vad:
            self.vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad"
            )
        else:
            self.vad_model = None

        # Local agreement maintains a history of transcriptions and decides which
        # words are "committed" (final) across multiple decoding passes.
        self.local_agreement = LocalAgreement(
            history_size=agreement_history_size,
            majority_threshold=agreement_majority_threshold,
        )

        self.current_audio_buffer: Optional[np.ndarray] = None
        self.buffer_start_time: float = 0.0
        self.current_time: float = 0.0

        self.audio_queue: List[np.ndarray] = []

    def __call__(self, chunk: np.ndarray) -> List[Dict[str, Any]]:
        """ """
        self.add_new_chunk(chunk)
        return self.process_new_chunk()

    def add_new_chunk(self, chunk: np.ndarray) -> None:
        """
        Add a new audio chunk to the processing queue.
        """
        self.audio_queue.append(chunk)

    def process_new_chunk(self) -> List[Dict[str, Any]]:
        """
        Process all queued audio chunks.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            (committed_words, not_committed_words)
        """
        if len(self.audio_queue) == 0:
            return [], []

        chunk: np.ndarray = np.concatenate(self.audio_queue)
        self.audio_queue = []
        self.current_time += len(chunk) / self.sample_rate

        # Extend the audio buffer with the new chunk
        if self.current_audio_buffer is None:
            self.current_audio_buffer = chunk
        else:
            self.current_audio_buffer = np.concatenate(
                [self.current_audio_buffer, chunk]
            )

        # Ensure the buffer does not grow unbounded
        if len(self.current_audio_buffer) > self.window_size * self.sample_rate:
            self._trim_audio_buffer()

        # Run transcription on the current buffer
        new_words = self._run_transcription(self.current_audio_buffer)

        # Update local agreement with the new transcription
        self.local_agreement.add_transcription(new_words)

        # Words that have just become "final" since the last call
        committed_now = self.local_agreement.get_last_commited_words()
        # Optional unstable tail that is not yet committed
        unstable_tail = self.local_agreement.not_committed_words

        return committed_now, unstable_tail

    def _trim_audio_buffer(self) -> None:
        """
        Trim the audio buffer to keep at most `self.window_size` seconds,
        preferably cutting at the end of a committed word to avoid
        chopping words that may still change in future transcriptions.
        """
        if self.current_audio_buffer is None or len(self.current_audio_buffer) == 0:
            return

        max_seconds: float = self.window_size

        committed = self.local_agreement.get_committed_words() # or get_last_commited_words() ? 

        # If we have no committed words yet, fall back to a simple tail crop.
        if not committed:
            self.current_audio_buffer = self.current_audio_buffer[
                -int(max_seconds * self.sample_rate) :
            ]
            self.buffer_start_time = (
                self.current_time
                - len(self.current_audio_buffer) / self.sample_rate
            )
            return

        # Earliest time we would like to keep in the buffer
        target_start_time: float = 1. + self.current_time - max_seconds

        # Find the last committed word that ends before or at target_start_time
        cut_word = None
        for w in committed:
            if w['end'] <= target_start_time:
                cut_word = w
            else:
                break

        if cut_word is not None:
            # Cut buffer at the end of this committed word
            new_buffer_start_time: float = cut_word['end']
            delta_sec: float = new_buffer_start_time - self.buffer_start_time
            if delta_sec > 0:
                delta_samples: int = int(delta_sec * self.sample_rate)
                self.current_audio_buffer = self.current_audio_buffer[delta_samples:]
                self.buffer_start_time = new_buffer_start_time
            else:
                # If for some numerical reason delta_sec is not positive,
                # fall back to a simple tail crop.
                self.current_audio_buffer = self.current_audio_buffer[
                    -int(max_seconds * self.sample_rate) :
                ]
                self.buffer_start_time = (
                    self.current_time
                    - len(self.current_audio_buffer) / self.sample_rate
                )
        else:
            # No committed word early enough: simple tail crop
            self.current_audio_buffer = self.current_audio_buffer[
                -int(max_seconds * self.sample_rate) :
            ]
            self.buffer_start_time = (
                self.current_time
                - len(self.current_audio_buffer) / self.sample_rate
            )
            
    def _run_transcription(
        self,
        audio: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Delegate to the backend.
        """
        new_tokens = self.backend.transcribe(
            audio=audio,
            buffer_start_time=self.buffer_start_time,
            sample_rate=self.sample_rate,
        )
        return new_tokens

    def clear(self) -> None:
        """
        Reset the pipeline to its initial state.
        """
        self.current_audio_buffer = None
        self.buffer_start_time = 0.0
        self.current_time = 0.0
        self.audio_queue = []
        self.no_speech_streak = 0
        self.speech_threshold = 0.5
