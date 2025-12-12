from __future__ import annotations

import os
import time
import zlib
import io
import wave
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import httpx

try:
    import torch  # type: ignore

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    from transformers import SequenceFeatureExtractor, PreTrainedTokenizer  # type: ignore
    from transformers.utils import logging as hf_logging  # type: ignore

    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    SequenceFeatureExtractor = PreTrainedTokenizer = None  # type: ignore
    hf_logging = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False

from .local_agreement import LocalAgreement

logger = logging.getLogger(__name__)

if TRANSFORMERS_AVAILABLE and hf_logging is not None:
    hf_logging.set_verbosity_error()  # type: ignore[attr-defined]

LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

if TORCH_AVAILABLE and torch is not None:
    torch.set_grad_enabled(False)  # type: ignore[call-arg]


def _compression_ratio(text: str) -> float:
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

    MAX_WORD_DURATION: float = 1.0
    GIBBERISH_THRESHOLD: float = 2.2

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

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers for API request."""
        headers: Dict[str, str] = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if self.lang_id:
            headers["X-Lang-Id"] = self.lang_id
        if self.model_name:
            headers["X-Model-Name"] = self.model_name
        return headers

    def _send_request(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Send audio to API and return parsed JSON response."""
        wav_bytes = self._audio_to_wav_bytes(audio, sample_rate)
        files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}

        resp = httpx.post(
            self.api_url,
            headers=self._build_headers(),
            files=files,
            timeout=self.request_timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()

        # Triton may return [ { ... } ]
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            data = data[0]
        return data

    def _parse_response(self, data: Dict[str, Any]) -> str:
        """Extract text from API response."""
        if "transcription" in data:
            return data["transcription"]
        if "text" in data and isinstance(data["text"], str):
            return data["text"]
        if "segments" in data and isinstance(data["segments"], list):
            return " ".join(seg.get("text", "") for seg in data["segments"])
        return ""

    def _approximate_word_tokens(
        self,
        text: str,
        audio_duration: float,
        buffer_start_time: float,
    ) -> List[Dict[str, Any]]:
        """Create tokens with approximate timestamps when real timestamps unavailable."""
        words = text.strip().split()
        if not words or audio_duration <= 0:
            return []

        num_words = len(words)
        per_word = audio_duration / num_words

        tokens: List[Dict[str, Any]] = []
        for i, w in enumerate(words):
            rel_start = i * per_word
            rel_end = (i + 1) * per_word
            duration = min(rel_end - rel_start, self.MAX_WORD_DURATION)

            token_start = buffer_start_time + rel_start
            token_end = token_start + duration

            tokens.append({
                "text": w + (" " if i < num_words - 1 else ""),
                "start": token_start,
                "end": token_end,
            })

        return tokens

    def transcribe(
        self,
        audio: np.ndarray,
        buffer_start_time: float,
        sample_rate: int,
    ) -> List[Dict[str, Any]]:
        data = self._send_request(audio, sample_rate)
        text = self._parse_response(data) or ""

        if not text.strip():
            return []

        if _compression_ratio(text) > self.GIBBERISH_THRESHOLD:
            return []

        audio_duration: float = len(audio) / sample_rate
        return self._approximate_word_tokens(text, audio_duration, buffer_start_time)

    @classmethod
    def from_env(
        cls,
        api_url: Optional[str] = None,
        api_auth_token: Optional[str] = None,
        api_model_name: Optional[str] = None,
        api_lang_id: Optional[str] = None,
        request_timeout_s: Optional[float] = None,
        bytes_per_sample: int = 2,
    ) -> "RemoteAPIBackend":
        """Create backend from kwargs or environment variables."""
        resolved_api_url = api_url or os.getenv("TRITON_URL", "")
        if not resolved_api_url:
            raise ValueError("TRITON_URL / api_url must be set")

        return cls(
            api_url=resolved_api_url,
            auth_token=(
                api_auth_token
                if api_auth_token is not None
                else os.getenv("TRITON_AUTH_TOKEN", "")
            ),
            model_name=(
                api_model_name
                if api_model_name is not None
                else os.getenv("TRITON_MODEL_NAME", "")
            ),
            lang_id=(
                api_lang_id
                if api_lang_id is not None
                else os.getenv("TRITON_LANG_ID", "")
            ),
            request_timeout_s=(
                request_timeout_s
                if request_timeout_s is not None
                else float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
            ),
            bytes_per_sample=bytes_per_sample,
        )



class RemoteAPITimestampsBackend(RemoteAPIBackend):
    """
    Backend that uses Triton JSON with metadata.chunks timestamps.
    Inherits from RemoteAPIBackend and adds support for word-level timestamps.
    """

    @classmethod
    def _parse_metadata_any(cls, metadata: object) -> Optional[Dict[str, Any]]:
        """
        Normalize metadata into a dict that may contain "chunks".
        Handles: dict, list[dict or str], JSON string.
        """
        import json

        if metadata is None:
            return None

        if isinstance(metadata, dict):
            return metadata

        if isinstance(metadata, str):
            try:
                loaded = json.loads(metadata)
            except json.JSONDecodeError:
                return None
            return cls._parse_metadata_any(loaded)

        if isinstance(metadata, list) and metadata:
            # Prefer dicts with "chunks"
            for m in metadata:
                if isinstance(m, dict) and "chunks" in m:
                    return m
            # Then any dict
            for m in metadata:
                if isinstance(m, dict):
                    return m
            # Then any JSON string
            for m in metadata:
                if isinstance(m, str):
                    try:
                        loaded = json.loads(m)
                        if isinstance(loaded, dict):
                            return loaded
                    except json.JSONDecodeError:
                        continue

        return None

    @staticmethod
    def _get_chunk_start(chunk: Dict[str, Any]) -> float:
        """Extract start timestamp from a chunk dict."""
        ts = chunk.get("timestamp") or chunk.get("timestamps") or chunk.get("time")
        if not ts or len(ts) < 1 or ts[0] is None:
            return 0.0
        try:
            return float(ts[0])
        except Exception:
            return 0.0

    def _parse_chunks_to_tokens(
        self,
        chunks: List[Dict[str, Any]],
        audio_duration: float,
        buffer_start_time: float,
    ) -> List[Dict[str, Any]]:
        """Convert metadata chunks with timestamps to token list."""
        tokens: List[Dict[str, Any]] = []

        for seg in sorted(chunks, key=self._get_chunk_start):
            seg_text = seg.get("text", "")
            ts = (
                seg.get("timestamp")
                or seg.get("timestamps")
                or seg.get("time")
            )
            if not ts or len(ts) != 2:
                continue

            start, end = ts
            if start is None:
                continue

            start = float(start)
            if end is None:
                if audio_duration - start < self.MAX_WORD_DURATION:
                    end = audio_duration
                else:
                    end = start + self.MAX_WORD_DURATION
            end = float(end)

            tokens.append({
                "text": seg_text,
                "start": buffer_start_time + start,
                "end": buffer_start_time + end,
            })

        return tokens

    def transcribe(
        self,
        audio: np.ndarray,
        buffer_start_time: float,
        sample_rate: int,
    ) -> List[Dict[str, Any]]:
        data = self._send_request(audio, sample_rate)

        audio_duration: float = len(audio) / sample_rate
        if audio_duration <= 0:
            return []

        # ---- 1) Prefer metadata.chunks with real timestamps ----
        raw_metadata = data.get("metadata")
        metadata = self._parse_metadata_any(raw_metadata)

        chunks = None
        if isinstance(metadata, dict):
            chunks = metadata.get("chunks")

        if isinstance(chunks, list) and chunks:
            text_from_chunks = " ".join(
                str(c.get("text", "")).strip() for c in chunks
            )
            if text_from_chunks and _compression_ratio(text_from_chunks) > self.GIBBERISH_THRESHOLD:
                return []

            return self._parse_chunks_to_tokens(chunks, audio_duration, buffer_start_time)

        # ---- 2) Fallback: approximate per word from plain text ----
        text = self._parse_response(data) or ""
        if not text.strip():
            return []

        if _compression_ratio(text) > self.GIBBERISH_THRESHOLD:
            return []

        return self._approximate_word_tokens(text, audio_duration, buffer_start_time)


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
        torch_dtype: "torch.dtype | None" = None,
        language: str = "en",
        feature_extractor: Optional["SequenceFeatureExtractor"] = None,
        tokenizer: Optional["PreTrainedTokenizer"] = None,
    ):
        if not TORCH_AVAILABLE or torch is None:
            raise ImportError(
                "LocalWhisperBackend requires PyTorch (`torch`) to be installed."
            )
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "LocalWhisperBackend requires `transformers` to be installed."
            )

        if platform == "apple":
            from ..apple import ASRPipeline

            device = "cpu"
        elif platform == "nvidia":
            from ..nvidia import ASRPipeline

            device = "cuda"
        else:
            raise ValueError(f"Invalid platform: {platform}")

        if torch_dtype is None:
            torch_dtype = torch.float16  # type: ignore[union-attr]

        self.chunk_length_s: float = chunk_length_s
        self.window_size: float = chunk_length_s - 2
        self.sample_rate: int = 16000
        self.device: str = device
        self.language: str = language

        self.asr_pipeline = ASRPipeline(
            model,
            model_size=model_size,
            chunk_length_s=chunk_length_s,
            torch_dtype=torch_dtype,
            device=device,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

        special_tokens: str = f"<|startoftranscript|><|{language}|><|transcribe|>"
        self.encoded_special_tokens = self.asr_pipeline.tokenizer(
            special_tokens, return_tensors="pt", add_special_tokens=False
        ).input_ids

    def transcribe(
        self,
        audio: np.ndarray,
        buffer_start_time: float,
        sample_rate: int,
    ) -> List[Dict[str, Any]]:
        audio_duration: float = len(audio) / sample_rate
        max_new_tokens = 64

        generate_kwargs: Dict[str, Any] = {
            "use_cache": True,
            "num_beams": 1,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "language": self.language,
        }

        result: Dict[str, Any] = self.asr_pipeline(
            audio,
            return_timestamps="word",
            generate_kwargs=generate_kwargs,
            chunk_length_s=self.chunk_length_s,
        )

        if _compression_ratio(result["text"]) > 2.2:
            return []

        generated_tokens: List[Dict[str, Any]] = []
        max_word_duration: float = 1.0

        for token in result["chunks"]:
            if token["timestamp"][1] is None:
                if audio_duration - token["timestamp"][0] < max_word_duration:
                    token["timestamp"] = (token["timestamp"][0], audio_duration)
                else:
                    token["timestamp"] = (
                        token["timestamp"][0],
                        token["timestamp"][0] + max_word_duration,
                    )
            generated_tokens.append(
                {
                    "text": token["text"],
                    "start": token["timestamp"][0] + buffer_start_time,
                    "end": token["timestamp"][1] + buffer_start_time,
                }
            )

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
        model: str = "",
        model_size: str = "S",
        chunk_length_s: int = 10,
        min_process_chunk_s: float = 0.5,
        use_vad: bool = False,
        agreement_history_size: int = 5,
        agreement_majority_threshold: int = 3,
        platform: str = "apple",
        torch_dtype: "torch.dtype | None" = None,
        language: str = "en",
        feature_extractor: Optional["SequenceFeatureExtractor"] = None,
        tokenizer: Optional["PreTrainedTokenizer"] = None,
        backend: Optional[TranscriptionBackend] = None,
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
        self.min_process_chunk_s: float = min_process_chunk_s
        self.window_size: float = chunk_length_s - 2
        self._pending_chunk: Optional[np.ndarray] = None

        # Choose backend via factory or use injected one
        self.backend = self._resolve_backend(
            backend=backend,
            use_remote_api=use_remote_api,
            model=model,
            model_size=model_size,
            chunk_length_s=chunk_length_s,
            platform=platform,
            torch_dtype=torch_dtype,
            language=language,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            api_url=api_url,
            api_auth_token=api_auth_token,
            api_model_name=api_model_name,
            api_lang_id=api_lang_id,
            request_timeout_s=request_timeout_s,
            bytes_per_sample=bytes_per_sample,
        )

        self.no_speech_streak: int = 0
        self.speech_threshold: float = 0.5
        self.vad_model = self._init_vad(use_vad)

        self.local_agreement = LocalAgreement(
            history_size=agreement_history_size,
            majority_threshold=agreement_majority_threshold,
        )

        self.current_audio_buffer: Optional[np.ndarray] = None
        self.buffer_start_time: float = 0.0
        self.current_time: float = 0.0
        self.audio_queue: List[np.ndarray] = []

    @staticmethod
    def _resolve_backend(
        backend: Optional[TranscriptionBackend],
        use_remote_api: bool,
        model: str,
        model_size: str,
        chunk_length_s: int,
        platform: str,
        torch_dtype: "torch.dtype | None",
        language: str,
        feature_extractor: Optional["SequenceFeatureExtractor"],
        tokenizer: Optional["PreTrainedTokenizer"],
        api_url: Optional[str],
        api_auth_token: Optional[str],
        api_model_name: Optional[str],
        api_lang_id: Optional[str],
        request_timeout_s: Optional[float],
        bytes_per_sample: int,
    ) -> TranscriptionBackend:
        """Resolve which backend to use: injected, remote, or local."""
        if backend is not None:
            return backend

        if use_remote_api:
            return RemoteAPIBackend.from_env(
                api_url=api_url,
                api_auth_token=api_auth_token,
                api_model_name=api_model_name,
                api_lang_id=api_lang_id,
                request_timeout_s=request_timeout_s,
                bytes_per_sample=bytes_per_sample,
            )

        if not model:
            raise ValueError("model is required when using LocalWhisperBackend")

        return LocalWhisperBackend(
            model=model,
            model_size=model_size,
            chunk_length_s=chunk_length_s,
            platform=platform,
            torch_dtype=torch_dtype,
            language=language,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

    @staticmethod
    def _init_vad(use_vad: bool) -> Any:
        """Initialize VAD model if requested."""
        if not use_vad:
            return None

        if not TORCH_AVAILABLE or torch is None:
            raise ImportError(
                "VAD is enabled but PyTorch (`torch`) is not installed."
            )
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad"
        )
        return model

    def __call__(
        self, chunk: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Main streaming entry point.

        Accepts small audio chunks (e.g. 0.05s), internally accumulates them
        until at least `min_process_chunk_s` seconds are available, and only
        then runs the backend processing.

        If not enough audio has been accumulated yet, returns empty
        transcriptions: ([], []).
        """
        self.add_new_chunk(chunk)
        return self.process_new_chunk()

    def add_new_chunk(self, chunk: np.ndarray) -> None:
        """
        Add a new *small* audio chunk to the internal accumulator.

        Small chunks are first accumulated in `_pending_chunk`. Once the
        accumulated duration reaches at least `min_process_chunk_s` seconds,
        the combined audio is pushed into `audio_queue` as a single larger
        chunk for downstream processing.
        """
        if chunk is None or len(chunk) == 0:
            return

        # Accumulate into pending buffer
        if self._pending_chunk is None:
            self._pending_chunk = chunk
        else:
            self._pending_chunk = np.concatenate([self._pending_chunk, chunk])

        pending_duration = len(self._pending_chunk) / self.sample_rate

        # If we have not yet reached the minimum processing window, do nothing.
        if pending_duration < self.min_process_chunk_s:
            return

        # Enough audio accumulated: push to main queue as one "big" chunk.
        self.audio_queue.append(self._pending_chunk)
        self._pending_chunk = None

    def process_new_chunk(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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

        # time.sleep(0.5)

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

        committed = (
            self.local_agreement.get_committed_words()
        )  # or get_last_commited_words() ?

        # If we have no committed words yet, fall back to a simple tail crop.
        if not committed:
            self.current_audio_buffer = self.current_audio_buffer[
                -int(max_seconds * self.sample_rate) :
            ]
            self.buffer_start_time = (
                self.current_time - len(self.current_audio_buffer) / self.sample_rate
            )
            return

        # Earliest time we would like to keep in the buffer
        target_start_time: float = 1.0 + self.current_time - max_seconds

        # Find the last committed word that ends before or at target_start_time
        cut_word = None
        for w in committed:
            if w["end"] <= target_start_time:
                cut_word = w
            else:
                break

        if cut_word is not None:
            # Cut buffer at the end of this committed word
            new_buffer_start_time: float = cut_word["end"]
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
                self.current_time - len(self.current_audio_buffer) / self.sample_rate
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
        self._pending_chunk = None
        self.buffer_start_time = 0.0
        self.current_time = 0.0
        self.audio_queue = []
        self.no_speech_streak = 0
        self.speech_threshold = 0.5
        self.local_agreement.clear()
