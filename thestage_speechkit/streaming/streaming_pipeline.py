from __future__ import annotations

import os
import zlib
import io
import wave
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import httpx

import torch
torch.set_grad_enabled(False)

from transformers import SequenceFeatureExtractor, PreTrainedTokenizer
from transformers.utils import logging as hf_logging

logger = logging.getLogger(__name__)

hf_logging.set_verbosity_error()


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

    def transcribe(
        self,
        audio: np.ndarray,
        buffer_start_time: float,
        sample_rate: int,
    ) -> List[Dict[str, Any]]:
        """Base class requires subclass implementation with real timestamps."""
        raise NotImplementedError(
            "Use RemoteAPITimestampsBackend for real word-level timestamps"
        )

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
        """Transcribe audio using real word-level timestamps from server."""
        data = self._send_request(audio, sample_rate)

        audio_duration: float = len(audio) / sample_rate
        if audio_duration <= 0:
            return []

        # Extract metadata.chunks with real timestamps
        raw_metadata = data.get("metadata")
        metadata = self._parse_metadata_any(raw_metadata)

        chunks = None
        if isinstance(metadata, dict):
            chunks = metadata.get("chunks")

        if not isinstance(chunks, list) or not chunks:
            logger.warning("No real timestamps in server response")
            return []

        text_from_chunks = " ".join(
            str(c.get("text", "")).strip() for c in chunks
        )
        if text_from_chunks and _compression_ratio(text_from_chunks) > self.GIBBERISH_THRESHOLD:
            return []

        return self._parse_chunks_to_tokens(chunks, audio_duration, buffer_start_time)


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
        self.window_size: float = chunk_length_s # - 2
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
        max_new_tokens = 128

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
    Maintains buffers, and postprocesses transcribtions.
    """

    def __init__(
        self,
        model: str = "",
        model_size: str = "S",
        chunk_length_s: int = 10,
        min_process_chunk_s: float = 0.5,
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
        self.window_size: float = chunk_length_s # - 2
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

        self.current_audio_buffer: Optional[np.ndarray] = None
        self.buffer_start_time: float = 0.0
        self.current_time: float = 0.0
        self.audio_queue: List[np.ndarray] = []

        self.history: List[List[Dict[str, Any]]] = []

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
            return RemoteAPITimestampsBackend.from_env(
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

        # Run transcription on the current buffer
        new_words = self._run_transcription(self.current_audio_buffer)
        new_words = self._postprocess_transcribtions(new_words)

        # update history
        self.history.append(new_words)

        # Ensure the buffer does not grow unbounded
        max_allowed_size = (self.window_size - self.min_process_chunk_s) * self.sample_rate
        if len(self.current_audio_buffer) > max_allowed_size:
            final_text = self._extract_final_text()
            truncation_time = self._get_truncation_time(final_text)
            self._trim_audio_buffer(truncation_time)
            commited_words = [word for word in final_text if word['end'] <= truncation_time]
            uncommited_words = [word for word in new_words if word['end'] > truncation_time]
            self.history = []
        else:
            commited_words = []
            uncommited_words = new_words

        return commited_words, uncommited_words

    def _postprocess_transcribtions(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Fuse tokens that are only spaces and dots into the previous token
        filtered_tokens = []
        for i, token in enumerate(tokens):
            text = token["text"]
            # Check if token contains only spaces and dots
            if text.strip() and all(c in (' ', '.') for c in text):
                # If there's a previous token, append the dot to it
                if filtered_tokens:
                    filtered_tokens[-1]["text"] += text.strip()
                    # Extend the end time of the previous token
                    # filtered_tokens[-1]["end"] = token["end"]
                # Skip adding this token to filtered_tokens
            else:
                filtered_tokens.append(token)
        
        # Add space to tokens that don't start with a space
        for token in filtered_tokens:
            if token["text"] and not token["text"].startswith(" "):
                token["text"] = " " + token["text"]
        
        for token in filtered_tokens:
            token['text'] = token['text'].replace('gonNA', 'gonna')
            token['text'] = token['text'].replace('gotTA', 'gotta')
            token['text'] = token['text'].replace('wanNA', 'wanna')

        if len(filtered_tokens) == 1 and filtered_tokens[0]['text'].strip() in ['The.', 'The']:
            filtered_tokens = []

        return filtered_tokens

    def _extract_final_text(self):
        return self.history[-1]

    def _get_truncation_time(self, final_words):
        last_end_of_sentence_index = None
        last_comma_index = None
        max_pause_index = None
        
        max_pause_duration = 0.0
        prev_word_end = 0.0
        last_word = len(final_words) - 1

        for i, word in enumerate(final_words):
            text = word['text'].strip()
            if (text.endswith('.') or text.endswith('?') or text.endswith('!')) and i != last_word:
                last_end_of_sentence_index = i
            
            if (text.endswith(',') or text.endswith(';') or text.endswith(':')) and i != last_word:
                last_comma_index = i
            
            if word['start'] - prev_word_end >= max_pause_duration:
                max_pause_duration = word['start'] - prev_word_end
                max_pause_index = i - 1
            
            prev_word_end = word['end']
        
        if last_end_of_sentence_index:
            out = final_words[last_end_of_sentence_index]['end']
        
        elif last_comma_index:
            out = final_words[last_comma_index]['end']
        
        elif max_pause_index is not None and max_pause_index >= 0:
            out = final_words[max_pause_index]['end']
        
        elif len(final_words) >= 2:
            out = final_words[-2]['end']
        
        elif len(final_words) == 1:
            out = final_words[0]['end']
        
        else:
            out = self.current_time - self.min_process_chunk_s * 2

        return out

    def _trim_audio_buffer(self, truncation_time: float) -> None:
        delta = truncation_time - self.buffer_start_time
        if delta > 0:
            delta_samples = int(delta * self.sample_rate)
            self.current_audio_buffer = self.current_audio_buffer[delta_samples:]
            self.buffer_start_time = truncation_time

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
