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
        prefix_text: Optional[List[Dict[str, Any]]],
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
        prefix_text: Optional[List[Dict[str, Any]]],
        buffer_start_time: float,
        sample_rate: int,
    ) -> List[Dict[str, Any]]:
        # Prefix is currently ignored by the remote backend; context would be
        # handled by the remote service itself if supported.
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
        self.encoded_special_tokens: torch.Tensor = self.asr_pipeline.tokenizer(
            special_tokens, return_tensors="pt", add_special_tokens=False
        ).input_ids

    def transcribe(
        self,
        audio: np.ndarray,
        prefix_text: Optional[List[Dict[str, Any]]],
        buffer_start_time: float,
        sample_rate: int,
    ) -> List[Dict[str, Any]]:
        audio_duration: float = len(audio) / sample_rate

        if prefix_text is not None and len(prefix_text) > 0:
            # Time since last prefix word in absolute time; we only need the gap,
            # so subtract starts.
            time_since_last_prefix_word = (
                buffer_start_time + audio_duration - prefix_text[-1]["end"]
            )
            rounded_time = int(max(time_since_last_prefix_word, 0))
            max_new_tokens = 16 + rounded_time * 16
        else:
            max_new_tokens = 32

        generate_kwargs: Dict[str, Any] = {
            "use_cache": True,
            "num_beams": 1,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
        }

        if prefix_text is not None and len(prefix_text) > 0:
            prefix_text_str: str = "".join([w["text"] for w in prefix_text])
            decoder_input_ids: torch.Tensor = self.asr_pipeline.tokenizer(
                prefix_text_str, return_tensors="pt", add_special_tokens=False
            ).input_ids
            generate_kwargs["decoder_input_ids"] = torch.cat(
                [self.encoded_special_tokens, decoder_input_ids], dim=1
            ).to(self.device)

        result: Dict[str, Any] = self.asr_pipeline(
            audio,
            return_timestamps="word",
            generate_kwargs=generate_kwargs,
            chunk_length_s=self.chunk_length_s,
        )

        if _compression_ratio(result["text"]) > 2.2:
            return []

        max_word_duration: float = 1.0

        # Filter chunks similar to the original implementation
        new_chunks_lst = []
        max_end_time = 0.0
        for token in result["chunks"]:
            end_time = token["timestamp"][1]
            if end_time is not None:
                if end_time > max_end_time and end_time < self.chunk_length_s:
                    max_end_time = end_time
                    new_chunks_lst.append(token)
                else:
                    # drop token
                    pass
            else:
                new_chunks_lst.append(token)
        result["chunks"] = new_chunks_lst

        tokens: List[Dict[str, Any]] = []
        for token in result["chunks"]:
            start_rel, end_rel = token["timestamp"]
            if end_rel is None:
                if audio_duration - start_rel < max_word_duration:
                    end_rel = audio_duration
                else:
                    end_rel = start_rel + max_word_duration

            tokens.append(
                {
                    "text": token["text"],
                    "start": buffer_start_time + start_rel,
                    "end": buffer_start_time + end_rel,
                }
            )

        return tokens


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
        agreement_history_size: int = 2,
        agreement_majority_threshold: int = 2,
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
        self.window_size: float = 10.0  # seconds

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

        self.local_agreement = LocalAgreement(
            history_size=agreement_history_size,
            majority_threshold=agreement_majority_threshold,
        )

        self.current_audio_buffer: Optional[np.ndarray] = None
        self.buffer_start_time: float = 0.0
        self._prefix_tokens: List[Dict[str, Any]] = []
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
        contains_speech: bool = True
        self.current_time += len(chunk) / self.sample_rate

        if self.vad_model is not None:
            contains_speech = batched_vad(
                self.vad_model, chunk,
                sampling_rate=self.sample_rate,
                threshold=self.speech_threshold
            )
            if not contains_speech:
                self.no_speech_streak += 1
            else:
                self.no_speech_streak = 0

            if self.no_speech_streak >= 3:
                if self.current_audio_buffer is None:
                    self.current_audio_buffer = chunk
                else:
                    self.current_audio_buffer = np.concatenate([self.current_audio_buffer, chunk])

                if len(self.current_audio_buffer) > self.window_size * self.sample_rate:
                    self._trim_audio_buffer(self.window_size)

                return [], []

        if (
            self.current_audio_buffer is not None
            and len(self.current_audio_buffer) / self.sample_rate > 1
        ):
            self.current_audio_buffer = np.concatenate(
                [self.current_audio_buffer, chunk]
            )

            if len(self.current_audio_buffer) > self.window_size * self.sample_rate:
                self._trim_audio_buffer(self.window_size)

            self.buffer_start_time = self.current_time - (
                len(self.current_audio_buffer) / self.sample_rate
            )

            self._prefix_tokens = self._get_prefix_tokens()

            new_tokens: List[Dict[str, Any]] = self._run_transcription(
                self.current_audio_buffer,
                prefix_text=self._prefix_tokens,
            )
            self._update_agreement(new_tokens)
            not_committed_words = (
                self.local_agreement.not_committed_words
                if self.no_speech_streak == 0
                else []
            )
            commited_words = self.local_agreement.get_last_commited_words()

            return (commited_words, not_committed_words)

        else:
            if self.current_audio_buffer is None:
                self.current_audio_buffer = chunk
            else:
                self.current_audio_buffer = np.concatenate(
                    [self.current_audio_buffer, chunk]
                )
            return [], []

    def _trim_audio_buffer(self, max_seconds: float = 10.0) -> None:
        """
        Trim the audio buffer to the configured window size.
        """
        self.current_audio_buffer = self.current_audio_buffer[
            -int(max_seconds * self.sample_rate) :
        ]

    def _get_prefix_tokens(self) -> List[Dict[str, Any]]:
        """
        Get tokens from previous transcriptions that are still relevant to the current buffer.
        """
        prefix_tokens: List[Dict[str, Any]] = []
        for token in self.local_agreement.get_committed_words()[::-1]:
            if token["start"] >= self.buffer_start_time:
                prefix_tokens.append(token)
            else:
                break
        self._prefix_tokens = prefix_tokens[::-1]
        return self._prefix_tokens

    def _update_agreement(self, new_tokens: List[Dict[str, Any]]) -> None:
        """
        Update the local agreement with new transcription tokens.
        """
        self.local_agreement.add_transcription(new_tokens)

    def _run_transcription(
        self,
        audio: np.ndarray,
        prefix_text: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Delegate to the backend and prepend prefix tokens so
        LocalAgreement always sees (prefix + new) tokens.
        """
        new_only_tokens = self.backend.transcribe(
            audio=audio,
            prefix_text=prefix_text,
            buffer_start_time=self.buffer_start_time,
            sample_rate=self.sample_rate,
        )
        return self._prefix_tokens + new_only_tokens

    def clear(self) -> None:
        """
        Reset the pipeline to its initial state.
        """
        self.current_audio_buffer = None
        self.buffer_start_time = 0.0
        self._prefix_tokens = []
        self.current_time = 0.0
        self.audio_queue = []
        self.no_speech_streak = 0
        self.speech_threshold = 0.5
        self.local_agreement.clear()
