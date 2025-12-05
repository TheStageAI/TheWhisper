import torch
import os
import numpy as np
from time import time
from typing import List, Dict, Optional, Union, Any
from transformers import SequenceFeatureExtractor, PreTrainedTokenizer
from transformers.utils import logging as hf_logging
import zlib

from .local_agreement import LocalAgreement
from ..vad import batched_vad

import logging

logger = logging.getLogger(__name__)
hf_logging.set_verbosity_error()
LOG_LEVEL = os.getenv('LOG_LEVEL', 'info')

torch.set_grad_enabled(False)


def _compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


class StreamingPipeline:
    """
    """
    
    def __init__(
        self,
        model: str,
        model_size: str = 'S',
        chunk_length_s: int = 10,
        use_vad: bool = False,
        agreement_history_size: int = 5,
        agreement_majority_threshold: int = 2,
        platform: str = 'apple',
        torch_dtype: torch.dtype = torch.float16,
        language: str = 'en',
        feature_extractor: Optional[SequenceFeatureExtractor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """
        """

        if platform == 'apple':
            from ..apple import ASRPipeline
            device = 'cpu'
        elif platform == 'nvidia':
            from ..nvidia import ASRPipeline
            device = 'cuda'
        else:
            raise ValueError(f"Invalid platform: {platform}")
        
        self.asr_pipeline = ASRPipeline(
            model, 
            model_size=model_size, 
            chunk_length_s=chunk_length_s, 
            torch_dtype=torch_dtype,
            device=device,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
        self.device = device

        self.chunk_length_s: float = chunk_length_s
        self.window_size: float = chunk_length_s - 2
        self.sample_rate: int = 16000
        self.language: str = language
        
        special_tokens: str = f"<|startoftranscript|><|{language}|><|transcribe|>"
        self.encoded_special_tokens: torch.Tensor = self.asr_pipeline.tokenizer(
            special_tokens, return_tensors="pt", add_special_tokens=False
        ).input_ids
        
        self.no_speech_streak: int = 0
        self.speech_threshold: float = 0.5

        if use_vad:
            self.vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', model='silero_vad'
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
        """
        """
        self.add_new_chunk(chunk)
        return self.process_new_chunk()
    
    def add_new_chunk(self, chunk: np.ndarray) -> None:
        """
        Add a new audio chunk to the processing queue.
        
        Args:
            chunk (np.ndarray): Audio chunk as numpy array
        """
        # Check license validity before processing
        self.audio_queue.append(chunk)

    def process_new_chunk(self) -> List[Dict[str, Any]]:
        """
        Process all queued audio chunks.
        
        Returns:
            List[Dict[str, Any]]: List of newly committed words with their timestamps
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

        # Require a minimum amount of audio before decoding
        # if len(self.current_audio_buffer) / self.sample_rate <= 3.0:
        #     return [], []

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

    def _run_transcription(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run the ASR model on the audio buffer.
        
        Args:
            audio (np.ndarray): Audio buffer to transcribe
            
        Returns:
            List[Dict[str, Any]]: List of transcribed tokens with timestamps
        """
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
                'start': token['timestamp'][0] + self.buffer_start_time,
                'end': token['timestamp'][1] + self.buffer_start_time
            })
        
        return generated_tokens

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
