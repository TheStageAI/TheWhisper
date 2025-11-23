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
        agreement_history_size: int = 2,
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
        self.window_size: float = 10
        self.sample_rate: int = 16000
        
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

        if self.current_audio_buffer is not None and len(self.current_audio_buffer) / self.sample_rate > 1:
            self.current_audio_buffer = np.concatenate([self.current_audio_buffer, chunk])
            
            if len(self.current_audio_buffer) > self.window_size * self.sample_rate:
                self._trim_audio_buffer(self.window_size)

            self.buffer_start_time = self.current_time - (len(self.current_audio_buffer) / self.sample_rate)

            self._prefix_tokens = self._get_prefix_tokens()

            new_tokens: List[Dict[str, Any]] = self._run_transcription(
                self.current_audio_buffer,
                prefix_text=self._prefix_tokens,
            )
            self._update_agreement(new_tokens)
            not_committed_words = self.local_agreement.not_committed_words if self.no_speech_streak == 0 else []
            commited_words = self.local_agreement.get_last_commited_words()

            return (commited_words, not_committed_words)
        
        else:
            if self.current_audio_buffer is None:
                self.current_audio_buffer = chunk
            else:
                self.current_audio_buffer = np.concatenate([self.current_audio_buffer, chunk])
            return [], []

    def _trim_audio_buffer(self, max_seconds: float = 10.0) -> None:
        """
        Trim the audio buffer to the configured window size.
        """
        self.current_audio_buffer = self.current_audio_buffer[-int(max_seconds * self.sample_rate):]

    def _get_prefix_tokens(self) -> List[Dict[str, Any]]:
        """
        Get tokens from previous transcriptions that are still relevant to the current buffer.
        
        Returns:
            List[Dict[str, Any]]: List of tokens to use as prefix for the next transcription
        """
        prefix_tokens: List[Dict[str, Any]] = []
        for token in self.local_agreement.get_committed_words()[::-1]:
            if token['start'] >= self.buffer_start_time:
                prefix_tokens.append(token)
            else:
                break
        self._prefix_tokens = prefix_tokens[::-1]
        return self._prefix_tokens

    def _update_agreement(self, new_tokens: List[Dict[str, Any]]) -> None:
        """
        Update the local agreement with new transcription tokens.
        
        Args:
            new_tokens (List[Dict[str, Any]]): New tokens from the latest transcription
        """
        self.local_agreement.add_transcription(new_tokens)

    def _run_transcription(self, audio: np.ndarray, prefix_text: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Run the ASR model on the audio buffer.
        
        Args:
            audio (np.ndarray): Audio buffer to transcribe
            prefix_text (Optional[List[Dict[str, Any]]]): Tokens to use as prefix for the transcription
            
        Returns:
            List[Dict[str, Any]]: List of transcribed tokens with timestamps
        """
        audio_duration: float = len(audio) / self.sample_rate
        
        if prefix_text is not None and len(prefix_text) > 0:
            time_since_last_prefix_word = self.current_time - prefix_text[-1]['end']
            rounded_time = int(time_since_last_prefix_word)
            max_new_tokens = 16 + rounded_time * 16
        else:
            max_new_tokens = 32

        generate_kwargs: Dict[str, Any] = {
            'use_cache': True,
            'num_beams': 1,
            'do_sample': False,
            'max_new_tokens': max_new_tokens,
        }
        
        if prefix_text is not None and len(prefix_text) > 0:
            prefix_text_str: str = ''.join([w['text'] for w in prefix_text])
            decoder_input_ids: torch.Tensor = self.asr_pipeline.tokenizer(
                prefix_text_str, return_tensors="pt", add_special_tokens=False
            ).input_ids
            generate_kwargs['decoder_input_ids'] = torch.cat(
                [self.encoded_special_tokens, decoder_input_ids], dim=1
            ).to(self.device)

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

        new_chunks_lst = []
        max_end_time = 0
        for token in result['chunks']:
            end_time = token['timestamp'][1]
            if end_time is not None:
                if end_time > max_end_time and end_time < 10:
                    max_end_time = end_time
                    new_chunks_lst.append(token)
                else:
                    pass
            else:
                new_chunks_lst.append(token)
        result['chunks'] = new_chunks_lst

        for token in result['chunks']:
            if token['timestamp'][1] is None:
                if audio_duration - token['timestamp'][0] < max_word_duration:
                    token['timestamp'] = (token['timestamp'][0], audio_duration)
                else:
                    token['timestamp'] = (token['timestamp'][0], token['timestamp'][0] + max_word_duration)
            generated_tokens.append({
                'text': token['text'],
                'start': token['timestamp'][0] + self.current_time - len(self.current_audio_buffer) / self.sample_rate,
                'end': token['timestamp'][1] + self.current_time - len(self.current_audio_buffer) / self.sample_rate
            })
        return self._prefix_tokens + generated_tokens

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
