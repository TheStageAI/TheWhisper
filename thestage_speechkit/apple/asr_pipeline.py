import torch
from typing import Union, Optional
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    SequenceFeatureExtractor,
    PreTrainedTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperConfig,
)

from .model import TheWhisperForConditionalGeneration


class ASRPipeline(AutomaticSpeechRecognitionPipeline):
    def __init__(
        self,
        model: Union[str, TheWhisperForConditionalGeneration],
        feature_extractor: Optional[SequenceFeatureExtractor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_size: str = "S",
        chunk_length_s: int = 30,
        device: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        revision = kwargs.pop('revision', 'main')
        
        if type(model) is str:
            model_name = model
            config = WhisperConfig.from_pretrained(model_name)
            model = TheWhisperForConditionalGeneration.from_pretrained(
                model_name,
                mode=model_size,
                chunk_length=chunk_length_s,
                torch_dtype=torch_dtype,
                revision=revision
            )
            processor = WhisperProcessor.from_pretrained(
                model_name, chunk_length=chunk_length_s
            )
            feature_extractor = processor.feature_extractor
            tokenizer = processor.tokenizer
        else:
            if feature_extractor is None:
                raise ValueError(
                    "feature_extractor must be provided when passing a model instance"
                )
            if tokenizer is None:
                raise ValueError(
                    "tokenizer must be provided when passing a model instance"
                )

        super().__init__(
            model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            device=device,
            chunk_length_s=chunk_length_s,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        self._set_chunk_length(chunk_length_s)

    def _set_chunk_length(self, chunk_length_s):
        max_source_positions = int(1500 * (chunk_length_s / 30))
        self.model.config.max_source_positions = max_source_positions
