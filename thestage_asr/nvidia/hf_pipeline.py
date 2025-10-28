import torch
from typing import Union, Optional
from transformers import (
    AutomaticSpeechRecognitionPipeline, SequenceFeatureExtractor, PreTrainedTokenizer
)
from transformers import WhisperForConditionalGeneration as HFWhisperForConditionalGeneration
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer


class ASRPipeline(AutomaticSpeechRecognitionPipeline):
    def __init__(
        self,
        model: Union[str, HFWhisperForConditionalGeneration],
        feature_extractor: Optional[SequenceFeatureExtractor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_size: str = None,
        chunk_length_s: int = 30,
        device: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if type(model) is str:
            model_name = model
            
            if model_size is not None:
                from elastic_models.transformers import WhisperForConditionalGeneration
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_name, mode=model_size, torch_dtype=torch_dtype # TODO: add chunk length in elastic models ?
                )
            else:
                model = HFWhisperForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=torch_dtype
                )
            
            if feature_extractor is None:
                feature_extractor = WhisperFeatureExtractor.from_pretrained(
                    model_name, torch_dtype=torch_dtype, chunk_length=chunk_length_s
                )
            
            if tokenizer is None:
                tokenizer = WhisperTokenizer.from_pretrained(
                    model_name, torch_dtype=torch_dtype
                )
        else:
            if feature_extractor is None:
                raise ValueError("feature_extractor must be provided when passing a model instance")
            if tokenizer is None:
                raise ValueError("tokenizer must be provided when passing a model instance")

        super().__init__(
            model, 
            feature_extractor=feature_extractor, 
            tokenizer=tokenizer, device=device, 
            torch_dtype=torch_dtype, 
            **kwargs
        )
        if chunk_length_s < 30:
            self._set_chunk_length(chunk_length_s)

    def _set_chunk_length(self, chunk_length_s):
        max_source_positions = int(1500 * (chunk_length_s / 30))
        self.model.config.max_source_positions = max_source_positions
        pos_embed = self.model.model.encoder.embed_positions.weight
        self.model.model.encoder.embed_positions.weight.data = pos_embed[-max_source_positions:]
