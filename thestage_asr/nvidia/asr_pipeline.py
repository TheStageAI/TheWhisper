from io import BytesIO
from typing import (
    Any,
    BinaryIO,
    Dict,
    List,
    Tuple,
    Union,
    Generator,
    Iterator,
    Optional,
)

import numpy as np
import torch
import torchaudio
from transformers import (
    WhisperForConditionalGeneration as HFWhisperForConditionalGeneration,
)
from transformers import (
    WhisperProcessor,
)


WHISPER_SAMPLE_RATE = 16000


def patch_hf_model(model, chunk_length_s):
    max_source_positions = int(1500 * (chunk_length_s / 30))
    model.config.max_source_positions = max_source_positions
    pos_embed = model.model.encoder.embed_positions.weight
    model.model.encoder.embed_positions.weight.data = pos_embed[-max_source_positions:]


class BatchedASRPipeline:
    def __init__(
        self,
        model: Union[str, HFWhisperForConditionalGeneration],
        model_size: str = None,
        chunk_length_s: int = 30,
        processor: Optional[WhisperProcessor] = None,
        device: str = "cuda",
        token: Optional[str] = None,
    ) -> None:
        """Initialize the ASR pipeline.

        Args:
            model: HF model id (e.g. "thestage/thewhisper-large-v3-trubo") or model instance
            model_size: Model size to use ('S' or 'M' or 'L')
            chunk_length_s: Chunk length in seconds
            processor: Optional Whisper processor. If not provided and model is str, it will be loaded
            device: Device to run inference on ('cpu' or 'cuda')
            token: Optional Hugging Face token for private models
        """
        self.device = device

        if isinstance(model, str):
            # Load model and processor from Hugging Face Hub
            self.processor = WhisperProcessor.from_pretrained(model, token=token)
            if model_size is not None:
                from elastic_models.transformers import WhisperForConditionalGeneration

                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model,
                    mode=model_size,
                    chunk_length=chunk_length_s,
                    torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
                    token=token,
                )
            else:
                self.model = HFWhisperForConditionalGeneration.from_pretrained(
                    model,
                    torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
                    token=token,
                )
        else:
            if processor is None:
                raise ValueError("processor must be provided when passing a model instance")
            self.model = model
            self.processor = processor

        self.model.to(self.device)
        self.model.eval()

        if isinstance(self.model, HFWhisperForConditionalGeneration):
            patch_hf_model(self.model, chunk_length_s)

    def _to_numpy_1d(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
        if audio_np.ndim == 2:
            # (channels, samples) or (samples, channels) -> make mono
            if audio_np.shape[0] < audio_np.shape[1]:
                audio_np = audio_np.mean(axis=0)
            else:
                audio_np = audio_np.mean(axis=1)
        audio_np = audio_np.astype(np.float32)
        # Normalize if looks like int16 range
        if audio_np.dtype.kind in {"i", "u"}:
            audio_np = audio_np.astype(np.float32) / 32768.0
        return audio_np

    def _load_audio_path(self, path: str) -> Tuple[np.ndarray, int]:
        try:
            wav, sr = torchaudio.load(path)
            wav_np = self._to_numpy_1d(wav)
            return wav_np, int(sr)
        except Exception:
            # Fallback to librosa if torchaudio fails
            import librosa

            wav_np, sr = librosa.load(path, sr=None, mono=False)
            wav_np = self._to_numpy_1d(wav_np)
            return wav_np, int(sr)

    def _resample_to_16k(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == WHISPER_SAMPLE_RATE:
            return audio
        # Use torchaudio resample for speed/consistency
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=WHISPER_SAMPLE_RATE)
        with torch.no_grad():
            resampled = resampler(waveform)
        return resampled.squeeze(0).cpu().numpy()

    def _prepare_inputs(
        self,
        audio: Union[str, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, torch.Tensor]]],
    ) -> List[np.ndarray]:
        if isinstance(audio, list):
            items = audio
        else:
            items = [audio]

        prepared: List[np.ndarray] = []
        for item in items:
            if isinstance(item, str):
                wav, sr = self._load_audio_path(item)
            elif isinstance(item, (np.ndarray, torch.Tensor)):
                wav = self._to_numpy_1d(item)
                sr = WHISPER_SAMPLE_RATE  # assume caller gives 16k; will still resample below if needed
            else:
                raise TypeError("audio must be a path, numpy array, tensor, or list of those")

            wav = self._resample_to_16k(wav, sr)
            prepared.append(wav)
        return prepared

    def chunk_generator_all(
        self, inputs: List[np.ndarray], chunk_length_samples: int
    ) -> Generator[Tuple[int, int, np.ndarray], None, None]:
        """Generate chunks from all input audio samples.

        Args:
            inputs: List of audio arrays
            chunk_length_samples: Length of each chunk in samples

        Yields:
            Tuple of (sample_id, chunk_id, audio_chunk)
        """
        for sample_id, audio in enumerate(inputs):
            num_chunks = (len(audio) + chunk_length_samples - 1) // chunk_length_samples
            for chunk_id in range(num_chunks):
                start_idx = chunk_id * chunk_length_samples
                end_idx = min((chunk_id + 1) * chunk_length_samples, len(audio))
                audio_chunk = audio[start_idx:end_idx]
                audio_chunk = self.pad_audio_chunk(audio_chunk, chunk_length_samples)
                yield (sample_id, chunk_id, audio_chunk)

    def pad_audio_chunk(
        self, audio_chunk: np.ndarray, target_length: int
    ) -> np.ndarray:
        """Pad or truncate audio chunk to target length.

        Args:
            audio_chunk: Audio chunk to process
            target_length: Target length in samples

        Returns:
            Processed audio chunk of target length
        """
        current_length = len(audio_chunk)
        if current_length < target_length:
            padding = np.zeros(target_length - current_length, dtype=audio_chunk.dtype)
            audio_chunk = np.concatenate((audio_chunk, padding))
        elif current_length > target_length:
            audio_chunk = audio_chunk[:target_length]
        return audio_chunk

    def batch_iterator(
        self,
        generator: Generator[Tuple[int, int, np.ndarray], None, None],
        batch_size: int,
    ) -> Iterator[List[Tuple[int, int, np.ndarray]]]:
        """Create batches from a generator.

        Args:
            generator: Generator yielding individual items
            batch_size: Size of each batch

        Yields:
            Lists of items forming batches
        """
        from itertools import islice

        iterator = iter(generator)
        for first in iterator:
            batch = [first] + list(islice(iterator, batch_size - 1))
            yield batch

    def generate(
        self,
        inputs: List[np.ndarray],
        lang_ids: List[str],
        batch_size: int,
        chunk_length_s: float = 30.0,
        return_chunks: bool = False,
        **generate_kwargs: Any,
    ) -> Union[List[str], Tuple[List[str], Dict[int, List[Tuple[int, str]]]]]:
        """Generate transcriptions for input audio samples.

        Args:
            inputs: List of audio arrays to transcribe
            lang_ids: List of language IDs for each input
            batch_size: Batch size for processing
            chunk_length_s: Length of each chunk in seconds
            **generate_kwargs: Additional arguments for generation

        Returns:
            List of transcription strings
        """
        chunks: Dict[int, List[Tuple[int, str]]] = {
            sample_id: [] for sample_id in range(len(inputs))
        }
        full_transcriptions: List[str] = []

        chunk_length_samples = int(chunk_length_s * WHISPER_SAMPLE_RATE)

        all_chunks_generator = self.chunk_generator_all(inputs, chunk_length_samples)
        batch_iter = self.batch_iterator(all_chunks_generator, batch_size)

        for batch in batch_iter:
            sample_ids = [sample_id for sample_id, chunk_id, audio_chunk in batch]
            chunk_ids = [chunk_id for sample_id, chunk_id, audio_chunk in batch]
            audio_chunks = [audio_chunk for sample_id, chunk_id, audio_chunk in batch]
            batch_lang_ids = [lang_ids[sample_id] for sample_id in sample_ids]

            padded_inputs = self.processor.feature_extractor(
                audio_chunks,
                sampling_rate=WHISPER_SAMPLE_RATE,
                return_tensors="pt",
                padding=False,  # No need to pad here since audio chunks are already padded
            )

            input_features = padded_inputs.input_features.to(self.device)
            if hasattr(self.model, "dtype"):
                input_features = input_features.to(self.model.dtype)

            with torch.inference_mode():
                predicted_ids = self.model.generate(
                    input_features,
                    language=batch_lang_ids,
                    task="transcribe",
                    **generate_kwargs,
                )

            transcription_chunks = self.processor.tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            for sample_id, chunk_id, transcription in zip(
                sample_ids, chunk_ids, transcription_chunks
            ):
                chunks[sample_id].append((chunk_id, transcription))

        for sample_id in range(len(inputs)):
            chunks[sample_id].sort(key=lambda x: x[0])

            full_transcription = " ".join(
                [transcription for chunk_id, transcription in chunks[sample_id]]
            )

            full_transcriptions.append(full_transcription)

        if return_chunks:
            return full_transcriptions, chunks
        return full_transcriptions

    def __call__(
        self,
        audio: Union[str, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, torch.Tensor]]],
        max_batch_size: int = 1,
        return_timestamps: Union[bool, str] = False,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        chunk_length_s: float = 30.0,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Run transcription.

        Args:
            audio: Single audio path/array/tensor or a list of them
            max_batch_size: Max batch size for chunked processing
            return_timestamps: False, True, or "segment"; only "segment" is supported
            generate_kwargs: Additional generation arguments passed to model.generate
            chunk_length_s: Chunk length in seconds for segmentation
            **kwargs: Additional keyword arguments including lang_ids

        Returns:
            Dict for single input, or list of dicts for batch. Includes "text" and optionally "segments".
        """
        if generate_kwargs is None:
            generate_kwargs = {}

        prepared_inputs = self._prepare_inputs(audio)

        lang_ids = kwargs.get("lang_ids", ["en"] * len(prepared_inputs))

        want_segments = isinstance(return_timestamps, str) and return_timestamps == "segment"
        if want_segments:
            texts, chunk_map = self.generate(
                inputs=prepared_inputs,
                lang_ids=lang_ids,
                batch_size=max_batch_size,
                chunk_length_s=chunk_length_s,
                return_chunks=True,
                **generate_kwargs,
            )
        else:
            texts = self.generate(
                inputs=prepared_inputs,
                lang_ids=lang_ids,
                batch_size=max_batch_size,
                chunk_length_s=chunk_length_s,
                **generate_kwargs,
            )
            chunk_map = None

        results: List[Dict[str, Any]] = []
        for idx, text in enumerate(texts):
            item: Dict[str, Any] = {"text": text}
            if want_segments and chunk_map is not None:
                segments: List[Dict[str, Any]] = []
                for chunk_id, seg_text in chunk_map[idx]:
                    start = float(chunk_id * chunk_length_s)
                    end = float((chunk_id + 1) * chunk_length_s)
                    segments.append({"id": chunk_id, "start": start, "end": end, "text": seg_text})
                item["segments"] = segments
            results.append(item)

        if isinstance(audio, list):
            return results
        return results[0]
