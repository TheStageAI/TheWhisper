import json
from pathlib import Path

import time
import numpy as np
import torch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import coremltools as ct

import pickle
import shutil

from huggingface_hub import snapshot_download
from transformers import (
    AutoProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig,
    GenerationConfig,
)
from transformers.models.whisper.generation_whisper import (
    _dynamic_time_warping,
    _median_filter,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    StaticCache,
)

from .mlx_modules import Whisper as MLXWhisper, TextDecoder as MLXTextDecoder
from .mlx_modules import ModelDimensions

from .quantization_utils import quantize_mlx_model


@dataclass
class AttributeContainer:
    stride: Tuple[int]


class BaseEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.conv1 = AttributeContainer(stride=(1, 1))
        self.conv2 = AttributeContainer(stride=(2, 2))
        self.total_time_worked = 0

    def convert_inputs(self, input_features, device):
        if isinstance(input_features, torch.Tensor):
            input_features = input_features.detach().cpu().numpy()
        return input_features

    def convert_outputs(self, hidden_states, device):
        if isinstance(hidden_states, np.ndarray):
            hidden_states = torch.from_numpy(hidden_states).to(device)
        return hidden_states

    def __call__(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        start_time = time.time()
        device = (
            input_features.device
            if isinstance(input_features, torch.Tensor)
            else input_features.device
        )
        input_features = self.convert_inputs(input_features, device)
        hidden_states = self.forward(input_features)
        hidden_states = self.convert_outputs(hidden_states, device)
        self.total_time_worked += time.time() - start_time
        if not return_dict:
            return tuple(v for v in [hidden_states, None, None] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )

    def forward(self, mel_features):
        raise NotImplementedError("Implement this method in the subclass")


class ANEEncoder(BaseEncoder):
    def __init__(self, encoder_path: str):
        if encoder_path.endswith(".mlmodelc"):
            encoder = ct.models.CompiledMLModel(
                encoder_path, compute_units=ct.ComputeUnit.CPU_AND_NE
            )
        elif encoder_path.endswith(".mlpackage"):
            encoder = ct.models.MLModel(
                encoder_path, compute_units=ct.ComputeUnit.CPU_AND_NE
            )
        else:
            raise ValueError("Encoder format is not supported")

        super().__init__(encoder)

    def forward(self, mel_features):
        return self.encoder.predict({"logmel_data": mel_features})


class HFMLXEncoder(BaseEncoder):
    """Wrapper for MLX AudioEncoder to make it compatible with HF's WhisperEncoder interface"""

    def convert_inputs(self, input_features, device):
        input_features = input_features.transpose(1, 2)
        input_features = super().convert_inputs(input_features, device)
        input_features = mx.array(input_features, dtype=mx.float16)
        return input_features

    def convert_outputs(self, hidden_states, device):
        hidden_states = super().convert_outputs(hidden_states, device)
        hidden_states = torch.from_numpy(np.array(hidden_states, copy=False)).to(device)
        return hidden_states

    def forward(self, input_features):
        return self.encoder(input_features)


class HFMLXDecoder(torch.nn.Module):
    """Wrapper for MLX TextDecoder to make it compatible with HF's WhisperDecoder interface"""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        # Copy attributes from MLX decoder
        self.token_embedding = decoder.token_embedding
        self.positional_embedding = decoder.positional_embedding
        self.blocks = decoder.blocks
        self.ln = decoder.ln
        self._mask = decoder._mask
        self.total_time_worked = 0

    def _convert_hf_cache_to_mlx(self, past_key_values: EncoderDecoderCache):
        """Convert HuggingFace cache format to MLX format"""
        if past_key_values is None:
            return None
        # MLX expects a list of tuples for self attention and cross attention caches
        self_attn_cache = past_key_values.self_attention_cache
        cross_attn_cache = past_key_values.cross_attention_cache
        # Initialize MLX cache format
        mlx_cache = [None] * len(self.blocks)
        # Convert self attention cache if it exists
        if self_attn_cache is not None and len(self_attn_cache.key_cache) > 0:
            # Convert all keys and values in parallel
            keys = [
                mx.array(
                    self_attn_cache.key_cache[i].detach().numpy(), dtype=mx.float16
                )
                for i in range(len(self.blocks))
            ]
            values = [
                mx.array(
                    self_attn_cache.value_cache[i].detach().numpy(), dtype=mx.float16
                )
                for i in range(len(self.blocks))
            ]
            mlx_cache = [(k, v) for k, v in zip(keys, values)]

        # Convert cross attention cache if it exists
        if cross_attn_cache is not None and len(cross_attn_cache.key_cache) > 0:
            # Convert all keys and values in parallel
            keys = [
                mx.array(
                    cross_attn_cache.key_cache[i].detach().numpy(), dtype=mx.float16
                )
                for i in range(len(self.blocks))
            ]
            values = [
                mx.array(
                    cross_attn_cache.value_cache[i].detach().numpy(), dtype=mx.float16
                )
                for i in range(len(self.blocks))
            ]

            # Update mlx_cache with cross attention values
            for i in range(len(self.blocks)):
                if mlx_cache[i] is None:
                    mlx_cache[i] = (None, None)
                mlx_cache[i] = (mlx_cache[i], (keys[i], values[i]))

        if mlx_cache[0] is None:
            mlx_cache = None

        return mlx_cache

    def _convert_mlx_cache_to_hf(self, mlx_cache, device):
        """Convert MLX cache format to HuggingFace format"""
        if mlx_cache is None:
            return None
        # Initialize HF cache containers
        self_key_cache = {}
        self_value_cache = {}
        cross_key_cache = {}
        cross_value_cache = {}
        # Convert each layer's cache
        for i, layer_cache in enumerate(mlx_cache):
            if layer_cache is not None:
                # Handle self attention cache
                if isinstance(layer_cache[0], tuple):
                    k, v = layer_cache[0]
                    if k is not None:
                        self_key_cache[i] = torch.from_numpy(
                            np.array(k, copy=False)
                        )  # .to(device)
                        self_value_cache[i] = torch.from_numpy(
                            np.array(v, copy=False)
                        )  # .to(device)
                # Handle cross attention cache
                if len(layer_cache) > 1 and layer_cache[1] is not None:
                    k, v = layer_cache[1]
                    if k is not None:
                        cross_key_cache[i] = torch.from_numpy(
                            np.array(k, copy=False)
                        )  # .to(device)
                        cross_value_cache[i] = torch.from_numpy(
                            np.array(v, copy=False)
                        )  # .to(device)
        # Create HF cache format
        self_attn_cache = DynamicCache()
        cross_attn_cache = DynamicCache()
        # self_attn_cache = StaticCache(config=self.config, max_cache_len=1500)
        # cross_attn_cache = StaticCache(config=self.config, max_cache_len=32)
        for i in self_key_cache:
            self_attn_cache.update(self_key_cache[i], self_value_cache[i], i)
        for i in cross_key_cache:
            cross_attn_cache.update(cross_key_cache[i], cross_value_cache[i], i)
        # Create the encoder-decoder cache with proper is_updated flags
        cache = EncoderDecoderCache(self_attn_cache, cross_attn_cache)
        for i in range(len(mlx_cache)):
            if i in self_key_cache or i in cross_key_cache:
                cache.is_updated[i] = True

        return cache

    def _convert_attention_weights(self, qk, device):
        """Convert MLX attention weights to PyTorch format"""
        if not qk or all(layer_qk is None for layer_qk in qk):
            return None, None

        self_attentions = []
        cross_attentions = []

        for layer_qk in qk:
            if layer_qk is not None:
                # Handle both cross attention and self attention weights
                if isinstance(layer_qk, tuple):
                    cross_qk, self_qk = layer_qk
                else:
                    # For encoder layers that only have self attention
                    self_qk = layer_qk
                    cross_qk = None

                # Convert and process self attention weights
                if self_qk is not None:
                    self_qk_torch = torch.from_numpy(
                        np.array(self_qk, copy=False)
                    )  # .to(device)
                    # Don't apply softmax - HF expects raw attention logits
                    self_attentions.append(self_qk_torch)
                else:
                    self_attentions.append(None)

                # Convert and process cross attention weights
                if cross_qk is not None:
                    cross_qk_torch = torch.from_numpy(
                        np.array(cross_qk, copy=False)
                    )  # .to(device)
                    # Don't apply softmax - HF expects raw attention logits
                    cross_attentions.append(cross_qk_torch)
                else:
                    cross_attentions.append(None)
            else:
                self_attentions.append(None)
                cross_attentions.append(None)

        # Filter out None values and ensure proper shape for HF
        # self_attentions = [a for a in self_attentions if a is not None]
        # cross_attentions = [a for a in cross_attentions if a is not None]

        # HF expects attention weights in format [batch_size, num_heads, seq_len, seq_len]
        # Ensure our tensors match this format
        if self_attentions:
            self_attentions = tuple(self_attentions)
        else:
            self_attentions = None

        if cross_attentions:
            cross_attentions = tuple(cross_attentions)
        else:
            cross_attentions = None

        return self_attentions, cross_attentions

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        if input_ids.shape[-1] == 3:
            input_ids = torch.cat([input_ids, torch.tensor([[50364]])], dim=-1)
        
        # Convert torch tensors to MLX arrays
        device = (
            input_ids.device
            if isinstance(input_ids, torch.Tensor)
            else encoder_hidden_states.device
        )

        start_time = time.time()

        if isinstance(input_ids, torch.Tensor):
            input_ids = mx.array(input_ids.detach().cpu().numpy(), dtype=mx.int32)
        if isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = mx.array(
                encoder_hidden_states.detach().cpu().numpy(), dtype=mx.float16
            )
        elif isinstance(encoder_hidden_states, np.ndarray):
            encoder_hidden_states = mx.array(encoder_hidden_states, dtype=mx.float16)

        # Convert HF cache to MLX format if using cache
        mlx_cache = None
        if use_cache:
            mlx_cache = self._convert_hf_cache_to_mlx(past_key_values)
        # Forward pass through MLX decoder
        hidden_states, kv_cache, qk = self.decoder(
            input_ids, encoder_hidden_states, kv_cache=mlx_cache
        )
        # Convert MLX arrays to torch tensors
        if isinstance(hidden_states, mx.array):
            hidden_states = torch.from_numpy(
                np.array(hidden_states, copy=False)
            )  # .to(device)
        # Convert MLX cache to HF format if using cache
        if use_cache:
            kv_cache = self._convert_mlx_cache_to_hf(kv_cache, device)
        else:
            kv_cache = None
        # Convert attention weights if needed
        self_attentions, cross_attentions = None, None
        if output_attentions:
            self_attentions, cross_attentions = self._convert_attention_weights(
                qk, device
            )

        self.total_time_worked += time.time() - start_time

        if not return_dict:
            outputs = (hidden_states,)
            if use_cache:
                outputs += (kv_cache,)
            if output_hidden_states:
                outputs += (None,)  # hidden_states
            if output_attentions:
                outputs += (self_attentions, cross_attentions)
            return outputs

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=kv_cache if use_cache else None,
            hidden_states=None,
            attentions=self_attentions,
            cross_attentions=cross_attentions,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value


def load_mlx_decoder(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float16,
    max_source_positions: int = 1500,
) -> MLXWhisper:
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        dec_quant_config = config.pop("decoder_quant_config", None)
        config["max_source_positions"] = max_source_positions

    model_args = ModelDimensions(**config)

    wf = model_path / "weights.safetensors"
    if not wf.exists():
        wf = model_path / "weights.npz"

    weights = mx.load(str(wf))

    model = MLXTextDecoder(
        model_args.n_vocab,
        model_args.n_text_ctx,
        model_args.n_text_state,
        model_args.n_text_head,
        model_args.n_text_layer,
        dtype,
        max_source_positions=model_args.max_source_positions,
    )

    if dec_quant_config:
        model = quantize_mlx_model(model, quant_config=dec_quant_config)

    weights = tree_unflatten(list(weights.items()))

    model.update(weights)
    mx.eval(model.parameters())

    return model


class TheWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        chunk_length: int = 10,
        mode="S",
        *model_args,
        **kwargs,
    ):
        """Load TheWhisper with CoreML encoder and MLX decoder without loading Torch weights.

        Expected repo/path layout (flexible):
        - An encoder CoreML compiled model somewhere under the repo ("*.mlmodelc" or "*.mlpackage").
        - A decoder directory or file containing MLX weights and config (directory with
          "config.json" + ("weights.safetensors"|"weights.npz") or a ".pkl").

        Optional kwargs:
        - encoder_path: relative path under repo to the CoreML encoder (string)
        - decoder_path: relative path under repo to the MLX decoder directory or ".pkl" (string)
        - max_source_positions: int passed to load_mlx_decoder (default: 1500)
        - token, cache_dir, local_files_only, resume_download, proxies, force_download, et al.: forwarded to snapshot_download
        - torch_dtype, mode, device_map, low_cpu_mem_usage: accepted and ignored for compatibility
        """
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", False)
        resume_download = kwargs.pop("resume_download", None)
        force_download = kwargs.pop("force_download", None)
        proxies = kwargs.pop("proxies", None)
        revision = kwargs.pop("revision", None)
        # compatibility-only kwargs accepted by HF pipelines
        _ = kwargs.pop("torch_dtype", None)
        _ = kwargs.pop("device_map", None)
        _ = kwargs.pop("low_cpu_mem_usage", None)
        _ = kwargs.pop("mode", None)  # optional user size hint, not used here

        max_source_positions = int(1500 * (chunk_length / 30))

        enc_rel = kwargs.pop("encoder_path", None)
        dec_rel = kwargs.pop("decoder_path", None)

        base_path = Path(pretrained_model_name_or_path)
        if not base_path.exists():
            subfolder = f"free/macos_15_ios_18/{str(mode)}/{chunk_length}sec"
            snapshot_path = snapshot_download(
                repo_id=pretrained_model_name_or_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                resume_download=resume_download,
                force_download=force_download,
                proxies=proxies,
                revision=revision,
                allow_patterns=[f"{subfolder}/**"],
                ignore_patterns=None,
                token=token,
            )
            base_path = Path(snapshot_path)

        # Expected under: <root>/free/macos_15_ios_18/{mode}/{chunk_length}sec
        platform_mode_path = (
            base_path / "free" / "macos_15_ios_18" / str(mode) / f"{chunk_length}sec"
        )
        if not platform_mode_path.exists():
            raise FileNotFoundError(
                f"Expected path '{platform_mode_path}' to exist (derived from '{pretrained_model_name_or_path}/free/macos_15_ios_18/{mode}')."
            )
        base_path = platform_mode_path

        def _find_encoder_path(root: Path) -> Path:
            if enc_rel is not None:
                p = root / enc_rel
                if p.exists():
                    # If user provided a zip, unzip and return extracted .mlmodelc
                    if p.suffix == ".zip" or p.name.endswith(".mlmodelc.zip"):
                        return _unzip_mlmodelc(p)
                    return p
            # search common CoreML compiled formats
            for pattern in ("*.mlmodelc.zip", "*.mlmodelc", "*.mlpackage"):
                found = next(root.rglob(pattern), None)
                if found is not None:
                    if found.suffix == ".zip" or found.name.endswith(".mlmodelc.zip"):
                        return _unzip_mlmodelc(found)
                    return found
            raise FileNotFoundError(
                f"CoreML encoder not found under '{root}'. Provide encoder_path explicitly."
            )

        def _unzip_mlmodelc(zip_path: Path) -> Path:
            """Ensure a .mlmodelc.zip is extracted next to the archive and return the .mlmodelc dir."""
            target_dir = (
                zip_path.parent / zip_path.stem
            )  # removes only .zip, leaves *.mlmodelc
            if not target_dir.exists():
                # Extract into parent directory to preserve inside structure
                shutil.unpack_archive(str(zip_path), extract_dir=str(zip_path.parent))
            # Prefer the expected target dir; otherwise, fall back to first *.mlmodelc under parent
            if target_dir.exists():
                return target_dir
            fallback = next(zip_path.parent.glob("*.mlmodelc"), None)
            if fallback is not None:
                return fallback
            raise FileNotFoundError(
                f"Failed to locate extracted .mlmodelc directory for '{zip_path}'."
            )

        def _find_decoder_path(root: Path) -> Path:
            if dec_rel is not None:
                p = root / dec_rel
                if p.exists():
                    return p
            # Prefer a directory with config.json and weights
            for cfg in root.rglob("config.json"):
                parent = cfg.parent
                if (parent / "weights.safetensors").exists() or (
                    parent / "weights.npz"
                ).exists():
                    return parent
            # Fallback to a decoder .pkl if present
            for pkl in root.rglob("*.pkl"):
                if "decoder" in pkl.name.lower():
                    return pkl
            # As a last resort, if root looks like a decoder folder, use it
            if (root / "config.json").exists() and (
                (root / "weights.safetensors").exists()
                or (root / "weights.npz").exists()
            ):
                return root
            raise FileNotFoundError(
                f"MLX decoder assets not found under '{root}'. Provide decoder_path explicitly."
            )

        encoder_path = _find_encoder_path(base_path)
        decoder_path = _find_decoder_path(base_path)

        # Load config and optionally generation config
        config = WhisperConfig.from_pretrained(str(base_path))
        gen_config = GenerationConfig.from_pretrained(str(base_path))

        # Build a lightweight skeleton model from config (no HF weights download)
        model_skeleton_path = base_path / "hf_model.pkl"
        with open(str(model_skeleton_path), "rb") as f:
            model = pickle.load(f)

        if gen_config is not None:
            model.generation_config = gen_config

        # Swap encoder and decoder with our CoreML/MLX wrappers
        model.model.encoder = ANEEncoder(str(encoder_path))
        mlx_decoder = None
        if decoder_path.suffix == ".pkl":
            with open(str(decoder_path), "rb") as f:
                mlx_decoder = pickle.load(f)
        else:
            mlx_decoder = load_mlx_decoder(
                str(decoder_path), max_source_positions=max_source_positions
            )
        model.model.decoder = HFMLXDecoder(mlx_decoder)

        mlx_decoder.set_alignment_heads(model.generation_config.alignment_heads)
        model.device_param = torch.nn.Parameter(torch.ones(1, device="cpu"))
        model.proj_out = torch.nn.Identity()

        return model

    def _postprocess_outputs(
        self,
        seek_outputs,
        decoder_input_ids,
        return_token_timestamps,
        generation_config,
        is_shortform,
    ):
        # remove all previously passed decoder input ids
        # should happen only if it is the first generated segment
        start_idx = decoder_input_ids.shape[-1]

        if isinstance(seek_outputs, torch.Tensor):
            return seek_outputs[:, start_idx:], seek_outputs

        if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
            num_frames = getattr(generation_config, "num_frames", None)
            seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                seek_outputs,
                generation_config.alignment_heads,
                num_frames=num_frames,
                num_input_ids=decoder_input_ids.shape[-1],
            )

        def split_by_batch_index(
            values, key, batch_idx, is_shortform, beam_indices=None
        ):
            if beam_indices is not None and key == "scores":
                return [
                    v[beam_idx].cpu()
                    for (v, beam_idx) in zip(
                        values, beam_indices[batch_idx][: len(values)]
                    )
                ]
            if key in [
                "scores",
                "encoder_attentions",
                "encoder_hidden_states",
                "logits",
            ]:
                return [v[batch_idx].cpu() for v in values]
            if key in [
                "decoder_attentions",
                "decoder_hidden_states",
                "cross_attentions",
            ]:
                return tuple(
                    tuple(w[batch_idx][None].cpu() for w in v if w is not None)
                    for v in values
                    if v is not None
                )
            elif key == "past_key_values":
                if not is_shortform:
                    # we don't save `past_key_values` as this is too costly for longform
                    return None
                elif isinstance(values, EncoderDecoderCache):
                    all_past_key_values = []
                    for layer_idx in range(self.config.decoder_layers):
                        layer_past_key_values = []
                        for cache_cls in [
                            values.self_attention_cache,
                            values.cross_attention_cache,
                        ]:
                            for v in [cache_cls.key_cache, cache_cls.value_cache]:
                                layer_past_key_values.append(
                                    v[layer_idx][batch_idx][None].cpu()
                                )
                        all_past_key_values.append(tuple(layer_past_key_values))
                    return tuple(all_past_key_values)
                else:
                    all_past_key_values = []
                    for v in range(len(values)):
                        layer_past_key_values = []
                        for w in values[v]:
                            if len(w) != 0:
                                layer_past_key_values.append(w[batch_idx][None].cpu())
                            else:
                                layer_past_key_values.append(w)
                        all_past_key_values.append(tuple(layer_past_key_values))
                    return tuple(all_past_key_values)

            return values[batch_idx].cpu()

        sequence_tokens = seek_outputs["sequences"][:, start_idx:]
        seek_outputs = [
            {
                k: split_by_batch_index(
                    v, k, i, is_shortform, beam_indices=seek_outputs.get("beam_indices")
                )
                for k, v in seek_outputs.items()
            }
            for i in range(sequence_tokens.shape[0])
        ]

        return sequence_tokens, seek_outputs

    def _extract_token_timestamps(
        self,
        generate_outputs,
        alignment_heads,
        time_precision=0.02,
        num_frames=None,
        num_input_ids=None,
    ):
        """
        Calculates token-level timestamps using the encoder-decoder cross-attentions and dynamic time-warping (DTW) to
        map each output token to a position in the input audio. If `num_frames` is specified, the encoder-decoder
        cross-attentions will be cropped before applying DTW.

        Returns:
            tensor containing the timestamps in seconds for each predicted token
        """
        cross_attentions = []
        layer_indices = [head[0] for head in alignment_heads]
        head_indices = [head[1] for head in alignment_heads]

        for i in layer_indices:
            cross_attentions.append(
                torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2)
            )

        weights = torch.stack(
            [cross_attentions[i][:, 0] for i in range(len(alignment_heads))]
        )  # head_indices[i]

        weights = weights.permute([1, 0, 2, 3])

        weight_length = None

        if "beam_indices" in generate_outputs:
            # If beam search has been used, the output sequences may have been generated for more timesteps than their sequence_lengths
            # since the beam search strategy chooses the most probable sequences at the end of the search.
            # In that case, the cross_attentions weights are too long and we have to make sure that they have the right output_length
            weight_length = (generate_outputs.beam_indices != -1).sum(-1).max()
            weight_length = (
                weight_length
                if num_input_ids is None
                else weight_length + num_input_ids
            )

            # beam search takes `decoder_input_ids` into account in the `beam_indices` length
            # but forgot to shift the beam_indices by the number of `decoder_input_ids`
            beam_indices = torch.zeros_like(
                generate_outputs.beam_indices[:, :weight_length]
            )
            # we actually shif the beam indices here
            beam_indices[:, num_input_ids:] = generate_outputs.beam_indices[
                :, : weight_length - num_input_ids
            ]

            weights = weights[:, :, :weight_length]

            # If beam index is still -1, it means that the associated token id is EOS
            # We need to replace the index with 0 since index_select gives an error if any of the indexes is -1.
            beam_indices = beam_indices.masked_fill(beam_indices == -1, 0)

            # Select the cross attention from the right beam for each output sequences
            weights = torch.stack(
                [
                    torch.index_select(
                        weights[:, :, i, :], dim=0, index=beam_indices[:, i]
                    )
                    for i in range(beam_indices.shape[1])
                ],
                dim=2,
            )

        # make sure timestamps are as long as weights
        input_length = weight_length or cross_attentions[0].shape[2]
        batch_size = generate_outputs.sequences.shape[0]
        timestamps = torch.zeros(
            (batch_size, input_length + 1),
            dtype=torch.float32,
            device=generate_outputs.sequences.device,
        )

        if num_frames is not None:
            # two cases:
            # 1. num_frames is the same for each sample -> compute the DTW matrix for each sample in parallel
            # 2. num_frames is different, compute the DTW matrix for each sample sequentially

            # we're using np.unique because num_frames can be int/list/tuple
            if isinstance(num_frames, int):
                weights = weights[..., : num_frames // 2]

            elif (
                isinstance(num_frames, (list, tuple, np.ndarray))
                and len(np.unique(num_frames)) == 1
            ):
                weights = weights[..., : num_frames[0] // 2]

            elif (
                isinstance(num_frames, (torch.Tensor))
                and len(torch.unique(num_frames)) == 1
            ):
                weights = weights[..., : num_frames[0] // 2]

            else:
                # num_frames is of shape (batch_size,) whereas batch_size is truely batch_size*num_return_sequences
                repeat_time = (
                    batch_size
                    if isinstance(num_frames, int)
                    else batch_size // len(num_frames)
                )
                num_frames = (
                    num_frames.cpu()
                    if isinstance(num_frames, (torch.Tensor))
                    else num_frames
                )
                num_frames = np.repeat(num_frames, repeat_time)

        if num_frames is None or isinstance(num_frames, int):
            # Normalize and smoothen the weights.
            std = torch.std(weights, dim=-2, keepdim=True, unbiased=False)
            mean = torch.mean(weights, dim=-2, keepdim=True)
            weights = (weights - mean) / std
            weights = _median_filter(weights, self.config.median_filter_width)

            # Average the different cross-attention heads.
            weights = weights.mean(dim=1)

        # Perform dynamic time warping on each element of the batch.
        for batch_idx in range(batch_size):
            if num_frames is not None and isinstance(
                num_frames, (tuple, list, np.ndarray, torch.Tensor)
            ):
                matrix = weights[batch_idx, ..., : num_frames[batch_idx] // 2]

                # Normalize and smoothen the weights.
                std = torch.std(matrix, dim=-2, keepdim=True, unbiased=False)
                mean = torch.mean(matrix, dim=-2, keepdim=True)
                matrix = (matrix - mean) / std
                matrix = _median_filter(matrix, self.config.median_filter_width)

                # Average the different cross-attention heads.
                matrix = matrix.mean(dim=0)
            else:
                matrix = weights[batch_idx]

            text_indices, time_indices = _dynamic_time_warping(
                -matrix.cpu().double().numpy()
            )
            jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(
                bool
            )
            jump_times = time_indices[jumps] * time_precision
            timestamps[batch_idx, 1:] = torch.tensor(jump_times)

        return timestamps
