import base64
import gzip
import math
from dataclasses import dataclass
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int
    max_source_positions: int


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def __call__(
        self,
        x,
        xa=None,
        mask=None,
        kv_cache=None,
        fast_qkv_attention=False,
        alignment_heads=None,
    ):
        q = self.query(x)

        if xa is None:
            k = self.key(x)
            v = self.value(x)
            if kv_cache is not None:
                k = mx.concatenate([kv_cache[0], k], axis=1)
                v = mx.concatenate([kv_cache[1], v], axis=1)
        elif kv_cache is None:
            k = self.key(xa)
            v = self.value(xa)
        else:
            k, v = kv_cache

        if fast_qkv_attention:
            wv, qk = self.fast_qkv_attention(q, k, v, mask, alignment_heads)
        else:
            wv, qk = self.qkv_attention(q, k, v, mask, alignment_heads)
        
        return self.out(wv), (k, v), qk

    def qkv_attention(self, q, k, v, mask=None, alignment_heads=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        k = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 3, 1) * scale
        v = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = mx.softmax(qk, axis=-1, precise=True)
        out = (w @ v).transpose(0, 2, 1, 3)
        out = out.reshape(n_batch, n_ctx, n_state)

        if alignment_heads is not None:
            qk = qk[:, [*alignment_heads], :, :]
        else:
            qk = None

        return out, qk
    
    def fast_qkv_attention(self, q, k, v, mask=None, alignment_heads=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.reshape(*q.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        k = k.reshape(*k.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3) * scale
        v = v.reshape(*v.shape[:2], self.n_head, -1).transpose(0, 2, 1, 3)

        if mask is not None:
            mask = mask[:n_ctx, :n_ctx].astype(q.dtype)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1., mask=mask
        )
        out = out.transpose(0, 2, 1, 3)
        out = out.reshape(n_batch, n_ctx, n_state)
        qk = None

        if alignment_heads is not None:
            q = q[:, [*alignment_heads], :, :]
            k = k[:, [*alignment_heads], :, :].transpose(0, 1, 3, 2)
            qk = q @ k
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]

        return out, qk


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, max_source_positions: int = 1500):
        super().__init__()
        self.alignment_heads = None
        self.max_source_positions = max_source_positions

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp1 = nn.Linear(n_state, n_mlp)
        self.mlp2 = nn.Linear(n_mlp, n_state)
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(self, x, xa=None, mask=None, kv_cache=None):
        kv, cross_kv = kv_cache if kv_cache else (None, None)
        y, kv, self_qk = self.attn(
            self.attn_ln(x), mask=mask, 
            kv_cache=kv, 
            fast_qkv_attention=True
        )
        x += y
        cross_qk = None
        
        if self.cross_attn:
            xa = xa[:, :self.max_source_positions, :]
            y, cross_kv, cross_qk = self.cross_attn(
                self.cross_attn_ln(x), xa, 
                kv_cache=cross_kv, 
                fast_qkv_attention=True,
                alignment_heads=self.alignment_heads,
            )
            x += y
        
        x = x + self.mlp2(nn.gelu(self.mlp1(self.mlp_ln(x))))
        
        if self.cross_attn is None:
            return x, (kv, cross_kv), self_qk
        else:
            return x, (kv, cross_kv), (cross_qk, self_qk)


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
        max_source_positions: int = 1500,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self._positional_embedding = sinusoids(n_ctx, n_state).astype(dtype)
        self.blocks = [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        self.ln_post = nn.LayerNorm(n_state)
        self.size = max_source_positions

    def __call__(self, x):
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        x = x + self._positional_embedding[:x.shape[1]]

        if self.size is not None and self.size < 1500:
            x = x[:, :self.size, :]

        for i, block in enumerate(self.blocks):
            x, _, _ = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dtype: mx.Dtype = mx.float16,
        max_source_positions: int = 1500,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = mx.zeros((n_ctx, n_state))

        self.blocks = [
            ResidualAttentionBlock(n_state, n_head, cross_attention=True, max_source_positions=max_source_positions)
            for _ in range(n_layer)
        ]
        self.ln = nn.LayerNorm(n_state)
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(n_ctx).astype(dtype)

    def __call__(self, x, xa, kv_cache=None):
        """
        x : mx.array, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : mx.array, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = kv_cache[0][0][0].shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        
        qk = [None] * len(self.blocks)

        for e, block in enumerate(self.blocks):
            x, kv_cache[e], qk[e] = block(
                x, xa, mask=self._mask, kv_cache=kv_cache[e]
            )
        
        x = self.ln(x)
        x = self.token_embedding.as_linear(x)
        
        return x, kv_cache, qk

    def set_alignment_heads(self, alignment_heads):
        for layer_indx, head_indx in alignment_heads:
            if self.blocks[layer_indx].alignment_heads is None:
                self.blocks[layer_indx].alignment_heads = []
            self.blocks[layer_indx].alignment_heads.append(head_indx)


class Whisper(nn.Module):
    def __init__(
        self, 
        dims: ModelDimensions, 
        dtype: mx.Dtype = mx.float16, 
    ):
        super().__init__()
        self.dims = dims
        self.max_source_positions = self.dims.max_source_positions
        
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dtype,
            max_source_positions=self.dims.max_source_positions,
        )
        
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            dtype,
            max_source_positions=self.dims.max_source_positions,
        )

    def embed_audio(self, mel):
        return self.encoder(mel)

    def logits(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)[0]

    def forward_with_cross_qk(self, mel, tokens):
        logits, _, cross_qk = self.decoder(tokens, self.encoder(mel))
        return logits, cross_qk

    def __call__(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)
