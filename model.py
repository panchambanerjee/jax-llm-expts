"""MiniGPT (Flax NNX) — shared architecture, generation, and config helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from flax import nnx


class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.ln1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs,
        )
        self.ffn = nnx.Sequential(
            nnx.Linear(embed_dim, ff_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(ff_dim, embed_dim, rngs=rngs),
        )

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        attn_out = self.attention(self.ln1(x), mask=mask)
        x = x + attn_out
        ff_out = self.ffn(self.ln2(x))
        x = x + ff_out
        return x


class MiniGPT(nnx.Module):
    def __init__(
        self,
        maxlen: int,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        num_transformer_blocks: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.maxlen = maxlen
        self.embedding = TokenAndPositionEmbedding(
            maxlen, vocab_size, embed_dim, rngs=rngs
        )
        self.transformer_blocks = nnx.List(
            [
                TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
                for _ in range(num_transformer_blocks)
            ]
        )
        self.final_ln = nnx.LayerNorm(embed_dim, rngs=rngs)

    def causal_attention_mask(self, seq_len: int) -> jax.Array:
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids: jax.Array) -> jax.Array:
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)
        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        x = self.final_ln(x)
        logits = x @ self.embedding.token_emb.embedding.T
        return logits


def load_model_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def create_model_from_config(
    cfg: Mapping[str, Any],
    *,
    rngs: nnx.Rngs | None = None,
) -> MiniGPT:
    if rngs is None:
        rngs = nnx.Rngs(0)
    return MiniGPT(
        int(cfg["maxlen"]),
        int(cfg["vocab_size"]),
        int(cfg["embed_dim"]),
        int(cfg["num_heads"]),
        int(cfg["ff_dim"]),
        int(cfg["num_transformer_blocks"]),
        rngs=rngs,
    )


def generate_text(
    model: MiniGPT,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    pad_token: int = 50256,
    rng_key: jax.Array | None = None,
) -> str:
    maxlen = int(model.maxlen)
    tokens = tokenizer.encode(prompt)

    for i in range(max_tokens):
        context = tokens[-maxlen:]
        input_ids = jnp.array([context], dtype=jnp.int32)
        logits = model(input_ids)[0, -1, :]

        for tok in set(tokens[-20:]):
            logits = logits.at[tok].set(logits[tok] / repetition_penalty)

        logits = logits / temperature
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        probs = jax.nn.softmax(top_k_logits)

        key = rng_key if rng_key is not None else jax.random.PRNGKey(i)
        next_idx = jax.random.categorical(key, jnp.log(probs)).item()
        next_token = int(top_k_indices[next_idx])

        if next_token == pad_token:
            break
        tokens.append(next_token)

    return tokenizer.decode(tokens)
