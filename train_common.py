"""Shared training utilities:loss, packing, batching, data loader."""

from __future__ import annotations

from typing import Iterable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import tiktoken
from flax import nnx
import optax


def pack_texts_to_sequences(
    texts: Iterable[str],
    tokenizer,
    maxlen: int,
    eos_token: int,
) -> np.ndarray:
    """Concatenate stories with EOS after each; chunk into (n, maxlen) non-overlapping rows."""
    buf: list[int] = []
    for text in texts:
        buf.extend(tokenizer.encode(text, allowed_special="all") + [eos_token])
    if len(buf) < maxlen:
        pad = eos_token
        buf.extend([pad] * (maxlen - len(buf)))
        return np.array([buf], dtype=np.int32)
    n = len(buf) // maxlen
    trimmed = buf[: n * maxlen]
    return np.array(trimmed, dtype=np.int32).reshape(n, maxlen)


class FastDataLoader:
    def __init__(
        self,
        data: np.ndarray,
        batch_size: int,
        *,
        shuffle: bool = True,
    ):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.num_batches = max(self.num_samples // batch_size, 0)

    def __iter__(self) -> Iterator[np.ndarray]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, self.num_samples - self.batch_size + 1, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield self.data[batch_indices]

    def __len__(self) -> int:
        return self.num_batches


def prepare_batch(batch: np.ndarray, pad_token: int) -> tuple[jax.Array, jax.Array]:
    input_batch = jnp.array(batch, dtype=jnp.int32)
    targets = jnp.concatenate(
        [
            input_batch[:, 1:],
            jnp.full((input_batch.shape[0], 1), pad_token, dtype=jnp.int32),
        ],
        axis=1,
    )
    return input_batch, targets


def make_train_step(pad_token: int):
    @nnx.jit
    def train_step(model, optimizer, inputs, targets):
        def _loss_fn(model):
            logits = model(inputs)
            mask = (targets != pad_token).astype(jnp.float32)
            per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, targets
            )
            masked_loss = per_token_loss * mask
            loss = masked_loss.sum() / (mask.sum() + 1e-8)
            bpb = loss / jnp.log(2.0)
            return loss, bpb

        grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
        (loss, bpb), grads = grad_fn(model)
        return grads, loss, bpb

    return train_step


def make_eval_step(pad_token: int):
    @nnx.jit
    def eval_step(model, inputs, targets):
        logits = model(inputs)
        mask = (targets != pad_token).astype(jnp.float32)
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, targets
        )
        masked_loss = per_token_loss * mask
        loss = masked_loss.sum() / (mask.sum() + 1e-8)
        bpb = loss / jnp.log(2.0)
        return loss, bpb

    return eval_step


@nnx.jit
def apply_gradients(optimizer, grads):
    optimizer.update(grads)


def evaluate_batches(
    model,
    val_data: np.ndarray,
    batch_size: int,
    pad_token: int,
    eval_step_fn,
    max_batches: int | None = None,
) -> tuple[float, float]:
    loader = FastDataLoader(val_data, batch_size, shuffle=False)
    losses, bpbs = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        inputs, targets = prepare_batch(batch, pad_token)
        loss, bpb = eval_step_fn(model, inputs, targets)
        losses.append(float(loss))
        bpbs.append(float(bpb))
    if not losses:
        return float("nan"), float("nan")
    return sum(losses) / len(losses), sum(bpbs) / len(bpbs)


def get_gpt2_tokenizer():
    return tiktoken.get_encoding("gpt2")
