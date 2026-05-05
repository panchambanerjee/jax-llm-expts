"""Orbax PyTree checkpoint save/restore with explicit sharding (CPU/GPU portable)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import SingleDeviceSharding

import orbax.checkpoint as ocp
from orbax.checkpoint import ArrayRestoreArgs


def _default_checkpointer() -> ocp.PyTreeCheckpointer:
    return ocp.PyTreeCheckpointer()


def target_sharding_for_device(device: jax.Device | None = None) -> SingleDeviceSharding:
    if device is None:
        device = jax.local_devices()[0]
    return SingleDeviceSharding(device)


def build_restore_args(template_item: Any, sharding: SingleDeviceSharding) -> Any:
    """Mirror Orbax docs: one ArrayRestoreArgs per JAX array leaf."""

    def leaf_fn(leaf: Any) -> Any:
        if isinstance(leaf, jax.Array):
            return ArrayRestoreArgs(sharding=sharding)
        return leaf

    return jax.tree.map(leaf_fn, template_item)


def save_model_weights(path: str | Path, model: nnx.Module, *, force: bool = True) -> None:
    path = Path(path)
    ckpt = _default_checkpointer()
    ckpt.save(path, nnx.state(model), force=force)


def restore_model_weights(
    path: str | Path,
    model: nnx.Module,
    *,
    device: jax.Device | None = None,
) -> None:
    """Load weights into an existing model instance (structure must match checkpoint)."""
    path = Path(path)
    ckpt = _default_checkpointer()
    template = nnx.state(model)
    sharding = target_sharding_for_device(device)
    restore_args = build_restore_args(template, sharding)
    restored = ckpt.restore(path, item=template, restore_args=restore_args)
    nnx.update(model, restored)


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nnx.Module,
    optimizer: nnx.ModelAndOptimizer,
    step: int,
    force: bool = True,
) -> None:
    """Save model + optimizer + step. Keep human-readable metrics in metrics.json only."""
    path = Path(path)
    ckpt = _default_checkpointer()
    ckpt.save(
        path,
        {
            "model": nnx.state(model),
            "optimizer": nnx.state(optimizer),
            "step": jnp.asarray(step, dtype=jnp.int32),
        },
        force=force,
    )


def restore_training_checkpoint(
    path: str | Path,
    *,
    model: nnx.Module,
    optimizer: nnx.ModelAndOptimizer,
    device: jax.Device | None = None,
) -> int:
    """Restore full training state (model + optimizer + step). Returns global step."""
    path = Path(path)
    ckpt = _default_checkpointer()
    template = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "step": jnp.asarray(0, dtype=jnp.int32),
    }
    sharding = target_sharding_for_device(device)

    def wrap_leaf(leaf: Any) -> Any:
        if isinstance(leaf, jax.Array):
            return ArrayRestoreArgs(sharding=sharding)
        return leaf

    restore_args = jax.tree.map(wrap_leaf, template)
    restored = ckpt.restore(path, item=template, restore_args=restore_args)
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["optimizer"])
    st = restored["step"]
    return int(st) if isinstance(st, jax.Array) else int(st)
