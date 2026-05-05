"""
Modular training script: shared `model` / `checkpoint_utils` / `train_common` (packed data).
The original self-contained script remains `gpt_improved.py`.
"""
import json
import os
import resource
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from flax import nnx
from tqdm import tqdm

from checkpoint_utils import save_model_weights, save_training_checkpoint
from model import MiniGPT, generate_text
from train_common import (
    FastDataLoader,
    apply_gradients,
    evaluate_batches,
    get_gpt2_tokenizer,
    make_eval_step,
    make_train_step,
    pack_texts_to_sequences,
    prepare_batch,
)
try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None

# ==========================================
# 0. CRITICAL PERFORMANCE SETTINGS
# ==========================================
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_fast_min_max=true --xla_gpu_deterministic_ops=false"
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

cache_dir = Path.home() / ".cache" / "jax"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"

print(f"JAX compilation cache: {cache_dir}")

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))

USE_BFLOAT16 = os.environ.get("USE_BFLOAT16", "").lower() in ("1", "true", "yes")
if USE_BFLOAT16:
    jax.config.update("jax_default_matmul_precision", "bfloat16")
    print("USE_BFLOAT16: jax_default_matmul_precision=bfloat16")

# ==========================================
# 1. HYPERPARAMETERS
# ==========================================
MAX_LEN = 512
EMBED_DIM = 768
NUM_HEADS = 12
FF_DIM = 3072
NUM_TRANSFORMER_BLOCKS = 16

BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 4

NUM_EPOCHS = 3
WARMUP_STEPS = 500
TOTAL_STEPS = 20000

LOG_EVERY = 50
EVAL_EVERY = 200
SAMPLE_EVERY = 500
SAVE_EVERY = 2000

PAD_TOKEN = 50256

RNG_SEED = int(os.environ.get("TRAIN_SEED", "42"))
np.random.seed(RNG_SEED)
ENABLE_WANDB = os.environ.get("ENABLE_WANDB", "").lower() in ("1", "true", "yes")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "jax-llm-expts")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")

# Early stopping on validation BPB (lower is better). 0 = disabled.
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "0"))
EARLY_STOP_MIN_DELTA = float(os.environ.get("EARLY_STOP_MIN_DELTA", "0.0"))

# ==========================================
# 2. FAST DATA LOADING (packed sequences)
# ==========================================
tokenizer = get_gpt2_tokenizer()
vocab_size = tokenizer.n_vocab

print("Loading dataset into memory...")
ds_train = load_dataset("roneneldan/TinyStories", split="train[:500000]")
ds_val = load_dataset("roneneldan/TinyStories", split="validation[:10000]")

train_texts = [s["text"] for s in ds_train]
val_texts = [s["text"] for s in ds_val]

print("Packing training sequences (concat + chunk)...")
train_tokens = pack_texts_to_sequences(train_texts, tokenizer, MAX_LEN, PAD_TOKEN)
print(f"Train data shape: {train_tokens.shape}")

print("Packing validation sequences...")
val_tokens = pack_texts_to_sequences(val_texts, tokenizer, MAX_LEN, PAD_TOKEN)
print(f"Val data shape: {val_tokens.shape}")

train_step = make_train_step(PAD_TOKEN)
eval_step = make_eval_step(PAD_TOKEN)


def evaluate_on_validation(model, val_data, num_batches=50):
    return evaluate_batches(
        model,
        val_data,
        BATCH_SIZE,
        PAD_TOKEN,
        eval_step,
        max_batches=num_batches,
    )


# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    print(f"\nJAX backend: {jax.devices()[0].platform}")
    print(f"JAX device: {jax.devices()[0]}")

    print("\nInitializing model...")
    rngs = nnx.Rngs(RNG_SEED)
    model = MiniGPT(
        MAX_LEN,
        vocab_size,
        EMBED_DIM,
        NUM_HEADS,
        FF_DIM,
        NUM_TRANSFORMER_BLOCKS,
        rngs=rngs,
    )

    total_params = sum(
        x.size for x in jax.tree_util.tree_leaves(nnx.state(model))
    )
    print(f"Model: {total_params:,} parameters ({total_params / 1e6:.2f}M)")

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=6e-4,
        warmup_steps=WARMUP_STEPS,
        decay_steps=TOTAL_STEPS,
        end_value=1e-5,
    )

    optimizer = nnx.ModelAndOptimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_schedule, weight_decay=0.1),
        ),
    )

    train_loader = FastDataLoader(train_tokens, BATCH_SIZE, shuffle=True)

    print("\n" + "=" * 80)
    print("Warming up JIT compilation (this takes ~30 seconds)...")
    print("=" * 80)
    warmup_batch = train_tokens[:BATCH_SIZE]
    warmup_inputs, warmup_targets = prepare_batch(warmup_batch, PAD_TOKEN)

    _ = train_step(model, optimizer, warmup_inputs, warmup_targets)
    _ = eval_step(model, warmup_inputs, warmup_targets)
    print("✓ JIT compilation complete!\n")

    run_id = f"run_{int(time.time())}"
    run_dir = Path.cwd() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if ENABLE_WANDB:
        if wandb is None:
            print("ENABLE_WANDB is set but wandb is not installed. Skipping telemetry.")
        else:
            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=run_id,
                dir=str(run_dir),
                config={
                    "max_len": MAX_LEN,
                    "embed_dim": EMBED_DIM,
                    "num_heads": NUM_HEADS,
                    "ff_dim": FF_DIM,
                    "num_transformer_blocks": NUM_TRANSFORMER_BLOCKS,
                    "batch_size": BATCH_SIZE,
                    "grad_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                    "num_epochs": NUM_EPOCHS,
                    "warmup_steps": WARMUP_STEPS,
                    "total_steps": TOTAL_STEPS,
                    "pad_token": PAD_TOKEN,
                    "seed": RNG_SEED,
                    "packed_sequences": True,
                    "early_stop_patience": EARLY_STOP_PATIENCE,
                    "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
                },
            )

    metrics_history = {
        "train_loss": [],
        "train_bpb": [],
        "val_loss": [],
        "val_bpb": [],
        "learning_rate": [],
        "tokens_per_sec": [],
        "samples": [],
        "steps": [],
    }

    print(f"{'=' * 80}")
    print("Training configuration (modular / packed):")
    print(f"  Model: {total_params / 1e6:.2f}M params")
    print(f"  Batch: {BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Tokens/step: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * MAX_LEN:,}")
    print(f"  Packed seqs train: {len(train_tokens):,}")
    print(f"  Total steps: {TOTAL_STEPS:,}")
    if EARLY_STOP_PATIENCE > 0:
        print(
            f"  Early stopping: patience={EARLY_STOP_PATIENCE} evals, "
            f"min_delta={EARLY_STOP_MIN_DELTA} (val BPB)"
        )
    print(f"  Run: {run_dir}")
    print(f"{'=' * 80}\n")

    best_val_bpb = float("inf")
    best_step: int | None = None
    evals_without_improve = 0
    stop_training = False

    step = 0
    micro_step = 0
    accumulated_grads = None

    running_loss = 0.0
    running_bpb = 0.0
    running_count = 0

    print("Starting training...\n")

    step_start = time.time()
    with tqdm(total=TOTAL_STEPS, desc="Training", ncols=100) as pbar:
        for _epoch in range(NUM_EPOCHS):
            if stop_training:
                break
            for batch in train_loader:
                if step >= TOTAL_STEPS or stop_training:
                    break

                step_start = time.time()

                inputs, targets = prepare_batch(batch, PAD_TOKEN)
                grads, loss, bpb = train_step(model, optimizer, inputs, targets)

                running_loss += float(loss)
                running_bpb += float(bpb)
                running_count += 1

                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = jax.tree.map(
                        lambda x, y: x + y, accumulated_grads, grads
                    )

                micro_step += 1

                if micro_step % GRADIENT_ACCUMULATION_STEPS == 0:
                    averaged_grads = jax.tree.map(
                        lambda x: x / GRADIENT_ACCUMULATION_STEPS,
                        accumulated_grads,
                    )
                    apply_gradients(optimizer, averaged_grads)
                    accumulated_grads = None

                    step_time = time.time() - step_start
                    tokens_per_sec = (
                        BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * MAX_LEN
                    ) / step_time

                    if (step + 1) % LOG_EVERY == 0:
                        current_lr = lr_schedule(step)
                        avg_loss = running_loss / running_count
                        avg_bpb = running_bpb / running_count

                        metrics_history["train_loss"].append(avg_loss)
                        metrics_history["train_bpb"].append(avg_bpb)
                        metrics_history["learning_rate"].append(float(current_lr))
                        metrics_history["tokens_per_sec"].append(float(tokens_per_sec))
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "train/loss": avg_loss,
                                    "train/bpb": avg_bpb,
                                    "train/learning_rate": float(current_lr),
                                    "train/tokens_per_sec": float(tokens_per_sec),
                                    "step": step + 1,
                                },
                                step=step + 1,
                            )

                        running_loss = 0.0
                        running_bpb = 0.0
                        running_count = 0

                        pbar.set_postfix(
                            {
                                "loss": f"{avg_loss:.3f}",
                                "tok/s": f"{tokens_per_sec:,.0f}",
                            }
                        )

                    if (step + 1) % EVAL_EVERY == 0:
                        val_loss, val_bpb = evaluate_on_validation(
                            model, val_tokens, num_batches=50
                        )
                        metrics_history["val_loss"].append(val_loss)
                        metrics_history["val_bpb"].append(val_bpb)
                        metrics_history["steps"].append(step + 1)
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "val/loss": val_loss,
                                    "val/bpb": val_bpb,
                                    "step": step + 1,
                                },
                                step=step + 1,
                            )

                        print(
                            f"\nStep {step + 1} | Train: {metrics_history['train_loss'][-1]:.4f} | Val: {val_loss:.4f}"
                        )

                        if EARLY_STOP_PATIENCE > 0:
                            if val_bpb < best_val_bpb - EARLY_STOP_MIN_DELTA:
                                best_val_bpb = val_bpb
                                best_step = step + 1
                                evals_without_improve = 0
                                save_model_weights(
                                    run_dir / "best_model.orbax", model, force=True
                                )
                                print(
                                    f"  New best val BPB {best_val_bpb:.6f} at step {best_step} → saved best_model.orbax"
                                )
                                if wandb_run is not None:
                                    wandb_run.log(
                                        {
                                            "early_stop/best_val_bpb": best_val_bpb,
                                            "early_stop/best_step": best_step,
                                            "step": step + 1,
                                        },
                                        step=step + 1,
                                    )
                            else:
                                evals_without_improve += 1
                                print(
                                    f"  No val BPB improvement ({evals_without_improve}/{EARLY_STOP_PATIENCE})"
                                )
                                if evals_without_improve >= EARLY_STOP_PATIENCE:
                                    print(
                                        "\nEarly stopping: val BPB did not improve enough "
                                        f"for {EARLY_STOP_PATIENCE} consecutive evaluations."
                                    )
                                    stop_training = True
                                    metrics_history["early_stopped"] = True
                                    if wandb_run is not None:
                                        wandb_run.log(
                                            {
                                                "early_stop/triggered": 1,
                                                "early_stop/best_val_bpb": best_val_bpb,
                                                "early_stop/best_step": best_step
                                                or 0,
                                                "step": step + 1,
                                            },
                                            step=step + 1,
                                        )

                    if (step + 1) % SAMPLE_EVERY == 0:
                        print(f"\n{'=' * 80}")
                        print(f"Sample Generation at Step {step + 1}")
                        print(f"{'=' * 80}")

                        prompts = [
                            "Once upon a time",
                            "The little girl",
                            "In the forest",
                        ]

                        for prompt in prompts:
                            sample = generate_text(
                                model,
                                tokenizer,
                                prompt,
                                max_tokens=120,
                                pad_token=PAD_TOKEN,
                            )
                            metrics_history["samples"].append(
                                {
                                    "step": step + 1,
                                    "prompt": prompt,
                                    "text": sample,
                                }
                            )
                            if wandb_run is not None:
                                wandb_run.log(
                                    {
                                        f"samples/{prompt}": wandb.Html(
                                            f"<pre>{sample}</pre>"
                                        ),
                                        "step": step + 1,
                                    },
                                    step=step + 1,
                                )
                            print(f"\nPrompt: '{prompt}'")
                            print(f"Output: {sample}")

                        print(f"\n{'=' * 80}\n")

                    if (step + 1) % SAVE_EVERY == 0:
                        checkpoint_path = run_dir / f"checkpoint_{step + 1}.orbax"
                        save_training_checkpoint(
                            checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            step=step + 1,
                        )
                        print(f"✓ Checkpoint saved at step {step + 1}")

                        with open(run_dir / "metrics.json", "w") as f:
                            json.dump(metrics_history, f, indent=2)

                    step += 1
                    pbar.update(1)
                    if stop_training:
                        break

    total_time = time.time() - step_start
    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print(f"{'=' * 80}")
    print(f"  Total Time: {total_time / 60:.2f} min ({total_time / 3600:.2f} hours)")
    print(f"  Total Steps: {step}")
    if metrics_history["train_loss"]:
        print(f"  Final Train Loss: {metrics_history['train_loss'][-1]:.6f}")
        print(f"  Final Train BPB: {metrics_history['train_bpb'][-1]:.6f}")
    if metrics_history["val_loss"]:
        print(f"  Final Val Loss: {metrics_history['val_loss'][-1]:.6f}")
        print(f"  Final Val BPB: {metrics_history['val_bpb'][-1]:.6f}")
    if metrics_history["tokens_per_sec"]:
        avg_throughput = sum(metrics_history["tokens_per_sec"]) / len(
            metrics_history["tokens_per_sec"]
        )
        print(f"  Avg Throughput: {avg_throughput:,.0f} tokens/sec")
    print(f"  Checkpoints: {run_dir}")
    if best_step is not None:
        print(
            f"  Best val BPB: {best_val_bpb:.6f} at step {best_step} (see best_model.orbax)"
        )
    if metrics_history.get("early_stopped"):
        print(
            "  Note: in-memory weights are from the last step; use best_model.orbax for the best val checkpoint."
        )
    print(f"{'=' * 80}\n")

    print("Saving final model...")
    final_path = run_dir / "final_model.orbax"
    save_model_weights(final_path, model)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    config = {
        "maxlen": MAX_LEN,
        "vocab_size": vocab_size,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "ff_dim": FF_DIM,
        "num_transformer_blocks": NUM_TRANSFORMER_BLOCKS,
        "pad_token": PAD_TOKEN,
        "total_params": total_params,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "packed_sequences": True,
        "train_seed": RNG_SEED,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "best_val_bpb": best_val_bpb if best_step is not None else None,
        "best_step": best_step,
    }
    with open(run_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Model saved to: {final_path}")
    print(f"✓ Config saved to: {run_dir / 'model_config.json'}")
    if wandb_run is not None:
        artifact = wandb.Artifact(name=f"{run_id}-artifacts", type="model")
        artifact.add_file(str(run_dir / "model_config.json"))
        artifact.add_file(str(run_dir / "metrics.json"))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    return model, metrics_history, run_dir


if __name__ == "__main__":
    model, metrics, run_dir = train()

    print("\n" + "=" * 80)
    print("Testing Final Model")
    print("=" * 80)

    test_prompts = [
        "Once upon a time in a magical forest,",
        "The brave knight",
        "A small puppy named Max",
        "On a sunny day,",
    ]

    tok = get_gpt2_tokenizer()
    for prompt in test_prompts:
        output = generate_text(model, tok, prompt, max_tokens=150, pad_token=PAD_TOKEN)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {output}\n")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("Training Summary:")
    print("=" * 80)
    if metrics["train_loss"]:
        print(
            f"Loss improvement: {metrics['train_loss'][0]:.4f} → {metrics['train_loss'][-1]:.4f}"
        )
        print(
            f"BPB improvement: {metrics['train_bpb'][0]:.4f} → {metrics['train_bpb'][-1]:.4f}"
        )
    if metrics["val_loss"]:
        print(
            f"Val loss: {metrics['val_loss'][0]:.4f} → {metrics['val_loss'][-1]:.4f}"
        )
    if metrics["tokens_per_sec"]:
        avg_tok_per_sec = sum(metrics["tokens_per_sec"]) / len(
            metrics["tokens_per_sec"]
        )
        print(f"Avg throughput: {avg_tok_per_sec:,.0f} tokens/sec")
    print(f"Samples generated: {len(metrics['samples'])}")
    print("=" * 80)
