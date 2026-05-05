import jax
import jax.numpy as jnp
from flax import nnx
import optax
from datasets import load_dataset
import tiktoken
import time
from tqdm import tqdm
import orbax.checkpoint as ocp
import json
from pathlib import Path
import os
import numpy as np

# ==========================================
# 0. CRITICAL PERFORMANCE SETTINGS
# ==========================================
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_fast_min_max=true --xla_gpu_deterministic_ops=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Enable persistent compilation cache
cache_dir = Path.home() / ".cache" / "jax"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['JAX_COMPILATION_CACHE_DIR'] = str(cache_dir)
os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'

print(f"JAX compilation cache: {cache_dir}")

# File descriptor fix
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))

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
SAMPLE_EVERY = 500  # RESTORED: Generate samples every 500 steps
SAVE_EVERY = 2000

PAD_TOKEN = 50256

EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "0"))
EARLY_STOP_MIN_DELTA = float(os.environ.get("EARLY_STOP_MIN_DELTA", "0.0"))

# ==========================================
# 2. FAST DATA LOADING
# ==========================================
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab

print("Loading dataset into memory...")
ds_train = load_dataset("roneneldan/TinyStories", split="train[:500000]")
ds_val = load_dataset("roneneldan/TinyStories", split="validation[:10000]")

def tokenize_story(story, maxlen=MAX_LEN):
    tokens = tokenizer.encode(story, allowed_special="all")
    tokens = tokens + [PAD_TOKEN]
    
    if len(tokens) > maxlen:
        tokens = tokens[:maxlen]
    else:
        tokens = tokens + [PAD_TOKEN] * (maxlen - len(tokens))
    
    return np.array(tokens, dtype=np.int32)

print("Pre-tokenizing training data...")
train_tokens = np.array([tokenize_story(s['text']) for s in tqdm(ds_train, desc="Tokenizing train")])
print(f"Train data shape: {train_tokens.shape}")

print("Pre-tokenizing validation data...")
val_tokens = np.array([tokenize_story(s['text']) for s in tqdm(ds_val, desc="Tokenizing val")])
print(f"Val data shape: {val_tokens.shape}")

class FastDataLoader:
    """Simple fast numpy-based data loader"""
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.num_batches = self.num_samples // batch_size
        
    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.num_samples - self.batch_size + 1, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.data[batch_indices]
    
    def __len__(self):
        return self.num_batches

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)

class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, *, rngs):
        self.ln1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=embed_dim,
            qkv_features=embed_dim, out_features=embed_dim,
            decode=False, rngs=rngs
        )
        self.ffn = nnx.Sequential(
            nnx.Linear(embed_dim, ff_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(ff_dim, embed_dim, rngs=rngs)
        )

    def __call__(self, x, mask=None):
        attn_out = self.attention(self.ln1(x), mask=mask)
        x = x + attn_out
        ff_out = self.ffn(self.ln2(x))
        x = x + ff_out
        return x

class MiniGPT(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, num_heads,
                 feed_forward_dim, num_transformer_blocks, *, rngs):
        self.maxlen = maxlen
        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, rngs=rngs)
        self.transformer_blocks = nnx.List([
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
            for _ in range(num_transformer_blocks)
        ])
        self.final_ln = nnx.LayerNorm(embed_dim, rngs=rngs)

    def causal_attention_mask(self, seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)
        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        x = self.final_ln(x)
        logits = x @ self.embedding.token_emb.embedding.T
        return logits

# ==========================================
# 4. TRAINING STEPS
# ==========================================
@nnx.jit
def train_step(model, optimizer, inputs, targets):
    """Single training step with masked loss"""
    def _loss_fn(model):
        logits = model(inputs)
        mask = (targets != PAD_TOKEN).astype(jnp.float32)
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        masked_loss = per_token_loss * mask
        loss = masked_loss.sum() / (mask.sum() + 1e-8)
        bpb = loss / jnp.log(2.0)
        return loss, bpb
    
    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
    (loss, bpb), grads = grad_fn(model)
    return grads, loss, bpb

@nnx.jit
def apply_gradients(optimizer, grads):
    """Apply gradients"""
    optimizer.update(grads)

@nnx.jit
def eval_step(model, inputs, targets):
    """Evaluation step"""
    logits = model(inputs)
    mask = (targets != PAD_TOKEN).astype(jnp.float32)
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    masked_loss = per_token_loss * mask
    loss = masked_loss.sum() / (mask.sum() + 1e-8)
    bpb = loss / jnp.log(2.0)
    return loss, bpb

# ==========================================
# 5. BATCH PREPARATION
# ==========================================
def prepare_batch(batch):
    """Convert numpy batch to JAX arrays"""
    input_batch = jnp.array(batch, dtype=jnp.int32)
    targets = jnp.concatenate([
        input_batch[:, 1:], 
        jnp.full((input_batch.shape[0], 1), PAD_TOKEN, dtype=jnp.int32)
    ], axis=1)
    return input_batch, targets

# ==========================================
# 6. VALIDATION
# ==========================================
def evaluate_on_validation(model, val_data, num_batches=50):
    """Fast validation"""
    val_loader = FastDataLoader(val_data, BATCH_SIZE, shuffle=False)
    val_losses = []
    val_bpbs = []
    
    for i, batch in enumerate(val_loader):
        if i >= num_batches:
            break
        inputs, targets = prepare_batch(batch)
        val_loss, val_bpb = eval_step(model, inputs, targets)
        val_losses.append(float(val_loss))
        val_bpbs.append(float(val_bpb))
    
    return sum(val_losses) / len(val_losses), sum(val_bpbs) / len(val_bpbs)

# ==========================================
# 7. TEXT GENERATION (RESTORED WITH REPETITION PENALTY)
# ==========================================
def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.9, 
                 top_k=50, repetition_penalty=1.2):
    """Generate text with repetition penalty"""
    tokens = tokenizer.encode(prompt)
    
    for _ in range(max_tokens):
        context = tokens[-MAX_LEN:]
        input_ids = jnp.array([context])
        logits = model(input_ids)[0, -1, :]
        
        # Apply repetition penalty
        for token in set(tokens[-20:]):
            logits = logits.at[token].set(logits[token] / repetition_penalty)
        
        logits = logits / temperature
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        probs = jax.nn.softmax(top_k_logits)
        
        next_idx = jax.random.categorical(
            jax.random.PRNGKey(len(tokens)), 
            jnp.log(probs)
        ).item()
        next_token = int(top_k_indices[next_idx])
        
        if next_token == PAD_TOKEN:
            break
        tokens.append(next_token)
    
    return tokenizer.decode(tokens)

# ==========================================
# 8. TRAINING LOOP
# ==========================================
def train():
    print(f"\nJAX backend: {jax.devices()[0].platform}")
    print(f"JAX device: {jax.devices()[0]}")
    
    # Initialize model
    print("\nInitializing model...")
    rngs = nnx.Rngs(0)
    model = MiniGPT(MAX_LEN, vocab_size, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_TRANSFORMER_BLOCKS, rngs=rngs)
    
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))
    print(f"Model: {total_params:,} parameters ({total_params/1e6:.2f}M)")
    
    # Optimizer
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, 
        peak_value=6e-4,
        warmup_steps=WARMUP_STEPS,
        decay_steps=TOTAL_STEPS, 
        end_value=1e-5
    )
    
    optimizer = nnx.ModelAndOptimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=0.1)
    ))
    
    # Data loaders
    train_loader = FastDataLoader(train_tokens, BATCH_SIZE, shuffle=True)
    
    # WARMUP JIT COMPILATION
    print("\n" + "="*80)
    print("Warming up JIT compilation (this takes ~30 seconds)...")
    print("="*80)
    warmup_batch = train_tokens[:BATCH_SIZE]
    warmup_inputs, warmup_targets = prepare_batch(warmup_batch)
    
    # Compile all functions
    _ = train_step(model, optimizer, warmup_inputs, warmup_targets)
    _ = eval_step(model, warmup_inputs, warmup_targets)
    print("✓ JIT compilation complete!\n")
    
    # Setup
    run_id = f"run_5090_{int(time.time())}"
    run_dir = Path.cwd() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_history = {
        'train_loss': [],
        'train_bpb': [],
        'val_loss': [],
        'val_bpb': [],
        'learning_rate': [],
        'tokens_per_sec': [],
        'samples': [],  # RESTORED: Track generated samples
        'steps': []
    }
    
    print(f"{'='*80}")
    print(f"RTX 5090 Training Configuration:")
    print(f"  Model: {total_params/1e6:.2f}M params")
    print(f"  Batch: {BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Tokens/step: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * MAX_LEN:,}")
    print(f"  Total steps: {TOTAL_STEPS:,}")
    if EARLY_STOP_PATIENCE > 0:
        print(
            f"  Early stopping: patience={EARLY_STOP_PATIENCE} evals, "
            f"min_delta={EARLY_STOP_MIN_DELTA} (val BPB)"
        )
    print(f"  Run: {run_dir}")
    print(f"{'='*80}\n")
    
    best_val_bpb = float('inf')
    best_step = None
    evals_without_improve = 0
    stop_training = False
    
    # Training
    step = 0
    micro_step = 0
    accumulated_grads = None
    
    running_loss = 0.0
    running_bpb = 0.0
    running_count = 0
    
    print("Starting training...\n")
    
    with tqdm(total=TOTAL_STEPS, desc="Training", ncols=100) as pbar:
        for epoch in range(NUM_EPOCHS):
            if stop_training:
                break
            for batch in train_loader:
                if step >= TOTAL_STEPS or stop_training:
                    break
                
                step_start = time.time()
                
                inputs, targets = prepare_batch(batch)
                grads, loss, bpb = train_step(model, optimizer, inputs, targets)
                
                running_loss += float(loss)
                running_bpb += float(bpb)
                running_count += 1
                
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = jax.tree.map(lambda x, y: x + y, accumulated_grads, grads)
                
                micro_step += 1
                
                if micro_step % GRADIENT_ACCUMULATION_STEPS == 0:
                    averaged_grads = jax.tree.map(
                        lambda x: x / GRADIENT_ACCUMULATION_STEPS, accumulated_grads
                    )
                    apply_gradients(optimizer, averaged_grads)
                    accumulated_grads = None
                    
                    step_time = time.time() - step_start
                    tokens_per_sec = (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * MAX_LEN) / step_time
                    
                    if (step + 1) % LOG_EVERY == 0:
                        current_lr = lr_schedule(step)
                        avg_loss = running_loss / running_count
                        avg_bpb = running_bpb / running_count
                        
                        metrics_history['train_loss'].append(avg_loss)
                        metrics_history['train_bpb'].append(avg_bpb)
                        metrics_history['learning_rate'].append(float(current_lr))
                        metrics_history['tokens_per_sec'].append(float(tokens_per_sec))
                        
                        running_loss = 0.0
                        running_bpb = 0.0
                        running_count = 0
                        
                        pbar.set_postfix({
                            'loss': f"{avg_loss:.3f}",
                            'tok/s': f"{tokens_per_sec:,.0f}"
                        })
                    
                    if (step + 1) % EVAL_EVERY == 0:
                        val_loss, val_bpb = evaluate_on_validation(model, val_tokens, num_batches=50)
                        metrics_history['val_loss'].append(val_loss)
                        metrics_history['val_bpb'].append(val_bpb)
                        metrics_history['steps'].append(step + 1)
                        
                        print(f"\nStep {step+1} | Train: {metrics_history['train_loss'][-1]:.4f} | Val: {val_loss:.4f}")

                        if EARLY_STOP_PATIENCE > 0:
                            if val_bpb < best_val_bpb - EARLY_STOP_MIN_DELTA:
                                best_val_bpb = val_bpb
                                best_step = step + 1
                                evals_without_improve = 0
                                chk = ocp.PyTreeCheckpointer()
                                chk.save(run_dir / "best_model.orbax", nnx.state(model), force=True)
                                print(
                                    f"  New best val BPB {best_val_bpb:.6f} at step {best_step} → saved best_model.orbax"
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
                                    metrics_history['early_stopped'] = True
                    
                    # RESTORED: Sample generation during training
                    if (step + 1) % SAMPLE_EVERY == 0:
                        print(f"\n{'='*80}")
                        print(f"Sample Generation at Step {step+1}")
                        print(f"{'='*80}")
                        
                        prompts = [
                            "Once upon a time",
                            "The little girl",
                            "In the forest"
                        ]
                        
                        for prompt in prompts:
                            sample = generate_text(model, tokenizer, prompt, max_tokens=120)
                            metrics_history['samples'].append({
                                'step': step + 1,
                                'prompt': prompt,
                                'text': sample
                            })
                            print(f"\nPrompt: '{prompt}'")
                            print(f"Output: {sample}")
                        
                        print(f"\n{'='*80}\n")
                    
                    if (step + 1) % SAVE_EVERY == 0:
                        checkpoint_path = run_dir / f"checkpoint_{step+1}.orbax"
                        checkpointer = ocp.PyTreeCheckpointer()
                        checkpointer.save(checkpoint_path, {
                            'model': nnx.state(model),
                            'optimizer': nnx.state(optimizer),
                            'step': step + 1,
                            'metrics': metrics_history
                        }, force=True)
                        print(f"✓ Checkpoint saved at step {step+1}")
                        
                        with open(run_dir / "metrics.json", "w") as f:
                            json.dump(metrics_history, f, indent=2)
                    
                    step += 1
                    pbar.update(1)
                    if stop_training:
                        break
    
    # Training complete
    total_time = time.time() - step_start
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"  Total Time: {total_time/60:.2f} min ({total_time/3600:.2f} hours)")
    print(f"  Total Steps: {step}")
    print(f"  Final Train Loss: {metrics_history['train_loss'][-1]:.6f}")
    print(f"  Final Train BPB: {metrics_history['train_bpb'][-1]:.6f}")
    if metrics_history['val_loss']:
        print(f"  Final Val Loss: {metrics_history['val_loss'][-1]:.6f}")
        print(f"  Final Val BPB: {metrics_history['val_bpb'][-1]:.6f}")
    avg_throughput = sum(metrics_history['tokens_per_sec']) / len(metrics_history['tokens_per_sec'])
    print(f"  Avg Throughput: {avg_throughput:,.0f} tokens/sec")
    print(f"  Checkpoints: {run_dir}")
    if best_step is not None:
        print(
            f"  Best val BPB: {best_val_bpb:.6f} at step {best_step} (see best_model.orbax)"
        )
    if metrics_history.get('early_stopped'):
        print(
            "  Note: in-memory weights are from the last step; use best_model.orbax for the best val checkpoint."
        )
    print(f"{'='*80}\n")
    
    # RESTORED: Save final model with config
    print("Saving final model...")
    final_path = run_dir / "final_model.orbax"
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(final_path, nnx.state(model), force=True)
    
    # RESTORED: Save model config
    config = {
        'maxlen': MAX_LEN, 
        'vocab_size': vocab_size, 
        'embed_dim': EMBED_DIM,
        'num_heads': NUM_HEADS, 
        'ff_dim': FF_DIM,
        'num_transformer_blocks': NUM_TRANSFORMER_BLOCKS,
        'pad_token': PAD_TOKEN,
        'total_params': total_params,
        'gpu': 'RTX_5090',
        'batch_size': BATCH_SIZE,
        'grad_accum_steps': GRADIENT_ACCUMULATION_STEPS,
        'effective_batch_size': BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'early_stop_min_delta': EARLY_STOP_MIN_DELTA,
        'best_val_bpb': best_val_bpb if best_step is not None else None,
        'best_step': best_step,
    }
    with open(run_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Model saved to: {final_path}")
    print(f"✓ Config saved to: {run_dir / 'model_config.json'}")
    
    return model, metrics_history, run_dir

# ==========================================
# 9. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    model, metrics, run_dir = train()
    
    # RESTORED: Final testing with multiple prompts
    print("\n" + "="*80)
    print("Testing Final Model")
    print("="*80)
    
    test_prompts = [
        "Once upon a time in a magical forest,",
        "The brave knight",
        "A small puppy named Max",
        "On a sunny day,"
    ]
    
    for prompt in test_prompts:
        output = generate_text(model, tokenizer, prompt, max_tokens=150)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {output}\n")
        print("-" * 80)
    
    print("\n" + "="*80)
    print("Training Summary:")
    print("="*80)
    print(f"Loss improvement: {metrics['train_loss'][0]:.4f} → {metrics['train_loss'][-1]:.4f}")
    print(f"BPB improvement: {metrics['train_bpb'][0]:.4f} → {metrics['train_bpb'][-1]:.4f}")
    if metrics['val_loss']:
        print(f"Val loss: {metrics['val_loss'][0]:.4f} → {metrics['val_loss'][-1]:.4f}")
    avg_tok_per_sec = sum(metrics['tokens_per_sec']) / len(metrics['tokens_per_sec'])
    print(f"Avg throughput: {avg_tok_per_sec:,.0f} tokens/sec")
    print(f"Samples generated: {len(metrics['samples'])}")
    print("="*80)