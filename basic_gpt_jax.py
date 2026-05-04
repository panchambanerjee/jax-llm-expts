
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import grain.python as grain
from datasets import load_dataset
import tiktoken
import time
from tqdm import tqdm

# ==========================================
# 1. HYPERPARAMETERS
# ==========================================
MAX_LEN = 128
EMBED_DIM = 192
NUM_HEADS = 6
FF_DIM = 512
NUM_TRANSFORMER_BLOCKS = 6
BATCH_SIZE = 32
NUM_EPOCHS = 1
WARMUP_STEPS = 50
TOTAL_STEPS = 1000

# ==========================================
# 2. DATA PREPARATION (TinyStories)
# ==========================================
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab

def get_tinystories_loader(max_samples=10000):
    print("Downloading/Loading TinyStories dataset...")
    # Load a tiny slice of the dataset for quick iteration
    ds = load_dataset("roneneldan/TinyStories", split=f"train[:{max_samples}]")
    stories = [story['text'] for story in ds]
    print(f"Loaded {len(stories)} stories.")
    
    class StoryDataset:
        def __init__(self, stories, maxlen, tokenizer):
            self.stories = stories
            self.maxlen = maxlen
            self.tokenizer = tokenizer
            self.end_token = 50256 # GPT-2 <|endoftext|>

        def __len__(self):
            return len(self.stories)

        def __getitem__(self, idx):
            story = self.stories[idx]
            # Encode and append the end-of-text token
            tokens = self.tokenizer.encode(story, allowed_special="all") + [self.end_token]

            if len(tokens) > self.maxlen:
                tokens = tokens[:self.maxlen]
                
            # Pad to maxlen
            tokens.extend([0] * (self.maxlen - len(tokens)))
            return tokens

    dataset = StoryDataset(stories, MAX_LEN, tokenizer)
    sampler = grain.IndexSampler(
        num_records=len(dataset),
        shuffle=True, 
        seed=42,
        shard_options=grain.NoSharding(),
        num_epochs=NUM_EPOCHS
    )
    dataloader = grain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[grain.Batch(batch_size=BATCH_SIZE, drop_remainder=True)]
    )
    return dataloader

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
        
        # --- [IMPROVEMENT SECTION 1: LAYER NORMALIZATION] ---
        # Uncomment below to stabilize deep network training (Standard GPT-2 feature)
        # self.ln1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        # self.ln2 = nnx.LayerNorm(embed_dim, rngs=rngs)

        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=embed_dim,
            qkv_features=embed_dim, out_features=embed_dim,
            decode=False, rngs=rngs
        )
        
        # --- [IMPROVEMENT SECTION 2: FEED-FORWARD NETWORK] ---
        # Uncomment below to add non-linear reasoning capacity to each block
        # self.ffn = nnx.Sequential(
        #     nnx.Linear(embed_dim, ff_dim, rngs=rngs),
        #     nnx.gelu,
        #     nnx.Linear(ff_dim, embed_dim, rngs=rngs)
        # )

    def __call__(self, x, mask=None):
        # Base implementation:
        attn_out = self.attention(x, mask=mask)
        x = x + attn_out
        
        # --- [IMPROVEMENT SECTION 3: APPLY FFN AND NORM] ---
        # If you uncommented Sections 1 & 2, REPLACE the two lines above with:
        # attn_out = self.attention(self.ln1(x), mask=mask)
        # x = x + attn_out
        # ff_out = self.ffn(self.ln2(x))
        # x = x + ff_out
        
        return x


class MiniGPT(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, num_heads,
                 feed_forward_dim, num_transformer_blocks, *, rngs):
        self.maxlen = maxlen
        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, rngs=rngs)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
            for _ in range(num_transformer_blocks)
        ]
        
        # --- [IMPROVEMENT SECTION 4: FINAL LAYER NORM] ---
        # Uncomment to stabilize the final logits prediction
        # self.final_ln = nnx.LayerNorm(embed_dim, rngs=rngs)
        
        self.output_layer = nnx.Linear(embed_dim, vocab_size, use_bias=False, rngs=rngs)
        
    def causal_attention_mask(self, seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)
        
        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
            
        # --- [IMPROVEMENT SECTION 5: APPLY FINAL NORM] ---
        # If you uncommented Section 4, add this line:
        # x = self.final_ln(x)

        logits = self.output_layer(x)
        return logits

# ==========================================
# 4. TRAINING LOGIC
# ==========================================
def loss_fn(model, batch):
    inputs, targets = batch
    logits = model(inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss, logits

@nnx.jit
def train_step(model, optimizer, metrics, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss) # simplified metric tracking
    optimizer.update(grads)

def train():
    # Initialize Model
    rngs = nnx.Rngs(0)
    model = MiniGPT(MAX_LEN, vocab_size, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_TRANSFORMER_BLOCKS, rngs=rngs)
    
    # Initialize Optimizer and LR Schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=3e-4, warmup_steps=WARMUP_STEPS, decay_steps=TOTAL_STEPS, end_value=1e-5
    )
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=lr_schedule, weight_decay=0.01))
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
    
    # Get DataLoader
    text_dl = get_tinystories_loader()
    
    # Helper to shift input tokens for the target label
    prep_target_batch = jax.vmap(lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0]))))
    metrics_history = {'train_loss': []}

    print("\nStarting Training...")
    step = 0
    with tqdm(total=TOTAL_STEPS, desc="Training") as pbar:
        for epoch in range(NUM_EPOCHS):
            for batch in text_dl:
                if step >= TOTAL_STEPS:
                    break
                    
                # Prepare inputs/targets
                input_batch = jnp.array(jnp.array(batch).T).astype(jnp.int32)
                target_batch = prep_target_batch(input_batch).astype(jnp.int32)
                
                # Execute JIT-compiled step
                train_step(model, optimizer, metrics, (input_batch, target_batch))
                
                # Logging
                if (step + 1) % 50 == 0:
                    current_loss = metrics.compute()['loss'].item()
                    metrics_history['train_loss'].append(current_loss)
                    metrics.reset()
                    
                    current_lr = lr_schedule(step)
                    pbar.set_postfix({'epoch': epoch + 1, 'loss': f"{current_loss:.4f}", 'lr': f"{current_lr:.2e}"})
                    
                step += 1
                pbar.update(1)

    print("Training Complete!")

if __name__ == "__main__":
    train()
