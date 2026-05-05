# jax-llm-expts

Experiments in building small language models with **JAX**, **Flax NNX**, and **Optax**, trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

The basic architecture is taken from this course: [Build and Train an LLM with JAX](https://learn.deeplearning.ai/courses/build-and-train-an-llm-with-jax/information).

## Scripts

| File | Purpose |
|------|---------|
| [`basic_gpt_jax.py`](basic_gpt_jax.py) | Minimal training loop: tiny model, Grain dataloader, attention-only blocks (FFN and layer norm left commented as learning hooks). |
| [`gpt_improved.py`](gpt_improved.py) | Full small-GPT-style setup: larger model, pre-tokenized data, masked loss, validation, sampling, gradient accumulation, checkpoints, and weight-tied LM head. Optional early stopping. |
| [`gpt_improved_modular.py`](gpt_improved_modular.py) | Modular trainer using shared `model.py`, `checkpoint_utils.py`, and `train_common.py`; supports packed sequences, optional W&B telemetry, and optional early stopping. |
| [`eval_lm.py`](eval_lm.py) | Evaluation harness for saved checkpoints; reports CE/PPL/BPB on TinyStories + WikiText-2 and writes a JSON report. |

## What `basic_gpt_jax.py` does

1. **Data** — Loads the first 10k TinyStories training examples via Hugging Face `datasets`, encodes with **tiktoken** (GPT-2 vocabulary), appends `<|endoftext|>` (50256), truncates or **pads with `0`** to fixed length `MAX_LEN` (128).

2. **Loading** — Wraps stories in a small `StoryDataset` and batches with **Grain** (`IndexSampler` + `DataLoader`).

3. **Model (`MiniGPT`)** — Token + learned positional embeddings, a stack of `TransformerBlock`s, then a **separate** `Linear` to vocabulary logits.
   - Each block is intentionally **minimal**: multi-head self-attention and a **single residual add**. The comments show where to add **LayerNorm**, **FFN (GELU)**, and **final LayerNorm** before the head—those pieces are **not** enabled in the basic script.

4. **Training** — Next-token prediction: inputs are token sequences; targets are inputs shifted left (last position padded with `0`). **Loss** is mean cross-entropy over **all** positions (no mask excluding pad). **AdamW** + warmup cosine schedule, **1k steps**, one epoch, `nnx.jit` on `train_step`, simple `MultiMetric` loss logging every 50 steps.

5. **Scope** — No validation split, no text generation, no checkpointing—fast to read and run for experimentation.

## What `gpt_improved.py` does

1. **Environment** — Sets XLA flags (e.g. fast min/max, non-deterministic GPU ops where allowed), JAX compilation cache under `~/.cache/jax`, allocator options, and raises the file descriptor limit for heavy I/O.

2. **Data** — Loads **500k** train and **10k** validation stories, **pre-tokenizes** everything into NumPy arrays (50256 used as **both** EOS-style delimiter and **pad** to `MAX_LEN` 512). A small **`FastDataLoader`** shuffles and yields contiguous batches (no Grain).

3. **Model** — GPT-style stack: **pre-norm** LayerNorm, attention, residual; **FFN** (expand → GELU → project); **final LayerNorm**; logits via **weight tying** to the token embedding matrix (`x @ token_emb.embedding.T`), so there is no separate output `Linear`. Blocks are held in **`nnx.List`**.

4. **Training** — **Masked** cross-entropy: positions where `targets == PAD_TOKEN` are excluded from the loss (and **bits per byte** is reported). **Gradient accumulation** (4 micro-batches), **global gradient clip** (1.0), **AdamW** with higher peak LR / weight decay tuned for the larger run. **`train_step`** and **`apply_gradients`** are split JIT functions to support accumulation.

5. **Evaluation & generation** — Periodic validation on a fixed number of batches; **sampling** with temperature, **top-k**, and **repetition penalty** on recent tokens.

6. **Artifacts** — Run directory under `runs/<run_id>/`: Orbax checkpoints (model + optimizer + metrics), `metrics.json`, `model_config.json`, final `final_model.orbax`; prints throughput (tokens/sec) and parameter count.

7. **Main** — After training, runs extra **test prompts** and prints a short summary (train/val loss, BPB, throughput, sample count).

### The improved script trains in ~4 hours on an NVIDIA RTX 5090 1x 

## Basic → improved: what changed

| Area | Basic | Improved |
|------|--------|----------|
| **Transformer block** | Attention + residual only; FFN and LayerNorm commented out | Pre-norm LN, MHA, FFN (GELU), residuals—standard decoder block |
| **Output head** | Separate `Linear(embed_dim, vocab_size)` | Weight-tied to token embeddings |
| **Padding / loss** | Pad with `0`; loss on all positions | Pad with `PAD_TOKEN` (50256); **masked** loss (and BPB) |
| **Scale** | e.g. 128 × 192 × 6 layers, 6 heads, FF 512, batch 32, 10k stories | e.g. 512 × 768 × 16 layers, 12 heads, FF 3072, batch 64, 500k train + val |
| **Data pipeline** | Grain + on-the-fly tokenization in `__getitem__` | Pre-tokenize once; NumPy `FastDataLoader` |
| **Optimization** | AdamW only; effective batch = `BATCH_SIZE` | Grad accumulation ×4; **clip_by_global_norm(1.0)**; stronger weight decay |
| **Training loop** | Single stream, 1 epoch cap at `TOTAL_STEPS` | Multi-epoch; logging / eval / sample / save intervals |
| **Observability** | Train loss only | Val loss, BPB, tokens/sec, periodic samples |
| **Persistence** | None | Orbax checkpoints + JSON metrics + config |
| **Inference** | None | `generate_text` with top-k, temperature, repetition penalty |
| **Performance hygiene** | Default JAX | XLA flags, compilation cache, FD limit, JIT warmup |

The basic script is a **deliberately stripped** decoder (attention-only blocks) with hooks in comments to “upgrade” toward GPT-2-style blocks; the improved script implements that full block and production-oriented training around it.

## Model & training parameters

All of these are **module-level constants** in the scripts (edit the source to change them). Vocabulary size comes from **tiktoken**’s GPT-2 encoding (`n_vocab` ≈ 50,257).

### Shared architecture (both scripts)

| Concept | Meaning |
|--------|---------|
| **Causal self-attention** | Lower-triangular mask; `MultiHeadAttention` with `in_features = qkv_features = out_features = EMBED_DIM` (head size is `EMBED_DIM / NUM_HEADS`). |
| **Embeddings** | Learned token embedding table + learned **absolute** position embeddings up to `MAX_LEN`. |

### Parameter counts (default hyperparameters)

Totals are for **trainable weights** (Flax NNX state), using **`tiktoken` GPT-2 `n_vocab` = 50,257**. If `tiktoken` ever changes vocabulary size, these numbers move slightly.

| Script | Approx. | Exact (float32 weights) | Notes |
|--------|---------|-------------------------|--------|
| [`basic_gpt_jax.py`](basic_gpt_jax.py) | **~20.2M** | 20,212,608 | Token + position embeddings, **6** attention-only blocks (each `MultiHeadAttention` with default biases), **untied** `Linear` output head `EMBED_DIM → vocab_size`. |
| [`gpt_improved.py`](gpt_improved.py) | **~152.4M** | 152,398,080 | Same embeddings idea, **16** full transformer blocks (LayerNorm + MHA + FFN), **final** LayerNorm, logits **`@` token embedding** (no extra LM-head matrix). |

`gpt_improved.py` prints this at startup (`Model: … parameters`). For `basic_gpt_jax.py`, you can log the same quantity with:

`sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))`.

Uncommenting the FFN / extra norms in the basic script would **increase** the basic model’s parameter count beyond the table above.

### Parameter math (how the totals are derived)

This section uses **plain text and code blocks** so it renders in any Markdown preview (LaTeX-style `\(…\)` math is not used).

**Symbols**

| Symbol | Meaning |
|--------|---------|
| **V** | `n_vocab` from tiktoken GPT-2 (here **50,257**) |
| **D** | `EMBED_DIM` |
| **L** | `MAX_LEN` |
| **F** | `FF_DIM` (FFN hidden size; improved script only) |
| **N** | Number of transformer blocks |

**Embeddings (both models)** — token matrix **V·D**, position matrix **L·D**:

| | Basic (D=192, L=128) | Improved (D=768, L=512) |
|---|----------------------|-------------------------|
| Token | 50,257 × 192 = **9,649,344** | 50,257 × 768 = **38,597,376** |
| Position | 128 × 192 = **24,576** | 512 × 768 = **393,216** |
| **Sum** | **9,673,920** | **38,990,592** |

**One `nnx.MultiHeadAttention`** with `in_features = qkv_features = out_features = D` and default **`use_bias=True`**: three **D→D** projections (each weights **D²** and bias **D**) plus output **D→D** (**D²** + **D**):

```
MHA(D) = 4·D² + 4·D
```

| | D = 192 (basic) | D = 768 (improved) |
|---|-----------------|---------------------|
| MHA(D) | 4×192² + 4×192 = **148,224** | 4×768² + 4×768 = **2,362,368** |

Basic has **N = 6** attention-only blocks → 6 × 148,224 = **889,344**.

**Basic-only LM head** — `nnx.Linear(D, V, use_bias=False)` adds **D·V** weights:

```
192 × 50,257 = 9,649,344   (untied from token embeddings; same product as V·D but a separate matrix)
```

**Basic grand total**

```
embeddings + N·MHA(D) + LM_head
= 9,673,920 + 889,344 + 9,649,344
= 20,212,608
```

**Improved: one transformer block** (D=768, F=3072). Each `LayerNorm(D)` has **2D** params (scale + bias); two norms per block → **4D = 3,072**. FFN: `Linear(D,F)` is **D·F + F**; `Linear(F,D)` is **F·D + D**:

```
FFN(D, F) = 2·D·F + F + D
          = 2×768×3072 + 3072 + 768
          = 4,722,432
```

One block = 2 LNs + MHA + FFN:

```
4·D + (4·D² + 4·D) + (2·D·F + F + D)
= 3,072 + 2,362,368 + 4,722,432
= 7,087,872
```

**N = 16** blocks → 16 × 7,087,872 = **113,405,952**. Final `LayerNorm(D)` adds **2D = 1,536**. Logits use **weight tying** (`x @ token_emb.embedding.T`), so **no** extra **D·V** matrix.

**Improved grand total**

```
embeddings + N·(one block) + final_LN
= 38,990,592 + 113,405,952 + 1,536
= 152,398,080
```

If you change **V** (tokenizer), **D**, **L**, **F**, or **N**, plug the new values into the same expressions.

### `basic_gpt_jax.py`

| Parameter | Default | Role |
|-----------|---------|------|
| `MAX_LEN` | 128 | Context length; position table length. |
| `EMBED_DIM` | 192 | Model width; attention Q/K/V/O width. |
| `NUM_HEADS` | 6 | Attention heads (head dim 32). |
| `FF_DIM` | 512 | Passed into blocks for the **commented-out** FFN only; inactive in the default minimal block. |
| `NUM_TRANSFORMER_BLOCKS` | 6 | Depth (attention + residual per block). |
| `BATCH_SIZE` | 32 | Grain batch size. |
| `NUM_EPOCHS` | 1 | Sampler epochs (training stops at `TOTAL_STEPS` anyway). |
| `WARMUP_STEPS` | 50 | LR warmup length. |
| `TOTAL_STEPS` | 1000 | Optimization steps. |
| **LR schedule** | peak `3e-4`, end `1e-5` | `optax.warmup_cosine_decay_schedule` over `TOTAL_STEPS`. |
| **AdamW** | `weight_decay=0.01` | Default `nnx.Optimizer` + AdamW. |
| **Data** | `max_samples=10000` | `get_tinystories_loader`; first 10k train stories. |
| **Special tokens** | `50256` (`<|endoftext|>`), pad `0` | Story end marker; **padding uses `0`** (loss is not masked). |
| **Grain** | `shuffle=True`, `seed=42` | `IndexSampler` + batched `DataLoader`. |

### `gpt_improved.py`

| Parameter | Default | Role |
|-----------|---------|------|
| `MAX_LEN` | 512 | Context length. |
| `EMBED_DIM` | 768 | Width (GPT-2–small-scale). |
| `NUM_HEADS` | 12 | Head dim 64. |
| `FF_DIM` | 3072 | FFN hidden (4× embed). |
| `NUM_TRANSFORMER_BLOCKS` | 16 | Depth. |
| `BATCH_SIZE` | 64 | Sequences per micro-batch. |
| `GRADIENT_ACCUMULATION_STEPS` | 4 | Micro-batches per optimizer step → **effective batch 256** sequences. |
| `NUM_EPOCHS` | 3 | Outer epoch loop (still capped by `TOTAL_STEPS`). |
| `WARMUP_STEPS` | 500 | LR warmup. |
| `TOTAL_STEPS` | 20000 | Optimizer steps (after accumulation). |
| `LOG_EVERY` | 50 | Postfix / running metrics cadence. |
| `EVAL_EVERY` | 200 | Validation frequency (steps). |
| `SAMPLE_EVERY` | 500 | Text sample generation frequency. |
| `SAVE_EVERY` | 2000 | Orbax checkpoint + `metrics.json` dump cadence. |
| `PAD_TOKEN` | 50256 | Padding and EOS-style fill; **excluded from loss** via mask. |
| **LR schedule** | peak `6e-4`, end `1e-5` | Warmup + cosine decay over `TOTAL_STEPS`. |
| **Optimizer chain** | global norm **1.0**, AdamW `weight_decay=0.1` | `optax.chain(clip_by_global_norm, adamw)`. |
| **Data** | `train[:500000]`, `validation[:10000]` | Pre-tokenized NumPy; `FastDataLoader`. |
| **Validation** | 50 batches | `evaluate_on_validation(..., num_batches=50)`. |

**Generation (`generate_text`)** — defaults: `max_tokens=100`, `temperature=0.9`, `top_k=50`, `repetition_penalty=1.2` (penalizes logits for tokens in the last 20 generated tokens). The training loop uses `max_tokens=120` for scheduled samples.

**Environment (top of file)** — XLA flags (`XLA_FLAGS`), `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_ALLOCATOR`, `JAX_COMPILATION_CACHE_DIR`, `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS`, and `RLIMIT_NOFILE` bump; these affect performance and stability, not the mathematical model.

## Potential improvements

**Architecture**

- Turn on the **commented LayerNorm + FFN + final norm** path in `basic_gpt_jax.py`, or keep using `gpt_improved.py` as the full block reference.
- **RoPE** or **ALiBi** instead of learned absolute positions for length generalization.
- **RMSNorm**, **SwiGLU** (or other) FFN variants; **dropout** on residual, attention, or embeddings.
- **KV-cache** path for efficient autoregressive inference (today `generate_text` recomputes the full prefix each step).

**Training & optimization**

- **Mixed precision** (e.g. `jax` `bfloat16`) and/or **rematerialization** to train larger models on the same GPU.
- **Learning-rate schedule** experiments (WSD, inv-sqrt with decay, constant with cooldown), **AdamW** `β`/`ε` sweeps, **warmup tokens** instead of steps.
- **Fused or compiled** data pipeline (e.g. `grain` or `tf.data` → JAX) for the improved script; **packed sequences** (multiple documents per row) to reduce pad waste.
- **Multi-device** (`pjit` / `shard_map`) or multi-host training for throughput.

**Data & objective**

- Train on **more or cleaner text**; add **deduplication** and **quality filters**.
- **Masking** and **loss weighting** aligned with your padding strategy everywhere (the basic script currently trains on pad `0` positions).
- Auxiliary losses (e.g. **auxiliary heads**, **BPE** tokenizer trained on domain) if you move beyond GPT-2 bytes.

**Evaluation & inference**

- **Perplexity** on full validation, **BLEU** / model-based metrics only if you add references appropriate for open-ended story generation.
- Stronger decoding: **top-p (nucleus)**, **min-p**, **typical sampling**, fixed **PRNG keys** per step (the current sampler uses a key derived from `len(tokens)`, which is not ideal for statistics or reproducibility).
- **Stoppable criteria** (early stopping on validation BPB) instead of a fixed step count.

**Engineering**

- **CLI or YAML/JSON config** instead of editing globals; **deterministic seeds** for NumPy, Grain, and JAX together.
- **Experiment tracking** (Weights & Biases, TensorBoard) and **artifact versioning** beyond local `runs/`.
- **Unit tests** for `prepare_batch`, mask shapes, and checkpoint round-trips.

## Running

### CPU / generic install

From the repo root, install dependencies (pick a [JAX build](https://jax.readthedocs.io/en/latest/installation.html) for your platform first if the default CPU wheel is not what you want):

```bash
pip install -r requirements.txt
python basic_gpt_jax.py
python gpt_improved.py
python gpt_improved_modular.py
```

`gpt_improved.py` expects more RAM/GPU memory and disk (dataset cache, larger arrays, checkpoints).

### NVIDIA GPU (CUDA 12)

`gpt_improved.py` has been run on an **NVIDIA GeForce RTX 5090**. Install JAX with CUDA 12 wheels from Google’s release index, then the Python stack:

```bash
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax optax grain datasets tiktoken orbax-checkpoint tqdm matplotlib seaborn gradio transformers wandb
```

### Telemetry (Weights & Biases)

`gpt_improved_modular.py` supports optional W&B logging.

```bash
export ENABLE_WANDB=1
export WANDB_PROJECT=jax-llm-expts
# optional:
# export WANDB_ENTITY=your_team_or_username

python gpt_improved_modular.py
```

Logged metrics include train/val loss, BPB, learning rate, throughput, and generated samples.

### Early stopping

Both [`gpt_improved.py`](gpt_improved.py) and [`gpt_improved_modular.py`](gpt_improved_modular.py) support optional **early stopping** on validation **BPB** (bits per byte; lower is better). When validation improves, weights are written to `runs/<run_id>/best_model.orbax`. `final_model.orbax` is always the **last** step; use `best_model.orbax` for the best validation checkpoint when early stopping ran.

```bash
# Stop after 5 consecutive evaluations with no val BPB improvement
export EARLY_STOP_PATIENCE=5
# Optional: require improvement by at least this much (default 0)
export EARLY_STOP_MIN_DELTA=0.0
```

Set `EARLY_STOP_PATIENCE=0` (default) to disable.

### Evaluation harness

Run evaluation on a saved checkpoint + config:

```bash
python eval_lm.py \
  --checkpoint runs/<run_id>/final_model.orbax \
  --config runs/<run_id>/model_config.json \
  --output runs/<run_id>/eval_report.json
```

The script evaluates:
- TinyStories validation (in-domain)
- WikiText-2 test (cross-domain)

And reports:
- cross-entropy (nats/token)
- perplexity
- bits-per-byte (BPB)

Confirm JAX is using the GPU: `jax.devices()[0].platform` should show **`gpu`**.

```bash
python3 -c "import jax; print(jax.devices()[0]); print(jax.devices()[0].platform)"
```

**One-liner for a fresh Ubuntu instance** (JAX + training deps + quick GPU check):

```bash
pip install -U "jax[cuda12]" flax optax grain datasets tiktoken orbax-checkpoint tqdm -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && python3 -c "import jax; print(f'JAX Device: {jax.devices()[0]}'); print('✓ Ready!' if jax.devices()[0].platform == 'gpu' else '✗ GPU not detected')"
```

## Repository layout

- `basic_gpt_jax.py` — minimal LLM training tutorial / baseline  
- `gpt_improved.py` — scaled-up training with checkpoints and sampling  
- `runs/` — created when you run the improved script (checkpoints and logs)
