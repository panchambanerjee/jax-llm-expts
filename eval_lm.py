"""Evaluation harness for MiniGPT checkpoints.

Computes CE/BPB/PPL on:
- TinyStories validation split
- WikiText-2 test split
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import optax
from datasets import load_dataset

from checkpoint_utils import restore_model_weights
from model import create_model_from_config, load_model_config
from train_common import get_gpt2_tokenizer


def iter_chunks(tokens: list[int], max_len: int) -> Iterable[list[int]]:
    for i in range(0, len(tokens), max_len):
        chunk = tokens[i : i + max_len]
        if len(chunk) < 2:
            continue
        yield chunk


def nll_on_tokens(model, tokens: list[int]) -> tuple[float, int]:
    arr = jnp.array(tokens, dtype=jnp.int32)[None, :]
    inputs = arr[:, :-1]
    targets = arr[:, 1:]
    logits = model(inputs)
    per_tok = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    nll = float(per_tok.sum())
    n_tokens = int(targets.size)
    return nll, n_tokens


def evaluate_texts(model, tokenizer, texts: list[str], max_len: int) -> dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    for text in texts:
        if not text:
            continue
        toks = tokenizer.encode(text, allowed_special="all")
        for chunk in iter_chunks(toks, max_len):
            nll, n_tokens = nll_on_tokens(model, chunk)
            total_nll += nll
            total_tokens += n_tokens
            total_bytes += len(tokenizer.decode(chunk[1:]).encode("utf-8"))

    if total_tokens == 0:
        return {
            "nll_sum": float("nan"),
            "n_tokens": 0,
            "n_bytes": 0,
            "ce_nats": float("nan"),
            "ppl": float("nan"),
            "bpb": float("nan"),
        }

    ce_nats = total_nll / total_tokens
    ppl = math.exp(ce_nats)
    bpb = total_nll / (math.log(2.0) * max(total_bytes, 1))
    return {
        "nll_sum": total_nll,
        "n_tokens": total_tokens,
        "n_bytes": total_bytes,
        "ce_nats": ce_nats,
        "ppl": ppl,
        "bpb": bpb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MiniGPT checkpoint.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to final_model.orbax directory.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to model_config.json.",
    )
    parser.add_argument(
        "--tiny-split",
        default="validation[:10000]",
        help="TinyStories split for in-domain eval.",
    )
    parser.add_argument(
        "--wikitext-split",
        default="test",
        help="WikiText-2 split name (default: test).",
    )
    parser.add_argument(
        "--output",
        default="eval_report.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    cfg = load_model_config(args.config)
    model = create_model_from_config(cfg)
    restore_model_weights(args.checkpoint, model)

    tokenizer = get_gpt2_tokenizer()
    max_len = int(cfg["maxlen"])

    print("Loading TinyStories...")
    tiny = load_dataset("roneneldan/TinyStories", split=args.tiny_split)
    tiny_texts = [x["text"] for x in tiny]

    print("Loading WikiText-2...")
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.wikitext_split)
    wiki_texts = [x["text"] for x in wiki if x["text"].strip()]

    print("Evaluating TinyStories...")
    tiny_metrics = evaluate_texts(model, tokenizer, tiny_texts, max_len=max_len)
    print("Evaluating WikiText-2...")
    wiki_metrics = evaluate_texts(model, tokenizer, wiki_texts, max_len=max_len)

    report = {
        "model": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "config_path": str(Path(args.config).resolve()),
            "max_len": max_len,
            "vocab_size": int(cfg["vocab_size"]),
            "params": int(cfg.get("total_params", -1)),
            "device": str(jax.devices()[0]),
        },
        "datasets": {
            "tinystories": {
                "split": args.tiny_split,
                **tiny_metrics,
            },
            "wikitext2": {
                "split": args.wikitext_split,
                **wiki_metrics,
            },
        },
        "notes": [
            "Metrics are comparable across models only with same tokenizer/chunking protocol.",
            "TinyStories is in-domain; WikiText-2 is cross-domain.",
        ],
    }

    out = Path(args.output)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report: {out.resolve()}")
    print(json.dumps(report["datasets"], indent=2))


if __name__ == "__main__":
    main()
