"""DAY 2 — Data prep for MoBE-BCN-70M alpha.

Top-tier math dataset mix (7B tokens, validated by 2024-2026 SOTA work):
  35% Proof-Pile-2 AlgebraicStack (Lean+Isabelle+Coq+SymPy+Mathematica)
  20% Proof-Pile-2 arXiv math
  15% MathPile (ProofWiki + StackExchange + Wikipedia math)
  10% AutoMathText (top quality scored slice)
  10% OpenMathReasoning TIR (tool-integrated reasoning, short traces)
   5% NuminaMath-CoT (olympiad)
   5% OpenThoughts3-1.2M math (shortest 50K)

Downloads from HuggingFace, tokenizes with GPT-2 BPE, shards to .bin format.

Usage:
    # Smoke test (~100MB total, quick validation):
    python prep_alpha_data.py --smoke

    # Full prep (~7B tokens, takes hours):
    python prep_alpha_data.py --full --out_dir data/alpha
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ============= Dataset specs =============

DATASETS = {
    "open_web_math": {
        "hf_path": "open-web-math/open-web-math",
        "config": None,
        "split": "train",
        "text_field": "text",
        "weight": 0.35,
        "desc": "OpenWebMath (ICLR 2024): math-dense web corpus, parquet format",
    },
    "proof_pile_v1": {
        "hf_path": "hoskinson-center/proof-pile",
        "config": None,
        "split": "train",
        "text_field": "text",
        "weight": 0.20,
        "desc": "proof-pile v1 (Lean+Isabelle+Coq+Math papers, simpler format)",
    },
    "mathpile": {
        "hf_path": "GAIR/MathPile_Commercial",  # commercial-friendly license
        "config": None,
        "split": "train",
        "text_field": "text",
        "weight": 0.15,
        "desc": "MathPile_Commercial: ProofWiki+StackExchange+Wikipedia math (curated, commercial license)",
    },
    "automath": {
        "hf_path": "math-ai/AutoMathText",
        "config": "web-0.70-to-1.00",  # top quality slice
        "split": "train",
        "text_field": "text",
        "weight": 0.10,
        "desc": "AutoMathText top quality slice",
    },
    "openmath_reasoning": {
        "hf_path": "nvidia/OpenMathReasoning",
        "config": "tir",  # tool-integrated reasoning split
        "split": "train",
        "text_field": "generated_solution",
        "weight": 0.10,
        "desc": "NVIDIA OpenMathReasoning TIR split",
    },
    "numina_math": {
        "hf_path": "AI-MO/NuminaMath-CoT",
        "config": None,
        "split": "train",
        "text_field": "solution",
        "weight": 0.05,
        "desc": "NuminaMath-CoT olympiad",
    },
    "openthoughts3": {
        "hf_path": "open-thoughts/OpenThoughts3-1.2M",
        "config": None,
        "split": "train",
        "text_field": "conversations",
        "weight": 0.05,
        "desc": "OpenThoughts3 (sample 50K examples, mostly math)",
        "max_examples": 50000,
        # Removed filter — OT3 is ~70% math anyway, take all
    },
}

TOTAL_TOKENS_TARGET = 7_000_000_000  # 7B tokens
SMOKE_TOKENS_PER_DATASET = 100_000  # 100K tokens for smoke


# ============= Download + Tokenize =============

def get_tokenizer():
    """GPT-2 BPE (50K vocab). Suboptimal for LaTeX (~30% waste) but works.
    For v2: train custom math-aware BPE on the corpus."""
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def safe_text_extract(example, field):
    """Extract text robustly, handling chat/conversations format."""
    val = example.get(field) or example.get("text") or example.get("solution") or ""
    if isinstance(val, list):
        # Chat format: list of {role, content} dicts
        parts = []
        for turn in val:
            if isinstance(turn, dict):
                content = turn.get("content") or turn.get("text") or ""
                parts.append(str(content))
            else:
                parts.append(str(turn))
        return "\n".join(parts)
    return str(val)


def stream_dataset(spec, max_tokens, tokenizer, smoke=False):
    """Stream a HF dataset, tokenize on-the-fly, yield batches of tokens.
    Uses streaming to avoid downloading full dataset at once."""
    from datasets import load_dataset

    name = spec["hf_path"]
    cfg = spec.get("config")
    split = spec["split"]
    field = spec["text_field"]

    print(f"Loading {name} (config={cfg}, split={split}) streaming...", flush=True)
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    kwargs = {"split": split, "streaming": True, "trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token
    try:
        if cfg:
            ds = load_dataset(name, cfg, **kwargs)
        else:
            ds = load_dataset(name, **kwargs)
    except Exception as e:
        print(f"  ERROR loading {name}: {e}", flush=True)
        return

    eos = tokenizer.eos_token_id
    tokens_so_far = 0
    examples_seen = 0
    max_ex = spec.get("max_examples", float("inf"))

    for example in ds:
        if examples_seen >= max_ex:
            break
        text = safe_text_extract(example, field)
        if not text or len(text) < 20:
            continue
        # Optional filter (e.g., math-only)
        flt = spec.get("filter")
        if flt and flt.lower() not in text.lower()[:500]:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 0:
            yield ids + [eos]
            tokens_so_far += len(ids) + 1
            examples_seen += 1
        if tokens_so_far >= max_tokens:
            break

    print(f"  {name}: collected {tokens_so_far:,} tokens from {examples_seen:,} examples", flush=True)


def write_shard(token_iter, out_path, max_tokens):
    """Write tokens to a .bin file (uint16 if vocab < 65k, else uint32)."""
    f = open(out_path, "wb")
    total = 0
    for tokens in token_iter:
        arr = np.array(tokens, dtype=np.uint16)  # GPT-2 vocab < 65k, fits uint16
        f.write(arr.tobytes())
        total += len(tokens)
        if total >= max_tokens:
            break
    f.close()
    return total


def prep_dataset(name, spec, total_target, tokenizer, out_dir, smoke=False):
    """Download + tokenize + shard one dataset."""
    target_tokens = SMOKE_TOKENS_PER_DATASET if smoke else int(total_target * spec["weight"])
    out_path = out_dir / f"{name}.bin"

    if out_path.exists() and not smoke:
        existing = out_path.stat().st_size // 2  # uint16 = 2 bytes per token
        if existing >= target_tokens * 0.9:
            print(f"[{name}] already done ({existing:,} tokens) — skip", flush=True)
            return existing
        print(f"[{name}] partial ({existing:,}/{target_tokens:,}) — re-doing", flush=True)

    print(f"\n[{name}] target {target_tokens:,} tokens — {spec['desc']}", flush=True)
    t0 = time.time()
    iter_tokens = stream_dataset(spec, target_tokens * 2, tokenizer, smoke)  # 2× safety margin
    written = write_shard(iter_tokens, out_path, target_tokens)
    print(f"[{name}] wrote {written:,} tokens to {out_path} ({time.time()-t0:.1f}s)", flush=True)
    return written


# ============= Main =============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="Quick validation: ~100K tokens per dataset")
    ap.add_argument("--full", action="store_true",
                     help="Full prep: 7B tokens total")
    ap.add_argument("--out_dir", type=str, default="data/alpha")
    ap.add_argument("--datasets", type=str, default="all",
                     help="Comma-sep dataset names or 'all'")
    args = ap.parse_args()

    if not args.smoke and not args.full:
        print("Specify --smoke or --full", flush=True)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prep_alpha] mode: {'SMOKE' if args.smoke else 'FULL'}", flush=True)
    print(f"[prep_alpha] out_dir: {out_dir}", flush=True)
    print(f"[prep_alpha] target tokens: {TOTAL_TOKENS_TARGET:,}" if args.full else "[prep_alpha] smoke samples", flush=True)

    print(f"\nLoading tokenizer (GPT-2 BPE)...", flush=True)
    tokenizer = get_tokenizer()
    print(f"Tokenizer loaded: vocab={tokenizer.vocab_size}", flush=True)

    if args.datasets == "all":
        names = list(DATASETS.keys())
    else:
        names = args.datasets.split(",")

    summary = {}
    for name in names:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}", flush=True)
            continue
        try:
            written = prep_dataset(name, DATASETS[name], TOTAL_TOKENS_TARGET,
                                     tokenizer, out_dir, smoke=args.smoke)
            summary[name] = written
        except Exception as e:
            print(f"[{name}] FAILED: {e}", flush=True)
            summary[name] = 0

    print(f"\n=== SUMMARY ===", flush=True)
    total = 0
    for name, n in summary.items():
        print(f"  {name}: {n:,} tokens", flush=True)
        total += n
    print(f"TOTAL: {total:,} tokens", flush=True)

    # Write metadata
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "mode": "smoke" if args.smoke else "full",
            "tokenizer": "gpt2",
            "vocab_size": tokenizer.vocab_size,
            "datasets": summary,
            "total_tokens": total,
        }, f, indent=2)
    print(f"Metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
