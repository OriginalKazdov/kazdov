"""Eval harness for Kazdov alpha + baselines.

Metrics:
  - `heldout_loss` — mean next-token LM loss on a fixed held-out slice of each
    dataset in data/alpha/. Measures generalization on the same distribution
    the model trained on. Use for ablations + main-run curves.
  - `gsm8k_sample` — a minimal 0-shot qualitative probe. Reads a few GSM8K
    problems from HF, generates a continuation, prints them. Not a metric,
    just a sanity check.

Models supported via --model:
  - kazdov:<path>  → load KazdovLM checkpoint (.pt)
  - hf:<hf_path>   → load HF causal LM (e.g. hf:EleutherAI/pythia-70m-deduped)

Example:
  python experiments/eval_alpha_lm.py --model hf:EleutherAI/pythia-70m-deduped
  python experiments/eval_alpha_lm.py --model kazdov:results/kazdov_smoke.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------- Held-out loss ----------

HELDOUT_FRAC = 0.005  # last 0.5% of each bin is the held-out split
BINS = {
    "open_web_math":    "data/alpha/open_web_math.bin",
    "proof_pile_v1":    "data/alpha/proof_pile_v1.bin",
    "mathpile":         "data/alpha/mathpile.bin",
    "automath":         "data/alpha/automath.bin",
    "numina_math":      "data/alpha/numina_math.bin",
}


def heldout_batches(bin_path, B, L, n_batches):
    """Deterministic batches from the LAST `HELDOUT_FRAC` of a .bin file."""
    arr = np.memmap(bin_path, dtype=np.uint16, mode='r')
    total = arr.shape[0]
    split = int(total * (1 - HELDOUT_FRAC))
    held = arr[split:]  # last slice
    # Skip last incomplete window
    n_complete = (held.shape[0] // L) - 1
    if n_complete < n_batches * B:
        n_batches = max(1, n_complete // B)
    # Grid of (n_batches * B) offsets evenly spread over held split
    offsets = np.linspace(0, n_complete - 1, n_batches * B).astype(np.int64) * L
    for bi in range(n_batches):
        start = bi * B
        xs = np.stack([held[o:o + L].astype(np.int64) for o in offsets[start:start + B]])
        yield torch.from_numpy(xs)


@torch.no_grad()
def eval_heldout_loss(model, forward_fn, bins, B, L, n_batches=20, device='cuda'):
    """Returns dict: per-dataset mean LM loss."""
    model.eval()
    out = {}
    for name, path in bins.items():
        t0 = time.time()
        losses = []
        for x in heldout_batches(path, B, L, n_batches):
            x = x.to(device)
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                loss = forward_fn(model, x)
            losses.append(loss.item())
        out[name] = {
            'loss': sum(losses) / len(losses),
            'n_batches': len(losses),
            'tokens': len(losses) * B * L,
            'time_s': time.time() - t0,
        }
        print(f"  {name:20s} loss {out[name]['loss']:.4f} ({out[name]['tokens']:,} tok, {out[name]['time_s']:.1f}s)", flush=True)
    out['mean'] = sum(v['loss'] for v in out.values() if isinstance(v, dict)) / len(bins)
    print(f"  {'MEAN':20s} loss {out['mean']:.4f}", flush=True)
    return out


# ---------- Model loaders ----------

def load_kazdov(ckpt_path):
    """Load a KazdovLM from a checkpoint, or fresh if ckpt_path is 'random'."""
    from kazdov.kazdov_lm import KazdovLM
    CFG = dict(
        vocab_size=50257, d_model=512, n_layers=12, n_heads=8, rank=128,
        mlp_dim=2048, max_len=256, n_experts=4,
        use_mobe=True, use_bi_bcn=False, use_hybrid_mha=True,
    )
    m = KazdovLM(**CFG).cuda()
    if ckpt_path and ckpt_path != 'random' and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location='cuda')
        if 'model' in state: state = state['model']
        m.load_state_dict(state, strict=False)
        print(f"Loaded Kazdov from {ckpt_path}", flush=True)
    else:
        print(f"Kazdov: random init (untrained baseline)", flush=True)

    def forward(model, x):
        out = model(x, labels=x)
        return out['loss']
    return m, forward


def load_hf(hf_path):
    """Load an HF causal LM. Tokenizer must match our GPT-2 vocab or we
    truncate inputs to the HF tokenizer's vocab range."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_path)
    m = AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype=torch.bfloat16).cuda()
    m.eval()
    hf_vocab = m.config.vocab_size
    print(f"Loaded HF model {hf_path}: vocab={hf_vocab}, params={sum(p.numel() for p in m.parameters())/1e6:.1f}M", flush=True)

    def forward(model, x):
        # Our data is GPT-2 tokenized (50257 vocab). HF models may have diff vocab.
        # Simple approach: clamp token ids to the HF model's vocab range. Sub-optimal
        # but gives a comparable loss; otherwise would need to re-tokenize from raw text.
        x_clamp = x.clamp(0, hf_vocab - 1)
        out = model(x_clamp, labels=x_clamp)
        return out.loss
    return m, forward


# ---------- GSM8K qualitative ----------

@torch.no_grad()
def gsm8k_sample(model, forward_fn_ignored, hf_tokenizer=None, n=3, max_new=80):
    """Print qualitative samples on a few GSM8K problems. No accuracy,
    just 'does it generate anything reasonable'."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed, skipping gsm8k_sample", flush=True)
        return
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    probs = list(next(iter(ds.take(n))) for _ in range(n))
    # Only supported for HF models here (Kazdov lacks generate())
    if hasattr(model, 'generate'):
        for p in probs:
            q = p['question']
            ids = hf_tokenizer(q + "\nAnswer: ", return_tensors='pt').input_ids.cuda()
            out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
            text = hf_tokenizer.decode(out[0])
            print(f"\nQ: {q}\nGEN: {text[len(q):]}\nGOLD: {p['answer'][:80]}...", flush=True)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="kazdov:<ckpt_path> | kazdov:random | hf:<hf_path>")
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--n_batches", type=int, default=20)
    ap.add_argument("--out", type=str, default=None,
                    help="JSON output path (optional)")
    ap.add_argument("--skip_gsm8k", action="store_true")
    args = ap.parse_args()

    kind, target = args.model.split(":", 1)
    if kind == 'kazdov':
        model, fwd = load_kazdov(target)
        hf_tok = None
    elif kind == 'hf':
        model, fwd = load_hf(target)
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(target)
    else:
        raise ValueError(f"unknown model kind: {kind}")

    print(f"\n=== Held-out LM loss ({args.n_batches} batches × B={args.B} × L={args.L} = {args.n_batches*args.B*args.L:,} tok/dataset) ===", flush=True)
    heldout = eval_heldout_loss(model, fwd, BINS, args.B, args.L, args.n_batches)

    if not args.skip_gsm8k and hasattr(model, 'generate'):
        print(f"\n=== GSM8K qualitative (first 3) ===", flush=True)
        gsm8k_sample(model, fwd, hf_tokenizer=hf_tok)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'model': args.model, 'heldout_loss': heldout}, f, indent=2)
        print(f"\nSaved: {args.out}", flush=True)


if __name__ == '__main__':
    main()
