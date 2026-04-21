"""Minimal example: load a Kazdov checkpoint, generate from a prompt.

Run from the repo root after installing dependencies:

    pip install -e .
    python examples/generate_demo.py
"""
import sys
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from kazdov import KazdovLM


def main():
    # Default config matches papers/02-kazdov-lm.md Section 4
    cfg = dict(
        vocab_size=50257, d_model=512, n_layers=12, n_heads=8, rank=128,
        mlp_dim=2048, max_len=256, n_experts=4,
        use_mobe=True, use_bi_bcn=False, use_hybrid_mha=True,
    )
    model = KazdovLM(**cfg)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Load a checkpoint if provided, otherwise the demo runs with random init
    ckpt_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if ckpt_arg and Path(ckpt_arg).exists():
        state = torch.load(ckpt_arg, map_location=device, weights_only=False)
        sd = state.get('model', state)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint: {ckpt_arg}")
    else:
        print("No checkpoint provided — running with random init "
              "(output will be meaningless).")
        print(f"Usage: python {sys.argv[0]} <checkpoint.pt>")

    model.eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    prompts = [
        "The derivative of x^2 is",
        r"\int_0^1 x^2 dx = ",
        "If a + b = 5 and a * b = 6, then",
        r"\sum_{i=1}^n i = ",
    ]

    for p in prompts:
        ids = tok.encode(p, return_tensors='pt').to(device)
        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device=='cuda')):
            out = model.generate(ids, max_new_tokens=40, temperature=0.7, top_k=40)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        print(f"\nPROMPT: {p}")
        print(f"GEN:    {gen}")


if __name__ == "__main__":
    main()
