"""Quick test script: load latest Kazdov checkpoint, generate from a prompt.

Usage:
  python experiments/test_kazdov.py "The derivative of x^2 is"
  python experiments/test_kazdov.py --ckpt results/kazdov_alpha/latest.pt --max_tokens 60 --temp 0.7 "\\int_0^1 x dx ="
  python experiments/test_kazdov.py --list-ckpts
"""
import argparse
import glob
import sys
from pathlib import Path

import torch
from kazdov.kazdov_lm import KazdovLM


DEFAULT_CKPT_DIR = Path("results/kazdov_alpha")


def latest_ckpt(directory):
    """Prefer 'latest.pt' if it exists, otherwise highest step_*.pt."""
    latest = directory / "latest.pt"
    if latest.exists():
        return str(latest)
    candidates = sorted(glob.glob(str(directory / "step_*.pt")))
    return candidates[-1] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="The derivative of x^2 is")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--max_tokens", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--list-ckpts", action="store_true")
    args = ap.parse_args()

    if args.list_ckpts:
        for p in sorted(glob.glob(str(DEFAULT_CKPT_DIR / "*.pt"))):
            sz = Path(p).stat().st_size / 1e6
            print(f"{p}  ({sz:.0f} MB)")
        return

    ckpt_path = args.ckpt or latest_ckpt(DEFAULT_CKPT_DIR)
    if not ckpt_path or not Path(ckpt_path).exists():
        print(f"No checkpoint found. Use --ckpt or wait for training to save one.")
        return

    state = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    cfg = state.get('cfg') or state.get('model_cfg') or dict(
        vocab_size=50257, d_model=512, n_layers=12, n_heads=8, rank=128,
        mlp_dim=2048, max_len=256, n_experts=4,
        use_mobe=True, use_bi_bcn=False, use_hybrid_mha=True,
    )
    model = KazdovLM(**cfg).cuda()
    sd = state.get('model', state)
    model.load_state_dict(sd, strict=False)
    model.eval()

    tokens_trained = state.get('tokens', '?')
    step = state.get('step', '?')
    print(f"[ckpt] {ckpt_path}  step={step}  tokens={tokens_trained:,}" if isinstance(tokens_trained, int) else f"[ckpt] {ckpt_path}  step={step}", flush=True)

    # Tokenize
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ids = tok.encode(args.prompt, return_tensors='pt').cuda()
    print(f"\nPROMPT: {args.prompt}", flush=True)
    print(f"(tokens: {ids.shape[1]})", flush=True)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model.generate(
            ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
        )
    gen_ids = out[0, ids.shape[1]:].cpu()
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    print(f"\nGEN: {gen_text}", flush=True)


if __name__ == '__main__':
    main()
