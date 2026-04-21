"""Kazdov-alpha main training loop — 98M params, math mix, overnight on 4090.

Config (locked in D3):
  model:  d=512 L=256 n_layers=12 n_heads=8 rank=128 K=4, cumsum-fused MoBE
  data:   5.95B-token math mix (open_web_math 41% / proof_pile_v1 24% / mathpile 18% /
          automath 12% / numina 6%)
  opt:    AdamW lr=5e-4 (beta1=0.9 beta2=0.95 wd=0.1), grad-clip 1.0
  sched:  WSD (warmup 500, stable, decay last 20%)
  mix:    bf16 autocast
  B=32, L=256 → 8192 tok/step
  target: 5B tokens = ~610K steps ≈ 23 hr at ~60K tok/s on 4090

Checkpoints:
  results/kazdov_alpha/latest.pt        every save_every steps (overwrites)
  results/kazdov_alpha/step_<N>.pt      milestones (100M, 500M, 1B, 2B, 5B tok)
  results/kazdov_alpha/metrics.jsonl    one JSON line per log
  results/kazdov_alpha/config.json      run config snapshot

Resume:
  python train_kazdov_alpha.py --resume results/kazdov_alpha/latest.pt
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from kazdov.kazdov_lm import KazdovLM
from kazdov.schedulers import make_wsd_lambda


# ---------- Config ----------

MODEL_CFG = dict(
    vocab_size=50257, d_model=512, n_layers=12, n_heads=8, rank=128,
    mlp_dim=2048, max_len=256, n_experts=4,
    use_mobe=True, use_bi_bcn=False, use_hybrid_mha=True,
)

DATA_MIX = {
    "open_web_math":    ("data/alpha/open_web_math.bin",    0.45),
    "proof_pile_v1":    ("data/alpha/proof_pile_v1.bin",    0.22),
    "mathpile":         ("data/alpha/mathpile.bin",         0.15),
    "automath":         ("data/alpha/automath.bin",         0.10),
    "numina_math":      ("data/alpha/numina_math.bin",      0.08),
}

MILESTONE_TOKENS = [100_000_000, 500_000_000, 1_000_000_000, 2_000_000_000, 5_000_000_000]


# ---------- Data ----------

class MemmapSampler:
    def __init__(self, mix, B, L):
        self.mix = mix
        self.B = B
        self.L = L
        self.mmaps = {n: np.memmap(p, dtype=np.uint16, mode='r') for n, (p, _) in mix.items()}
        # Carve held-out: last 0.5% of each file (same as eval_alpha_lm.py)
        self.heldout_frac = 0.005
        self.train_end = {n: int(m.shape[0] * (1 - self.heldout_frac)) for n, m in self.mmaps.items()}
        self.names = list(mix.keys())
        self.weights = np.array([w for _, w in mix.values()])
        self.weights = self.weights / self.weights.sum()

    def batch(self):
        x = np.empty((self.B, self.L), dtype=np.int64)
        for i in range(self.B):
            ds = self.names[np.random.choice(len(self.names), p=self.weights)]
            arr = self.mmaps[ds]
            limit = self.train_end[ds] - self.L - 1
            o = random.randint(0, limit)
            x[i] = arr[o:o + self.L].astype(np.int64)
        return torch.from_numpy(x)

    @torch.no_grad()
    def heldout_batches(self, name, n_batches=5):
        """Deterministic eval batches from the held-out tail of `name`."""
        arr = self.mmaps[name]
        start = self.train_end[name]
        held = arr[start:]
        n_windows = (held.shape[0] // self.L) - 1
        take = min(n_batches * self.B, n_windows)
        offsets = np.linspace(0, n_windows - 1, take).astype(np.int64) * self.L
        for bi in range(0, take, self.B):
            chunk = offsets[bi:bi + self.B]
            if len(chunk) < self.B: break
            xs = np.stack([held[o:o + self.L].astype(np.int64) for o in chunk])
            yield torch.from_numpy(xs)


# ---------- Eval ----------

@torch.no_grad()
def evaluate(model, sampler, device='cuda', n_batches=5):
    model.eval()
    out = {}
    for name in sampler.mix.keys():
        losses = []
        for x in sampler.heldout_batches(name, n_batches):
            x = x.to(device)
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                loss = model(x, labels=x)['loss']
            losses.append(loss.item())
        out[name] = sum(losses) / max(1, len(losses))
    out['mean'] = sum(v for k, v in out.items() if k != 'mean') / len(sampler.mix)
    model.train()
    return out


# ---------- Train ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results/kazdov_alpha")
    ap.add_argument("--total_tokens", type=int, default=5_000_000_000)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--mobe_bias_every", type=int, default=100)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    tok_per_step = args.B * args.L
    total_steps = args.total_tokens // tok_per_step
    print(f"[config] {args.total_tokens:,} target tokens / {tok_per_step} per step = {total_steps:,} steps", flush=True)

    # Model
    model = KazdovLM(**MODEL_CFG).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.2f}M params", flush=True)

    # Optimizer + scheduler
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                 weight_decay=args.wd)
    sched = LambdaLR(opt, make_wsd_lambda(args.warmup, total_steps,
                                            decay_frac=0.2, final_lr_frac=0.1))

    # Resume
    start_step = 0
    start_tokens = 0
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler'])
        start_step = state['step']
        start_tokens = state.get('tokens', start_step * tok_per_step)
        print(f"[resume] from {args.resume} at step {start_step} ({start_tokens:,} tokens)", flush=True)

    # Snapshot config
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            'model_cfg': MODEL_CFG,
            'train_args': vars(args),
            'total_steps': total_steps,
            'tok_per_step': tok_per_step,
            'n_params': n_params,
            'data_mix': {n: w for n, (_, w) in DATA_MIX.items()},
        }, f, indent=2)

    sampler = MemmapSampler(DATA_MIX, args.B, args.L)

    # Main loop
    model.train()
    t_run = time.time()
    t_log = t_run
    tokens_since_log = 0
    milestones_done = {m for m in MILESTONE_TOKENS if m <= start_tokens}

    for step in range(start_step, total_steps):
        x = sampler.batch().cuda()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(x, labels=x)
            loss = out['loss']
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad()

        if (step + 1) % args.mobe_bias_every == 0:
            model.update_mobe_biases()

        tokens_since_log += tok_per_step
        cur_tokens = start_tokens + (step + 1 - start_step) * tok_per_step

        # Log
        if (step + 1) % args.log_every == 0 or step == start_step:
            now = time.time()
            tok_s = tokens_since_log / max(1e-6, now - t_log) if step != start_step else 0
            entry = {
                't': int(now), 'step': step + 1,
                'tokens': cur_tokens,
                'loss': float(loss.item()),
                'lr': float(sched.get_last_lr()[0]),
                'gnorm': float(gnorm),
                'tok_s': tok_s,
                'elapsed_hr': (now - t_run) / 3600,
            }
            with open(metrics_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            print(f"step {step+1:>6} | tok {cur_tokens/1e6:>7.1f}M | loss {loss.item():.4f} | lr {entry['lr']:.2e} | gnorm {gnorm:.2f} | {tok_s:,.0f} tok/s", flush=True)
            t_log = now
            tokens_since_log = 0

        # Eval
        if (step + 1) % args.eval_every == 0:
            eval_out = evaluate(model, sampler)
            print(f"  [eval] mean {eval_out['mean']:.4f}  " +
                  "  ".join(f"{n}: {v:.3f}" for n, v in eval_out.items() if n != 'mean'), flush=True)
            with open(metrics_path, 'a') as f:
                f.write(json.dumps({'t': int(time.time()), 'step': step + 1,
                                      'tokens': cur_tokens, 'eval': eval_out}) + '\n')

        # Save
        if (step + 1) % args.save_every == 0:
            ckpt = {
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict(),
                'step': step + 1,
                'tokens': cur_tokens,
                'loss': float(loss.item()),
                'cfg': MODEL_CFG,
            }
            torch.save(ckpt, out_dir / 'latest.pt')

        # Milestone (named checkpoint + eval)
        for m in MILESTONE_TOKENS:
            if m not in milestones_done and cur_tokens >= m:
                milestones_done.add(m)
                print(f"  [MILESTONE {m/1e9:.1f}B tokens reached at step {step+1}]", flush=True)
                torch.save({'model': model.state_dict(), 'step': step + 1,
                             'tokens': cur_tokens, 'cfg': MODEL_CFG},
                            out_dir / f'step_{step+1}_tok_{m//1_000_000}M.pt')

    # Final save
    torch.save({'model': model.state_dict(), 'step': total_steps,
                 'tokens': args.total_tokens, 'cfg': MODEL_CFG},
                out_dir / 'final.pt')
    final_eval = evaluate(model, sampler, n_batches=10)
    print(f"\n=== FINAL ===", flush=True)
    print(f"Final eval: mean {final_eval['mean']:.4f}", flush=True)
    for n, v in final_eval.items():
        if n != 'mean':
            print(f"  {n}: {v:.4f}", flush=True)
    with open(out_dir / 'final_eval.json', 'w') as f:
        json.dump(final_eval, f, indent=2)


if __name__ == '__main__':
    main()
