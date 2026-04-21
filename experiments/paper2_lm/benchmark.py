"""Diagnostic: isolate MoBE-BCN bottleneck — test small configs + ablation."""
import sys, time
import torch
from kazdov.kazdov_lm import KazdovLM

print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)

def bench(cfg, B, L, N=5, grad_ckpt=False):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = KazdovLM(**cfg).cuda()
    if grad_ckpt: m.enable_grad_checkpointing()
    m.train()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    # warmup
    x = torch.randint(0, 50257, (B, L)).cuda()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = m(x, labels=x); out['loss'].backward()
    m.zero_grad()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        x = torch.randint(0, 50257, (B, L)).cuda()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = m(x, labels=x); out['loss'].backward()
        m.zero_grad()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / N
    tok_s = B * L / dt
    peak = torch.cuda.max_memory_allocated() / 1e9
    del m
    return {'params_M': n, 'ms_step': dt*1000, 'tok_s': tok_s, 'peak_GB': peak}

BASE = dict(vocab_size=50257, d_model=512, n_layers=12, n_heads=8, rank=128,
            mlp_dim=2048, max_len=256, n_experts=4, use_mobe=True, use_hybrid_mha=True)

tests = [
    ("full alpha (L=256, hybrid, MoBE, K=4, ckpt)", {**BASE}, 4, 256, True),
    ("no hybrid MHA",                                 {**BASE, 'use_hybrid_mha': False}, 4, 256, True),
    ("no MoBE (K=1 BCN only)",                        {**BASE, 'use_mobe': False, 'use_hybrid_mha': False}, 4, 256, True),
    ("L=128, full alpha",                             {**BASE, 'max_len': 128}, 4, 128, True),
    ("L=64, full alpha",                              {**BASE, 'max_len': 64}, 4, 64, True),
    ("tiny (d=192, L=128, 6 layers, MoBE, hybrid)",   {**BASE, 'd_model': 192, 'n_layers': 6, 'rank': 64, 'mlp_dim': 768, 'n_heads': 4, 'max_len': 128}, 8, 128, False),
    ("no grad ckpt (L=128)",                          {**BASE, 'max_len': 128}, 4, 128, False),
]

print(f"{'cfg':<50} {'params':>8} {'ms/step':>8} {'tok/s':>10} {'peak GB':>8}", flush=True)
for name, cfg, B, L, ckpt in tests:
    try:
        r = bench(cfg, B, L, grad_ckpt=ckpt)
        print(f"{name:<50} {r['params_M']:>7.1f}M {r['ms_step']:>8.0f} {r['tok_s']:>10,.0f} {r['peak_GB']:>7.2f}", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        print(f"{name:<50} OOM", flush=True)
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{name:<50} ERR {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()
