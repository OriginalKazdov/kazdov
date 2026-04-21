"""Phase A2b — Power 2022 grokking replication for BCN-LM validation.

CANONICAL setup from Power et al. 2022 (arXiv:2201.02177) +
Nanda 2023 (arXiv:2301.05217):
- Task: a + b = c (mod P), single operation, fixed 5-token format
- Vocab: P digits + `+` + `=` (P+2 tokens)
- Train fraction: 0.3 of P×P pairs
- Full-batch gradient descent (NOT mini-batch — known grokking killer)
- AdamW lr=1e-3, wd=1.0, betas=(0.9, 0.98)
- Up to 40K epochs (each epoch = 1 full-batch update)
- Eval: predict result token at position 3 (the `=` position)

Tests 4 architectures matched in compute:
- Transformer-1L (Power/Nanda baseline)
- BCN-LM 1-layer (Bi-BCN attention only)
- MoBE-BCN-LM 1-layer K=4 (mixture of bilinear experts)
- Bi-BCN + MHA hybrid

Gate: Transformer MUST grok. If not, setup still wrong.
Then: compare BCN family grok speed vs Transformer baseline.

Format example (P=11):
  Input:  [3, 12, 5, 13, ?]   where 12 = '+' and 13 = '='
  Target: 8                    (3 + 5 mod 11)

We predict at position 3 (the '=' position) what comes next.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============= Vocab + data =============

def build_vocab(P, ops=("+",)):
    """Vocab: digits 0..P-1, then ops..., then `=`."""
    digits = [str(i) for i in range(P)]
    return digits + list(ops) + ["="]


def gen_full_dataset(P, ops=("+",)):
    """All ops × P*P pairs of (a, b) with their result.
    Returns: input_ids (n_ops*P*P, 4), labels (n_ops*P*P,) — predict at pos 3.
    Vocab layout: 0..P-1 = digits, P..P+n_ops-1 = ops, P+n_ops = '='.
    """
    op_id_base = P
    eq_id = P + len(ops)
    inputs, labels = [], []
    for op_idx, op in enumerate(ops):
        op_id = op_id_base + op_idx
        for a in range(P):
            for b in range(P):
                if op == "+": c = (a + b) % P
                elif op == "*": c = (a * b) % P
                elif op == "-": c = (a - b) % P
                else: raise ValueError(op)
                inputs.append([a, op_id, b, eq_id])
                labels.append(c)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def split_data(inputs, labels, train_frac, seed):
    n = inputs.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    cut = int(train_frac * n)
    return (inputs[perm[:cut]], labels[perm[:cut]]), (inputs[perm[cut:]], labels[perm[cut:]])


# ============= Models =============

class TransformerPower(nn.Module):
    """Power/Nanda canonical: 1-layer transformer, no LayerNorm, no tied embed.
    Predicts result at position 3 (the `=` position)."""
    def __init__(self, vocab_size, d_model=128, n_heads=4, mlp_dim=512, max_len=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, d_model),
        )
        self.head = nn.Linear(d_model, vocab_size)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.embed(x) + self.pos(pos)
        # Causal mask
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h2, _ = self.attn(h, h, h, attn_mask=causal, need_weights=False)
        h = h + h2
        h = h + self.mlp(h)
        # Predict at last position (3 = '=')
        return self.head(h[:, -1, :])  # (B, vocab_size)


class BilinearComposition(nn.Module):
    """out = U @ ((Vᵀx) ⊙ (Wᵀy)) + bias."""
    def __init__(self, d_in, d_out, rank, init_scale=0.02):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_out, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(d_in, rank) * init_scale)
        self.W = nn.Parameter(torch.randn(d_in, rank) * init_scale)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x, y):
        return ((x @ self.V) * (y @ self.W)) @ self.U.T + self.bias


class MixtureBilinear(nn.Module):
    """K bilinear experts + router."""
    def __init__(self, d_in, d_out, rank, n_experts=4, init_scale=0.02):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            BilinearComposition(d_in, d_out, rank, init_scale)
            for _ in range(n_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(d_in * 2, max(d_in, 32)), nn.GELU(),
            nn.Linear(max(d_in, 32), n_experts),
        )

    def forward(self, x, y):
        weights = torch.softmax(self.router(torch.cat([x, y], -1)), -1)
        outs = torch.stack([e(x, y) for e in self.experts], -1)  # (B, d, K)
        return (outs * weights.unsqueeze(1)).sum(-1)


class BCNPower(nn.Module):
    """BCN for Power task: embed(a), embed(b) → bilinear compose → head.
    Mimics Power's transformer in spirit but with bilinear primitive instead of attention.
    """
    def __init__(self, vocab_size, d_model=128, rank=64, mlp_dim=512,
                  use_mobe=False, n_experts=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.use_mobe = use_mobe
        if use_mobe:
            self.compose = MixtureBilinear(d_model, d_model, rank, n_experts)
        else:
            self.compose = BilinearComposition(d_model, d_model, rank)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, d_model),
        )
        self.head = nn.Linear(d_model, vocab_size)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        # x: (B, 4) = [a, op, b, =]
        e = self.embed(x)  # (B, 4, d)
        # Modulate operands by op embedding (so model knows WHICH op).
        # For single-op this is harmless; for multi-op it's required.
        op_e = e[:, 1]  # the operation token embedding
        a_e = e[:, 0] + op_e
        b_e = e[:, 2] + op_e
        h = self.compose(a_e, b_e)
        h = h + self.mlp(h)
        return self.head(h)  # (B, vocab_size)


def make_model(arch, vocab_size, d_model, rank, mlp_dim, n_heads):
    if arch == "transformer":
        return TransformerPower(vocab_size, d_model, n_heads, mlp_dim)
    elif arch == "bcn":
        return BCNPower(vocab_size, d_model, rank, mlp_dim, use_mobe=False)
    elif arch == "bcn_mobe":
        return BCNPower(vocab_size, d_model, rank, mlp_dim, use_mobe=True, n_experts=4)
    else:
        raise ValueError(arch)


# ============= Train loop (full-batch GD as Power/Nanda) =============

def train_run(arch, P, ops, args, device, eid, seed, out_root):
    ops_str = "".join(ops)
    print(f"\n=== {arch} P={P} ops={ops_str} seed={seed} ===", flush=True)
    torch.manual_seed(seed)

    vocab_size = P + len(ops) + 1  # digits + ops + `=`
    inputs, labels = gen_full_dataset(P, ops)
    (tr_x, tr_y), (te_x, te_y) = split_data(inputs, labels, args["train_frac"], seed)
    tr_x_d, tr_y_d = tr_x.to(device), tr_y.to(device)
    te_x_d, te_y_d = te_x.to(device), te_y.to(device)
    print(f"vocab={vocab_size} train={tr_x.shape[0]} test={te_x.shape[0]}", flush=True)

    model = make_model(arch, vocab_size, args["d_model"], args["rank"],
                        args["mlp_dim"], args["n_heads"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args["lr"],
                              weight_decay=args["wd"], betas=(0.9, 0.98))

    # Sanitize ops_str for Windows filenames (no *, ?, <, >, etc.)
    safe_ops = ops_str.replace("+", "p").replace("-", "m").replace("*", "x")
    cond = f"{arch}_P{P}_ops{safe_ops}"
    metadata = {
        "experiment_id": eid, "arch": arch, "P": P, "ops": list(ops), "seed": seed,
        "args": args, "n_params": n_params, "vocab_size": vocab_size,
        "n_train": int(tr_x.shape[0]), "n_test": int(te_x.shape[0]),
    }
    log = {
        "metadata": metadata, "history": [],
        "best_test_acc": 0.0, "first_05_step": None, "first_grok_step": None,
    }

    out_dir = Path(out_root) / "bcn" / time.strftime("%Y-%m-%d") / eid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cond}_seed{seed}.json"
    print(f"Writing to {out_path}", flush=True)

    model.train()
    t0 = time.time()
    for step in range(1, args["epochs"] + 1):
        # Full batch
        logits = model(tr_x_d)
        loss = F.cross_entropy(logits, tr_y_d)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args["eval_every"] == 0 or step == 1:
            with torch.no_grad():
                train_acc = (logits.argmax(-1) == tr_y_d).float().mean().item()
                te_logits = model(te_x_d)
                te_loss = F.cross_entropy(te_logits, te_y_d).item()
                te_acc = (te_logits.argmax(-1) == te_y_d).float().mean().item()
            entry = {"step": step, "train_loss": loss.item(),
                      "train_acc": train_acc, "test_acc": te_acc, "test_loss": te_loss,
                      "elapsed_s": time.time() - t0}
            log["history"].append(entry)
            if te_acc > log["best_test_acc"]: log["best_test_acc"] = te_acc
            if log["first_05_step"] is None and te_acc >= 0.5: log["first_05_step"] = step
            if log["first_grok_step"] is None and te_acc >= 0.99: log["first_grok_step"] = step
            print(f"[{cond}/s{seed}] step={step:>6d} loss={loss.item():.4f} "
                  f"train={train_acc:.3f} test={te_acc:.3f} best={log['best_test_acc']:.3f}",
                  flush=True)
            with open(out_path, "w") as f:
                json.dump(log, f, indent=2)
            if log["first_grok_step"] is not None and step - log["first_grok_step"] > 1000:
                print(f"[{cond}/s{seed}] grokked at {log['first_grok_step']} — exit early.",
                      flush=True)
                break

    log["total_time_s"] = time.time() - t0
    log["completed"] = True
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    return log


# ============= Main =============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", type=str, default="transformer,bcn,bcn_mobe")
    ap.add_argument("--P", type=int, default=97,
                     help="Modulus prime (Power used 97, Nanda 113)")
    ap.add_argument("--ops", type=str, default="+",
                     help="Comma-sep ops: +,-,* (multi-op uses MoBE expert specialization)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--train_frac", type=float, default=0.3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--mlp_dim", type=int, default=512)
    ap.add_argument("--out_root", type=str, default="results")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 500
        args.eval_every = 50
        args.archs = "transformer"
        args.seeds = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[modarith_power] device={device}", flush=True)

    eid = "modarith_power_phaseA2b"
    args_dict = {"epochs": args.epochs, "eval_every": args.eval_every,
                  "lr": args.lr, "wd": args.wd, "train_frac": args.train_frac,
                  "d_model": args.d_model, "rank": args.rank,
                  "mlp_dim": args.mlp_dim, "n_heads": args.n_heads}
    archs = args.archs.split(",")
    ops = tuple(args.ops.split(","))
    for arch in archs:
        for seed in range(args.seeds):
            train_run(arch, args.P, ops, args_dict, device, eid, seed, args.out_root)

    print("\n[modarith_power] DONE.", flush=True)


if __name__ == "__main__":
    main()
