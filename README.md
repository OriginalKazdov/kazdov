# Kazdov

**Reference implementation and math-specialized language model for MoBE-BCN:
Mixture of Bilinear Experts — Bilinear Composition Network.**

An attention primitive that replaces softmax attention with K parallel
low-rank bilinear experts coordinated by a learned router. Designed to
bake algebraic composition into the architecture itself, rather than
relying on a Transformer to approximate it.

<p>
  <a href="LICENSE"><img alt="License: MIT (code)" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="LICENSE-PAPERS"><img alt="License: CC BY-NC 4.0 (papers)" src="https://img.shields.io/badge/papers-CC%20BY--NC%204.0-lightgrey.svg"></a>
  <a href="papers/01-mobe-bcn.md"><img alt="Paper 1" src="https://img.shields.io/badge/paper%201-MoBE--BCN-green.svg"></a>
  <a href="papers/02-kazdov-lm.md"><img alt="Paper 2" src="https://img.shields.io/badge/paper%202-Kazdov--LM%20%28WIP%29-orange.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="PyTorch 2.3+" src="https://img.shields.io/badge/pytorch-2.3%2B-ee4c2c.svg">
</p>

---

## TL;DR — headline results

**Paper 1 · algebraic composition** (`papers/01-mobe-bcn.md`)

| Task | Transformer baseline | MoBE-BCN | Speedup |
|---|---|---|---|
| S₅ grokking (sample efficiency) | 41,500 steps | 5,500 steps | **7.55×** |
| S₆ grokking | 13,500 steps | 3,500 steps | **3.86×** |
| Power 2022 `a + b mod 97` | never groks (0/3 seeds) | 3,000 steps (3/3 seeds) | **>13×** |
| Universal algebra (S₅, frozen core) | — | **0.99 test acc** | first demonstration |
| NonAssoc8 negative control | — | **0.00 test acc** | passes control |

Matched-parameter comparison. MoBE-BCN at 4.4× *fewer* parameters.

**Paper 2 · math language modeling** (`papers/02-kazdov-lm.md`, *work in progress*)

| Step | Throughput on RTX 4090 | VRAM | 24h token budget |
|---|---|---|---|
| Naive Python loop + pairwise expand | 288 tok/s | 15 GB | 25M |
| + vectorized K-expert kernel | 853 tok/s | 8 GB | 74M |
| + cumsum factorization (O(L)) | 6,006 tok/s | 1.3 GB | 519M |
| + larger batch (memory freed) | **74,654 tok/s** | 19 GB | **6.5B** |

**259× end-to-end** — enables overnight training of a 98M-parameter math
LM on a single consumer GPU. Held-out math loss benchmark vs GPT-2 124M
pending completion of the main training run.

---

## What is MoBE-BCN?

A Transformer's attention approximates bilinear interactions internally
when forced to learn modular arithmetic (Nanda et al., 2023). MoBE-BCN
short-circuits the approximation: the attention primitive is itself a
low-rank bilinear operator, with K of them in parallel and a learned
router:

```
BCNₖ(x, y) = Uₖ · ((Vₖᵀ x) ⊙ (Wₖᵀ y)) + bₖ
MoBE(x, y) = Σₖ softmax(Router(·))[k] · BCNₖ(x, y)
```

- **Low-rank bilinear** — a finite-group multiplication table is
  representable exactly when `rank ≥ |G|`.
- **K experts** — specialization across group families, operations,
  and algebraic substructures.
- **Query-side routing** — unlocks an `O(L)` cumsum-factorized attention
  kernel (Paper 2) while preserving expert specialization.

Full derivation and correctness proof in `docs/architecture.md`.

---

## Papers

Both papers are in this repository. They target complementary stories
that read independently but cite each other.

### Paper 1 — `papers/01-mobe-bcn.md` — **ready to read**

*MoBE-BCN: Mixture of Bilinear Experts for Sample-Efficient Algebraic
Composition Learning*

Introduces the architecture and reports sample-efficiency experiments
on symmetric groups (S₅, S₆, S₇), modular arithmetic (Power 2022),
multi-operation arithmetic, and multi-algebra joint training. The
headline result is the first empirical demonstration of **universal
algebra emergence** via a frozen shared bilinear core, validated by a
non-associative-quasigroup negative control.

### Paper 2 — `papers/02-kazdov-lm.md` — **work in progress**

*Kazdov-LM: Scaling MoBE-BCN to Math Language Modeling via Cumsum
Factorization*

Derives the `O(L)` cumsum-factorized attention kernel that enables
MoBE-BCN at language-model scale, presents the vectorized K-expert
implementation, and reports held-out math-loss benchmarks for a 98M
Kazdov-α model trained on a 5.95B-token math corpus. Final numbers
pending the in-progress training run.

---

## Install

```bash
git clone https://github.com/OriginalKazdov/kazdov.git
cd kazdov
pip install -e .
```

See `docs/install.md` for hardware notes, HuggingFace token setup for
the math datasets (Paper 2 reproduction), and a sanity check.

---

## Quick demo

```python
import torch
from kazdov import KazdovLM

model = KazdovLM(
    vocab_size=50257, d_model=512, n_layers=12, n_heads=8,
    rank=128, mlp_dim=2048, max_len=256, n_experts=4,
    use_mobe=True, use_hybrid_mha=True,
).cuda()

print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

x = torch.randint(0, 50257, (4, 256)).cuda()
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    out = model(x, labels=x)
print(f"loss: {out['loss'].item():.4f}")
```

Or from the CLI, once a checkpoint is available:

```bash
python examples/generate_demo.py path/to/checkpoint.pt
```

---

## Reproducing the papers

### Paper 1 — canonical grokking benchmark

```bash
python experiments/paper1_algebra/run_modarith_power.py \
    --archs transformer,bcn,bcn_mobe --P 97 --seeds 3 --epochs 40000
```

### Paper 2 — train Kazdov-α

```bash
export HF_TOKEN="hf_..."
python experiments/paper2_lm/prep_data.py  --full --out_dir data/alpha    # ~2.5h
python experiments/paper2_lm/benchmark.py                                   # throughput table
python experiments/paper2_lm/train_alpha.py --total_tokens 5000000000       # ~20-24h
python experiments/paper2_lm/eval_heldout.py --model kazdov:results/kazdov_alpha/final.pt
python experiments/paper2_lm/test_kazdov.py "The derivative of x^2 is"
```

Full instructions, dataset licenses, and hardware notes in
`experiments/paper2_lm/README.md`.

---

## Repository layout

```
kazdov/
├── kazdov/                     # installable Python package (`pip install -e .`)
│   ├── kazdov_lm.py            # core: MixtureBilinear + CausalMoBEBCNAttention + KazdovLM
│   └── schedulers.py           # WSD / cosine LR lambdas
├── papers/
│   ├── 01-mobe-bcn.md          # algebraic-composition paper (complete)
│   └── 02-kazdov-lm.md         # LM-scaling paper (WIP, updated post-training)
├── experiments/
│   ├── paper1_algebra/         # Power 2022 replication, universal algebra
│   └── paper2_lm/              # data prep, training, eval, inference demo
├── configs/
│   └── alpha.yaml              # canonical Kazdov-α config
├── examples/
│   └── generate_demo.py
├── docs/
│   ├── architecture.md         # technical reference
│   └── install.md
├── LICENSE                     # MIT — code
├── LICENSE-PAPERS              # CC BY-NC 4.0 — papers
├── CITATION.cff
└── pyproject.toml
```

---

## How to cite

**Paper 1 (algebraic composition):**

```bibtex
@article{dovzak2026mobebcn,
  title   = {MoBE-BCN: Mixture of Bilinear Experts for
             Sample-Efficient Algebraic Composition Learning},
  author  = {Dovzak, Juan Cruz},
  year    = {2026},
  month   = {April},
  note    = {Preprint. https://github.com/OriginalKazdov/kazdov},
}
```

**Paper 2 (LM scaling, once complete):**

```bibtex
@article{dovzak2026kazdov,
  title   = {Kazdov-LM: Scaling MoBE-BCN to Math Language Modeling
             via Cumsum Factorization},
  author  = {Dovzak, Juan Cruz},
  year    = {2026},
  note    = {In preparation. https://github.com/OriginalKazdov/kazdov},
}
```

A machine-readable `CITATION.cff` is provided at the repository root.

---

## License

- **Code** (`kazdov/`, `experiments/`, `examples/`, `configs/`, `docs/`)
  — MIT License (see `LICENSE`). Free for academic and commercial use,
  modification, and redistribution with attribution.
- **Papers** (`papers/`) — Creative Commons Attribution-NonCommercial 4.0
  (see `LICENSE-PAPERS`). Share and adapt with attribution; no
  commercial use without explicit written permission.
- **Patent notice** — the MoBE-BCN architecture, cumsum-factorized
  bilinear attention, and related methods may be subject to patent
  protection. The CC and MIT licenses above do **not** grant any patent
  rights; commercial applications should contact the author.

---

## Contact & contributions

Author: **Juan Cruz Dovzak** — independent researcher, Argentina.
Email: [juandovzak@gmail.com](mailto:juandovzak@gmail.com)

- **Issues / questions:** open a GitHub issue.
- **Collaboration, replication, scaling to 1B+ parameters:** email is
  open; I am especially interested in feedback from researchers who
  work on grokking, linear attention, or algebraic reasoning.
- **Pull requests:** welcome. For substantial changes please open an
  issue first to discuss scope.

This project is developed by a solo researcher without institutional
support. If you find it useful, a citation, a ⭐, or a mention on
social media materially helps.
