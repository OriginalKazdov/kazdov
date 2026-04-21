# Architecture overview

Quick technical reference for MoBE-BCN and its scalable attention kernel.
For the full treatment, see `papers/01-mobe-bcn.md` (algebraic properties)
and `papers/02-kazdov-lm.md` (LM scaling).

## 1. Bilinear composition primitive

The atomic operation is a low-rank bilinear map:

```
BCN(x, y) = U · ((Vᵀ x) ⊙ (Wᵀ y)) + b

where
  x, y ∈ ℝᴰ          (two input embeddings)
  V, W ∈ ℝ^(D×r)     (low-rank projections, shared with U)
  U   ∈ ℝ^(D×r)
  b   ∈ ℝᴰ
  ⊙                  elementwise (Hadamard) product
  r   ≪ D            rank
```

This is equivalent to the CP-decomposed 3-tensor

```
M[a,b,c] = Σ_{ρ=1..r} U[a,ρ] · V[b,ρ] · W[c,ρ]
BCN(x,y)[a] = Σ_{b,c} M[a,b,c] · x[b] · y[c]
```

A finite-group multiplication table is exactly expressible by this form
when `r ≥ |G|` — that is the architectural rank limit discussed in
Paper 1 §6.

Implementation: [`kazdov.kazdov_lm.BilinearComposition`](../kazdov/kazdov_lm.py).

## 2. Mixture of Bilinear Experts (MoBE)

K parallel bilinear primitives with a learned router:

```
w(x, y) = softmax(Router([x; y]))          ∈ ℝᴷ
out     = Σ_k w[k] · BCNₖ(x, y)
```

Each `BCNₖ` has independent `(Uₖ, Vₖ, Wₖ, bₖ)`. Router is a 2-layer MLP.

Implementation: [`kazdov.kazdov_lm.MixtureBilinear`](../kazdov/kazdov_lm.py).

Parameters are stored as stacked tensors `(K, D, r)` so that all K
experts compute in a single batched matmul; there is no Python loop
over experts.

## 3. Causal MoBE-BCN attention — cumsum-factorized O(L)

Naive causal attention with MoBE would compute

```
outᵢ = (1 / i) · Σ_{j ≤ i} MoBE(Qᵢ, Kⱼ)                       # O(L²)
```

but bilinearity lets the sum factor inside the composition:

```
Σ_{j ≤ i} Uₖ ((Vₖᵀ Qᵢ) ⊙ (Wₖᵀ Kⱼ))
  = Uₖ ((Vₖᵀ Qᵢ) ⊙ cumsum_j(Wₖᵀ Kⱼ)ᵢ)                         # O(L)
```

This factorization requires the routing weight `wₖ` to depend only on
`Qᵢ`, not on the `(Qᵢ, Kⱼ)` pair — otherwise the router's softmax
normalization couples `i` and `j` and the cumsum breaks. We therefore
use **query-side routing** (the standard token-level MoE convention).

Resulting kernel:

```python
# Project per-expert  (B, L, K, r)
xV = einsum('bld,kdr->blkr', Q, V_all)
yW = einsum('bld,kdr->blkr', K, W_all)
# Causal cumulative sum along sequence axis
cum_yW = yW.cumsum(dim=1)
# Bilinear combination per position + expert
prod = xV * cum_yW                                              # (B, L, K, r)
# Query-side routing
w = softmax(router(Q), dim=-1)                                  # (B, L, K)
prod_w = prod * w.unsqueeze(-1)
# Stack-project back to D via merged U  (K·r, D) matmul
out = einsum('blkr,kdr->bld', prod_w, U_all)
# Normalize by |valid j ≤ i|
out = out / (arange(1, L+1).view(1, L, 1))
```

Implementation:
[`kazdov.kazdov_lm.CausalMoBEBCNAttention`](../kazdov/kazdov_lm.py).

Memory drops from `O(B·L²·D·K)` to `O(B·L·r·K)`; throughput goes from
~300 tok/s to ~75,000 tok/s on a single RTX 4090 at B=64, L=256, D=512,
K=4 (260× end-to-end, see `experiments/paper2_lm/benchmark.py`).

## 4. Bi-BCN inverse-aware branch

Optional symmetric branch that mimics the group axiom
`(g·h)⁻¹ = h⁻¹ · g⁻¹`:

```
fwd = MoBE(Qᵢ, Kⱼ)                              # V on Qᵢ, W on Kⱼ
inv = MoBE(W_inv·Kⱼ, W_inv·Qᵢ)                  # V on inv(K), W on inv(Q)
out = fwd + α · inv
```

`W_inv ∈ ℝ^(D×D)` is initialized near identity; `α` is a learnable
scalar. Empirically yields an additional ~14-16% sample-efficiency
improvement on group composition tasks.

## 5. Full model — `KazdovLM`

```
KazdovLM(
  vocab_size=50257,
  d_model=512,
  n_layers=12,
  n_heads=8,
  rank=128,
  mlp_dim=2048,
  max_len=256,
  n_experts=4,
  use_mobe=True,
  use_hybrid_mha=True,
)
```

Per-block structure (standard pre-norm transformer):

```
LN → [MoBE-BCN-attention] + [MHA] (parallel, summed)  →  residual
LN → MLP(GELU)                                        →  residual
```

Embeddings are tied with the LM head. Positional embedding is learned
(not RoPE). See `papers/02-kazdov-lm.md` §4 for training recipe.

## 6. Balanced routing

Expert collapse is mitigated with DeepSeek-V3-style gating bias:

```
expert_bias ← expert_bias + η · (1/K − usage_k)        per update
router_logits ← router_logits + expert_bias            per forward
```

`expert_bias` is a buffer (not a parameter; no gradient). Updates are
applied every N training steps via `model.update_mobe_biases()`.
Current usage is exposed through `model.routing_diagnostics()`.
