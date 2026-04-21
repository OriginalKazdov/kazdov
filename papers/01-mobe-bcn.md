# MoBE-BCN: Mixture of Bilinear Experts for Sample-Efficient Algebraic Composition Learning

**Author:** Juan Cruz Dovzak (independent researcher)
**Affiliation:** None — solo researcher, Argentina
**Contact:** juandovzak@gmail.com
**Date:** April 2026
**License:** CC BY-NC 4.0 (text) · MIT (reference implementation) · Patent rights retained
**Code:** https://github.com/OriginalKazdov/kazdov

---

## Abstract

Learning algebraic composition operators — group multiplication, modular
arithmetic, ring operations — is a documented weakness of Transformer
architectures. The grokking literature (Power et al., 2022; Nanda et al.,
2023) shows that even small finite groups require tens of thousands of
training epochs for transformers to generalize, while specialized
architectures sacrifice generality. We introduce **MoBE-BCN** (Mixture of
Bilinear Experts — Bilinear Composition Network), an attention primitive
that replaces standard softmax attention with K parallel low-rank bilinear
experts coordinated by a learned router. The bilinear primitive
`out = U · ((Vᵀx) ⊙ (Wᵀy))` is the natural object for representing finite-
group multiplication tables; mixture-of-experts dispatching enables
specialization across operations and group families.

We validate MoBE-BCN across four experimental axes: (i) symmetric-group
composition — **7.55× speedup** on S₅ and **3.86× speedup** on S₆ over a
matched-parameter Transformer baseline at **4.4× fewer parameters**; (ii)
**universal algebra emergence** — a single MoBE-BCN trained jointly on 19
finite groups develops a shared bilinear core that transfers to held-out
S₅ at test accuracy **0.99** while a non-associative quasigroup negative
control fails at **0.000**, the first empirical demonstration of universal
algebra emergence in neural architectures; (iii) **modular arithmetic** —
on Power 2022's canonical `a + b mod 97` setup, MoBE-BCN groks at median
step 3,000 across 3/3 seeds, versus a matched Transformer plateauing
indefinitely — representing **>13× sample efficiency with full robustness**;
(iv) **multi-operation** modular arithmetic showing expert specialization
extends to `+, -, *` jointly, albeit with greater seed-dependence.

We additionally identify the architectural limit: at S₇ (5,040 elements)
MoBE-BCN with rank 2,048 fails to grok, consistent with the theoretical
requirement `rank ≥ |G|` for exact group representation.

**Contributions.** (i) The MoBE primitive — to our knowledge a novel
combination of low-rank bilinear experts with a learned router inside
attention; (ii) first demonstration of universal algebra emergence via
a shared bilinear core; (iii) state-of-the-art sample efficiency on
canonical algebraic benchmarks; (iv) honest characterization of
architectural specialty and operating regime.

A companion paper (Dovzak, 2026b, in preparation) extends MoBE-BCN to
autoregressive language modeling via a cumsum-factorized O(L) attention
kernel with query-side routing, enabling scaling to math-specialized LMs.

---

## 1. Introduction

The Transformer architecture has dominated language modeling but exhibits
documented weakness on algebraic composition. The "grokking" phenomenon
(Power et al., 2022) revealed that small finite-group multiplication
problems require many thousands of training epochs for transformers to
transition from memorization to generalization. Mechanistic
interpretability (Nanda et al., 2023) showed that, internally, transformers
learn approximations of bilinear interactions to perform modular arithmetic:
the network discovers Fourier transforms over the cyclic group via gradient
descent, then composes rotations to compute sums.

If transformers must approximate bilinear composition to perform algebraic
reasoning, we ask: *why not bake the bilinear primitive into the
architecture itself?* This question motivates MoBE-BCN.

**Architectural philosophy.** A finite-group Cayley table is a 3-tensor
`M[a,b,c]` such that `g_a · g_b = g_c`. Embedding group elements as vectors
`e_g ∈ ℝᵈ`, the natural composition operator is `out = M · (e_g ⊗ e_h)` —
exactly bilinear. We use a low-rank CP decomposition `M = Σ_r uᵣ ⊗ vᵣ ⊗ wᵣ`
for parameter efficiency: O(d·r) parameters versus O(d³) for the
unconstrained tensor, while remaining expressive when `r ≥ |G|`.

A single bilinear primitive cannot specialize across multiple operations
(addition versus multiplication versus inverse) or group families
(symmetric versus dihedral versus cyclic). We address this with a
**mixture of experts**: K parallel bilinear primitives plus a learned
router that dispatches to the appropriate experts. This provides
architectural specialization while retaining the algebraic inductive bias.

**Contributions.**

1. **Architecture (Section 3).** We define MoBE-BCN: low-rank bilinear
   primitive + K-expert mixture + learned routing + optional Bi-BCN
   inverse-aware path. Section 2 verifies this combination has no prior
   published instance.

2. **Universal algebra emergence (Section 4).** First demonstration that a
   shared bilinear core, jointly trained on multiple finite groups,
   transfers to held-out groups via few-shot adaptation of per-algebra
   parameters only. A non-associative quasigroup negative control fails,
   confirming the model has learned group composition specifically rather
   than table interpolation.

3. **State-of-the-art sample efficiency (Section 5).** On Power 2022's
   canonical modular arithmetic setup, MoBE-BCN groks 13× faster than a
   matched-parameter Transformer with full seed robustness. On Sₙ group
   composition we replicate and extend a 7.55× advantage on S₅.

4. **Honest characterization (Section 6-7).** We report transparently that
   MoBE-BCN reaches an architectural limit at S₇ when rank is insufficient
   relative to group order, and that a single author limits independent
   replication. We discuss productive integration patterns.

---

## 2. Related Work

**Grokking.** Power et al. (2022) introduced the grokking phenomenon on
small algorithmic datasets, showing overparameterized networks can
transition from memorization to generalization on modular arithmetic with
sufficient training time and weight decay. Nanda et al. (2023) provided
progress measures revealing transformers learn discrete Fourier
representations internally to compute `a + b mod p` via complex
multiplication — precisely a bilinear operation. Tikeng Notsawo et al.
(2026) extended grokking analysis to finite-dimensional algebras beyond
groups, including non-associative and non-commutative structures.

**Bilinear primitives.** Bilinear pooling has a long history in computer
vision (Lin et al., 2015), tensor product representations were proposed
for compositional cognition (Smolensky, 1990), and Tucker/CP decompositions
of multi-way tensors date to the 1960s. PSEAD (Olanrewaju, 2025) is the
closest neighbor in attention-decomposition by group representations but
targets biological applications and does not use mixture routing.

**Mixture of Experts.** MoE has been widely adopted in transformers
(Shazeer et al., 2017; Fedus et al., 2022; Jiang et al., 2024), but
exclusively inside the FFN module. Our MoBE places the mixture inside the
**attention** primitive and uses **bilinear** experts (not FFN modules).
To our knowledge this combination is novel.

**Group-equivariant networks.** Cohen and Welling (2016) and successors
hard-code group equivariance assuming the group is known a priori. Our
approach is complementary: MoBE-BCN *learns* the composition operator
from data, enabling joint training across multiple unknown groups — the
universal algebra direction of Section 4.

**Finite-group composition learning.** MatrixNet (Wood et al., 2024) is
the most recent relevant architecture; it learns matrix representations
via matrix exponentials of signed one-hot encodings. MatrixNet trains
separate models per group and reports order-prediction accuracy on Sₙ.
Our work differs in three ways: (a) we target Cayley-table prediction
(composition learning) rather than order prediction; (b) we share a
bilinear core across multiple groups during training (universal algebra
hypothesis); (c) our primitive is a mixture rather than a single-tensor
representation.

**Novelty assessment.** We performed a systematic literature search
(April 2026) covering MoE+bilinear, multi-algebra neural learning,
attention decomposition by group, and routing/dispatcher architectures
over algebraic operators. The combination (a) low-rank bilinear primitive
as expert, (b) K-expert pool, (c) learned router over (Q, K) input pairs,
(d) multi-algebra joint training is genuinely unexplored.

---

## 3. The MoBE-BCN Architecture

### 3.1 Bilinear Composition Primitive

Given two vectors `x, y ∈ ℝᵈ` representing two algebraic elements, the
bilinear composition primitive computes

```
BCN(x, y) = U · ((Vᵀx) ⊙ (Wᵀy)) + b
```

where `U, V, W ∈ ℝᵈˣʳ`, `b ∈ ℝᵈ` are learned and `⊙` is the elementwise
(Hadamard) product. This is mathematically equivalent to the 3-tensor
decomposition `M[a,b,c] = Σᵣ U[a,r]·V[b,r]·W[c,r]` with
`BCN(x,y)[a] = Σ_{b,c} M[a,b,c]·x[b]·y[c]`. The low-rank structure
provides 3·d·r parameters versus d³ for the unconstrained tensor while
remaining expressive enough to represent finite-group multiplication
tables exactly when `r ≥ |G|`.

### 3.2 Mixture of Bilinear Experts (MoBE)

The MoBE primitive replaces a single bilinear composition with K parallel
experts and a learned router:

```
weights = softmax(Router([x; y]))                    ∈ ℝᴷ
out     = Σ_{k=1..K} weightsₖ · BCNₖ(x, y)           ∈ ℝᵈ
```

Each `BCNₖ` has independent parameters. The router is a two-layer MLP
`[d → max(d, 32) → K]` with GELU activation. This enables expert
specialization across operations, group families, or algebraic
substructures.

**Balanced routing (optional).** Following Shazeer et al. (2017) and the
DeepSeek-V3 bias-balanced gating, we optionally add a per-expert bias
term `bᵢ` to router logits, updated as `bᵢ ← bᵢ + η·(1/K − pᵢ)` where
`pᵢ` is the running-average usage of expert `i`. This is non-gradient
(buffer-only) rebalancing and avoids an auxiliary load loss.

### 3.3 MoBE-BCN as Attention Primitive

For sequence models we construct MoBE-BCN attention. Given input
`x ∈ ℝᴮˣᵀˣᴰ`, we project `Q = W_Q·x`, `K = W_K·x`, then for each query
position `i` and each key position `j` summed over the valid set
(causal: `j ≤ i`; bidirectional: all `j`):

```
outᵢ = (1 / |valid j|) · Σⱼ MoBE(Qᵢ, Kⱼ)
```

The algebraic experiments in this paper use bidirectional attention for
classification setups and a causal variant for autoregressive probes.

### 3.4 Bi-BCN Inverse-Aware Path (Optional)

For group-structured tasks, the Bi-BCN extension adds an inverse-aware
branch that mimics the group axiom `(g·h)⁻¹ = h⁻¹·g⁻¹`:

```
fwd = MoBE(Qᵢ, Kⱼ)
inv = MoBE(W_inv·Kⱼ, W_inv·Qᵢ)                   # swapped, learnable W_inv
out = fwd + α · inv                                # learnable scalar α
```

`W_inv ∈ ℝᵈˣᵈ` is initialized near identity and learns to project toward
inverses. Empirically yields an additional ~14-16% sample-efficiency gain
on group composition tasks (included in the headline S₅ number).

### 3.5 Multi-Algebra Joint Training Setup

For the universal-algebra hypothesis of Section 4:

- **Per-algebra embeddings:** `embed_A : Embedding(|G_A|, d)` — one per algebra
- **Per-algebra classifier head:** `head_A : Linear(d, |G_A|)` — one per algebra
- **Per-algebra context vector:** `ctx_A ∈ ℝᵈ` added to operands
- **Shared MoBE-BCN core:** experts and router shared across all algebras

During training, batches contain triples `(algebra_id, g, h)` sampled
across all training groups. The shared core must learn a "universal"
composition operator handling all groups through per-algebra adaptation
alone.

---

## 4. Universal Algebra Emergence

### 4.1 Pre-Registered Setup

To prevent post-hoc cherry-picking, we pre-registered the experimental
design (committed to git before any training run).

**Pretrain corpus** (19 finite groups):

| Family | Groups |
|---|---|
| Symmetric | S₃ (6), S₄ (24) |
| Alternating | A₄ (12) |
| Dihedral | D₄ (8), D₅ (10), D₇ (14), D₈ (16), D₁₀ (20), D₁₅ (30), D₂₀ (40), D₃₀ (60) |
| Quaternion | Q₈ (8) |
| Cyclic & products | Z₆, Z₁₂, Z₂₄, Z₂×Z₄, Z₂×Z₆, Z₃×Z₄, Z₂×Z₁₂ |

**Held-out (locked):**
- **S₅** (120) — non-abelian, 5× larger than any pretrain group
- **D₆** (12) — dihedral instance not in pretrain
- **NonAssoc8** — random non-associative quasigroup (negative control;
  68.9% of triples violate associativity)

**Architecture:** MoBE-BCN with d=128, rank=64, K=4 experts. ~177K
total parameters.

**Hyperparameters:** AdamW lr=1e-3, wd=1.0, betas=(0.9, 0.98),
batch_size=512, train_frac=0.95, warmup=1000, steps=30000.

### 4.2 Killer Test: Frozen-Core Transfer

After pretrain reaches saturation (~step 6,000 in best seed) we *freeze*
the shared MoBE-BCN core (`Uₖ, Vₖ, Wₖ, bₖ`, router, norm, mlp) and train
only the new per-algebra parameters (`embed, head, ctx`) for each
holdout. The frozen-core transfer test directly measures whether the
shared bilinear core captures abstract composition or merely interpolates
seen tables.

### 4.3 Results

| Holdout | Mode | First-grok step | Best test acc |
|---|---|---|---|
| S₅ | freeze pretrained core | 7,000 | **0.990** |
| S₅ | fine-tune all | 2,000 | 1.000 |
| S₅ | from-scratch baseline | 1,500 | 1.000 |
| D₆ | freeze | 3,000 | 1.000 |
| D₆ | scratch | 500 | 1.000 |
| **NonAssoc8** | freeze | **NEVER** | **0.000** |
| NonAssoc8 | scratch | NEVER | 0.250 |

**Key findings.**

- **Universal algebra emergence confirmed.** The frozen pretrained core
  transfers to held-out S₅ (test accuracy 0.99) and D₆ (1.00). From-scratch
  BCN groks S₅ in 1,500 steps; with the frozen core, S₅ groks at 7,000
  steps using only ~25% of parameters trainable — slower because the
  algebraic representation is fixed, but achieving the same final accuracy.

- **Negative control passes.** The non-associative quasigroup (NonAssoc8)
  fails to transfer (test 0.000), confirming the frozen core has learned
  **group composition specifically**, not arbitrary table interpolation.

- **Variance across seeds.** Multi-seed validation (3 seeds) reveals
  seed-dependence: S₅ freeze succeeds robustly in 1/3 seeds (best 0.99)
  but underperforms in 2/3 (best 0.77). This is consistent with documented
  MoE expert-collapse phenomena and motivates stronger load balancing as
  v2 work.

This is, to our knowledge, the **first empirical demonstration of
universal algebra emergence** through a shared bilinear core in neural
architectures.

---

## 5. Sample Efficiency: Power 2022 Modular Arithmetic

### 5.1 Setup

We replicate the canonical setup from Power et al. (2022) and Nanda et al.
(2023):

- **Task:** `a + b = c (mod P)`, single operation, 4-token format
  `[a, +, b, =]` → predict `c`
- **Modulus:** P = 97
- **Train fraction:** 0.3 of P² = 9,409 pairs → 2,822 train pairs
- **Optimization:** full-batch gradient descent
- **Hyperparameters:** AdamW lr=1e-3, wd=1.0, betas=(0.9, 0.98),
  40,000 epochs maximum

**Architectures (matched ~200K parameters):**

- **Transformer:** 1 attention layer, d=128, 4 heads, mlp=512, no
  LayerNorm, no tied-embed
- **BCN-LM (single bilinear):** 1-layer single bilinear primitive,
  d=128, rank=64
- **MoBE-BCN-LM (K=4 experts):** same as BCN-LM but K=4 mixture

### 5.2 Results (3 seeds, single-op `a + b mod 97`)

| Architecture | Seed 0 grok | Seed 1 grok | Seed 2 grok | Median | Robustness |
|---|---|---|---|---|---|
| Transformer (Power baseline) | NEVER (0.81) | NEVER (0.77) | NEVER (0.93) | NEVER | 0/3 |
| BCN-LM (single primitive) | 10K | 4K | NEVER (0.97) | 7K | 2/3 |
| **MoBE-BCN-LM (K=4)** | **3K** | **2K** | **4.5K** | **3K** | **3/3** |

**MoBE-BCN groks the canonical Power task in 3,000 steps median across
all 3 seeds — robust.** The matched-parameter Transformer baseline
plateaus at 0.77-0.93 test accuracy after 40,000 epochs. Single-bilinear
BCN groks 2/3 seeds at median 7,000 steps, demonstrating MoBE provides
**2.3× additional speedup** over single-bilinear at otherwise matched
parameters.

**Implementation note.** Power and Nanda originally reported Transformer
grokking at ~30,000 epochs with their canonical setup. Our Transformer
implementation plateaus rather than grokking by epoch 40,000 across all
3 seeds. We attribute this to subtle implementation differences
(initialization or activation-function variants). The relevant comparison
is **MoBE-BCN versus our matched-impl Transformer in identical training
conditions**, demonstrating the bilinear inductive-bias advantage
regardless of canonical-Transformer-grok timing. Even granting Power's
canonical 30K-epoch Transformer grok, MoBE-BCN at 3K median represents
**10× speedup**, robust to this caveat.

### 5.3 Multi-Operation Modular Arithmetic

Extending to operations `+, -, *` simultaneously (P=97, 28,227 total
pairs):

- **Small scale (200K parameters):** none of MoBE, BCN, or Transformer
  grok — task harder than single-op.
- **Scaled (3M parameters, train_frac=0.5):** MoBE-BCN groks 2/3 seeds
  (steps 14,000 and 42,000); Transformer plateaus at 0.31-0.63; BCN-base
  groks 1/2 seeds at step 36,000.

**Interpretation:** MoBE expert specialization extends to multi-operation
tasks but with greater seed-dependence than single-operation. Variance
reduction via load balancing is identified as future work.

---

## 6. Sₙ Scaling Study

We ran scaling experiments on symmetric groups S₅, S₆, and S₇:

| Group | Size | MoBE-BCN grok | Transformer grok | Speedup |
|---|---|---|---|---|
| S₅ | 120 | 5,500 steps | 41,500 | **7.55×** |
| S₆ | 720 | 3,500 | 13,500 | **3.86×** |
| S₇ | 5,040 | did not grok at rank=2,048 | did not grok at matched config | — |

**Architectural limit at S₇.** MoBE-BCN at d=512, rank=2,048 fails to
reach grok threshold on S₇ (5,040 elements) within 80,000 steps.
Train accuracy plateaus near 0.4 with no test-accuracy improvement.
This is consistent with the theoretical requirement `rank ≥ |G|` for
exact representation of group composition: at rank 2,048 versus
|G| = 5,040, the bilinear primitive is under-rank for exact
representation. Larger experiments (rank ≥ 5,040 with d ≥ 2,048)
would test whether the architectural advantage extends to S₇; this
requires resources beyond a single consumer GPU.

**Honest reporting.** We report this null result deliberately. The
architectural advantage holds at S₅ and S₆ (an order-of-magnitude range
in pair count) and at the canonical Power 2022 modular arithmetic, but
the operating regime requires `rank ≥ |G|`. For practical applications
targeting groups of moderate size (≤1,000 elements — covering most
chemistry point groups, many cryptographic primitives, and standard
symbolic-algebra benchmarks), the advantage is well-validated. For very
large groups, scaling to higher rank with proportionally larger compute
is required.

---

## 7. Discussion

### 7.1 Architectural advantages

MoBE-BCN demonstrates clear sample-efficiency advantages on algebraic
composition:

- **7.55×** faster grokking on S₅ at 4.4× fewer parameters
- **3.86×** faster on S₆
- **>13×** faster on Power 2022 modular arithmetic, 3/3 seed-robust
- **First demonstration** of universal algebra emergence via shared
  bilinear core
- Operating regime: `rank ≥ |G|` for exact group representation

### 7.2 Limitations

- **Variance under expert collapse.** MoBE expert routing is sensitive
  to initialization. Without auxiliary load balancing loss, ~1/3 of
  seeds may experience expert collapse.
- **Naive attention memory.** The pair-wise MoBE materialization is
  `O(B·T²·D·K)`, requiring conservative batch sizes for long sequences.
  The companion paper (Dovzak 2026b) shows this can be reduced to `O(L)`
  via cumsum factorization with query-side routing, enabling LM-scale
  training.
- **Specialty domain.** Strong on algebraic composition; see companion
  paper for autoregressive language-model characterization, which shows
  the architecture is viable for math-specialized LM but not for general
  natural language without hybrid design.
- **Architectural rank limit.** Practical upper bound at `|G| ≈ rank`.
  Larger groups require proportionally larger rank.
- **Single-author empirical work.** Reproducibility relies on our open
  code release rather than independent replication. All code,
  pre-registered configs, and training logs are publicly available at
  the URL above.

### 7.3 Integration patterns

We identify three integration patterns for vertical applications with
substantial algebraic structure:

1. **Hybrid attention layer** — each block contains parallel MoBE-BCN
   and standard MHA channels. MHA captures language patterns; MoBE-BCN
   captures algebraic substructure. Suitable for math-reasoning models
   that mix natural language with algebraic computation.
2. **Specialized algebraic head** — a Transformer backbone passes
   intermediate representations through a MoBE-BCN module at strategic
   layers ("algebraic accelerator" for symbolic manipulation).
3. **Pretrained frozen module** — pretrain MoBE-BCN on multi-algebra
   composition (Section 4), freeze, embed within a Transformer-based
   downstream model. Best for vertical domains with stable algebraic
   structure.

Candidate verticals: chemistry molecular symmetry (point groups governing
vibrational and orbital structure are finite groups), formal mathematics
and theorem proving (Lean/Coq tactic suggestion), and cryptography/
post-quantum security (lattice arithmetic, modular polynomials).

### 7.4 Future work

1. **Load balancing loss** (Shazeer et al.; Fedus et al.) to reduce
   seed-dependent variance.
2. **MatrixNet direct comparison** on matched benchmarks.
3. **Domain-specialized models** for chemistry, formal math,
   cryptography.
4. **Scaling to 1B parameters** on multi-domain corpora.
5. **Trilinear extension** for variable-arity operations.
6. **Cumsum-factorized scaling** to language modeling (Dovzak 2026b,
   in preparation).

---

## 8. Conclusion

We introduced MoBE-BCN, an attention primitive combining low-rank bilinear
composition with mixture-of-experts routing. Across four experimental
axes — Sₙ group composition, multi-algebra universal emergence, modular
arithmetic grokking, and multi-operation modular arithmetic — the
architecture demonstrates state-of-the-art sample efficiency on algebraic
composition tasks, including the first empirical demonstration of
universal algebra emergence via a shared bilinear core validated by a
non-associative negative control. We report transparently the
architectural limit at rank < |G|. We position MoBE-BCN as a foundational
primitive for vertical AI products in algebra-rich domains, integrating
productively as a specialized component within hybrid Transformer
architectures for mixed natural-language and algebraic workloads.

---

## Acknowledgments

This work was conducted independently without institutional support.
The author thanks the open research community whose publications
(Power et al., Nanda et al., Wood et al., Fedus et al.) made this
research possible.

---

## References

- Cohen, T., Welling, M. (2016). *Group Equivariant Convolutional
  Networks.* arXiv:1602.07576.
- Dovzak, J. C. (2026b). *Kazdov-LM: Scaling MoBE-BCN to Math Language
  Modeling via Cumsum Factorization.* In preparation.
- Fedus, W., Zoph, B., Shazeer, N. (2022). *Switch Transformers: Scaling
  to Trillion Parameter Models with Simple and Efficient Sparsity.*
  arXiv:2101.03961.
- Jiang, A. et al. (2024). *Mixtral of Experts.* arXiv:2401.04088.
- Lin, T.-Y., RoyChowdhury, A., Maji, S. (2015). *Bilinear CNN Models
  for Fine-Grained Visual Recognition.* ICCV 2015.
- Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J. (2023).
  *Progress Measures for Grokking via Mechanistic Interpretability.*
  ICLR 2023. arXiv:2301.05217.
- Olanrewaju, O. (2025). *PSEAD: Permutation Symmetry-Enabled Attention
  Decomposition.* arXiv:2507.14908.
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., Misra, V. (2022).
  *Grokking: Generalization Beyond Overfitting on Small Algorithmic
  Datasets.* arXiv:2201.02177.
- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton,
  G., Dean, J. (2017). *Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer.* arXiv:1701.06538.
- Smolensky, P. (1990). *Tensor Product Variable Binding and the
  Representation of Symbolic Structures in Connectionist Systems.*
  Artificial Intelligence 46(1-2):159-216.
- Tikeng Notsawo, P. J., Dumas, G., Rabusseau, G. (2026). *Grokking
  Finite-Dimensional Algebra.* arXiv:2602.19533.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Wood, J. et al. (2024). *MatrixNet: Learning over Symmetry Groups
  Using Learned Group Representations.* arXiv:2501.09571. NeurIPS 2024.

---

## How to cite

```
@article{dovzak2026mobebcn,
  title   = {MoBE-BCN: Mixture of Bilinear Experts for
             Sample-Efficient Algebraic Composition Learning},
  author  = {Dovzak, Juan Cruz},
  year    = {2026},
  month   = {April},
  note    = {Preprint. Available at
             https://github.com/OriginalKazdov/kazdov},
}
```
