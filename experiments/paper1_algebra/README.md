# Paper 1 — Algebraic composition experiments

Reproduces the headline results of `papers/01-mobe-bcn.md`.

## `run_modarith_power.py`

Replicates Power et al. (2022) `a + b mod 97` grokking with three
matched-parameter architectures: Transformer baseline, single-bilinear
BCN-LM, and MoBE-BCN-LM (K=4). Self-contained — no dependencies beyond
torch.

```bash
python experiments/paper1_algebra/run_modarith_power.py \
    --archs transformer,bcn,bcn_mobe \
    --P 97 \
    --seeds 3 \
    --epochs 40000
```

Expected result (median over 3 seeds):

| Architecture | Grok step | Robustness |
|---|---|---|
| Transformer | NEVER (plateau 0.77-0.93) | 0/3 |
| BCN-LM (K=1) | 7,000 | 2/3 |
| **MoBE-BCN-LM (K=4)** | **3,000** | **3/3** |

Output is written to `results/` as JSON with per-seed trajectories.

## Universal-algebra emergence

The multi-algebra joint-training experiment (Section 4 of the paper)
uses scaffolding that depends on internal tooling (algebra factories,
per-algebra embedding management). A standalone version is planned for
a later release; in the meantime, the frozen-core transfer protocol is
documented in the paper's Section 4.1-4.2 and the MoBE-BCN core is
available in `kazdov/kazdov_lm.py::MixtureBilinear`.
