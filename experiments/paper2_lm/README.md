# Paper 2 — Kazdov-LM math language-model experiments

Reproduces the training setup and benchmarks of `papers/02-kazdov-lm.md`.

## End-to-end pipeline

```bash
# 1. Prepare the 5.95B-token math corpus (~2.5h, ~14 GB on disk)
export HF_TOKEN="<your_huggingface_token>"     # gated datasets need this
python experiments/paper2_lm/prep_data.py --full --out_dir data/alpha

# 2. Throughput benchmark (optional, shows the 259× speedup)
python experiments/paper2_lm/benchmark.py

# 3. Main training run (~20-24h on a single RTX 4090)
python experiments/paper2_lm/train_alpha.py \
    --total_tokens 5000000000 \
    --out_dir results/kazdov_alpha

# 4. Held-out loss vs baseline
python experiments/paper2_lm/eval_heldout.py --model kazdov:results/kazdov_alpha/final.pt
python experiments/paper2_lm/eval_heldout.py --model hf:openai-community/gpt2

# 5. Interactive prompt → generation
python experiments/paper2_lm/test_kazdov.py "The derivative of x^2 is"
```

## Files

| File | Purpose |
|---|---|
| `prep_data.py` | Stream + tokenize 5 math datasets to `.bin` shards |
| `benchmark.py` | Micro-benchmark showing naive vs cumsum MoBE speedup |
| `train_alpha.py` | Main training loop (WSD scheduler, checkpoints every 2000 steps) |
| `eval_heldout.py` | Held-out LM loss vs. GPT-2 124M / other HF models |
| `test_kazdov.py` | Load latest checkpoint, generate from a prompt |

## Data sources

The math corpus is a weighted mix of public datasets. All downloads
happen the first time you run `prep_data.py`:

| Weight | Dataset | Notes |
|---|---|---|
| 45% | `open-web-math/open-web-math` | Math-dense web corpus (ICLR 2024) |
| 22% | `hoskinson-center/proof-pile` | Lean + Isabelle + Coq + math papers |
| 15% | `GAIR/MathPile_Commercial` | ProofWiki + StackExchange (CC BY-SA 4.0) |
| 10% | `math-ai/AutoMathText` (`web-0.70-to-1.00`) | Top-quality slice |
| 8% | `AI-MO/NuminaMath-CoT` | Olympiad problem-solutions |

`MathPile_Commercial` and `OpenWebMath` require accepting terms on
HuggingFace; `HF_TOKEN` env var must be set for streaming.

## Checkpoint naming

`train_alpha.py` writes:

- `results/kazdov_alpha/latest.pt` — overwritten every 2000 steps
- `results/kazdov_alpha/step_<N>_tok_<M>M.pt` — named checkpoints at
  100M / 500M / 1B / 2B / 5B token milestones
- `results/kazdov_alpha/metrics.jsonl` — one JSON line per log event
- `results/kazdov_alpha/config.json` — snapshot of run config
- `results/kazdov_alpha/final.pt` — end-of-run checkpoint
- `results/kazdov_alpha/final_eval.json` — held-out loss at end of run

Resume with `--resume results/kazdov_alpha/latest.pt`.
