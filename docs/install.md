# Installation

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.3 (bf16 autocast APIs). Tested with 2.6 on CUDA 12.4.
- For training / evaluation: `transformers` (GPT-2 tokenizer), `datasets`
  (HuggingFace streaming), `numpy`.
- For the benchmark script: `matplotlib` (optional).

## Install from source

```bash
git clone https://github.com/OriginalKazdov/kazdov.git
cd kazdov
pip install -e .
```

This installs the `kazdov` Python package in editable mode. Dependencies
are listed in `pyproject.toml` (to be published alongside the first
tagged release).

Quick sanity check:

```bash
python -c "from kazdov import KazdovLM; \
    m = KazdovLM(vocab_size=50257, d_model=128, n_layers=2, n_heads=4, \
                 rank=32, mlp_dim=256, max_len=64, use_mobe=True); \
    print(f'OK: {sum(p.numel() for p in m.parameters())/1e6:.2f}M params')"
```

Expected output: `OK: ~7M params`.

## Hardware

The LM experiments target a single consumer GPU. Verified working:

- **RTX 4090** 24 GB (reference) — ~60-70K tok/s at 98M params, L=256, B=32
- **A100** 40/80 GB — untested but expected ~2× faster throughput
- **H100** — untested but expected ~3-4× faster

Algebraic-composition experiments (Paper 1) are small enough to run on
CPU or any modest GPU (RTX 20xx / M-series GPUs).

## Data preparation (Paper 2 reproduction only)

Downloading the 5.95B-token math corpus requires a HuggingFace account
and acceptance of the terms-of-use on:

- [`GAIR/MathPile_Commercial`](https://huggingface.co/datasets/GAIR/MathPile_Commercial)
- [`open-web-math/open-web-math`](https://huggingface.co/datasets/open-web-math/open-web-math)

After accepting terms, set `HF_TOKEN` in your environment and run:

```bash
export HF_TOKEN="hf_..."
python experiments/paper2_lm/prep_data.py --full --out_dir data/alpha
```

Approximate disk usage: 14 GB (tokenized `.bin` shards; raw data streams
from HuggingFace cache, ~5-10 GB additional during first run).
