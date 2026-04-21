"""Kazdov: MoBE-BCN reference implementation.

Main entry points:
  - `kazdov.kazdov_lm.KazdovLM`        autoregressive LM with MoBE-BCN attention
  - `kazdov.kazdov_lm.MixtureBilinear` K-expert bilinear primitive
  - `kazdov.kazdov_lm.CausalMoBEBCNAttention` cumsum-factorized O(L) attention
  - `kazdov.schedulers.make_wsd_lambda` warmup-stable-decay LR schedule

See `papers/01-mobe-bcn.md` for the architecture paper and
`papers/02-kazdov-lm.md` for the scaling paper.
"""

from .kazdov_lm import (
    KazdovLM,
    KazdovBlock,
    BilinearComposition,
    MixtureBilinear,
    CausalMoBEBCNAttention,
    HybridCausalMoBEAttention,
    count_params,
)
from .schedulers import make_wsd_lambda, make_cosine_lambda

__version__ = "0.1.0"
__all__ = [
    "KazdovLM",
    "KazdovBlock",
    "BilinearComposition",
    "MixtureBilinear",
    "CausalMoBEBCNAttention",
    "HybridCausalMoBEAttention",
    "count_params",
    "make_wsd_lambda",
    "make_cosine_lambda",
]
