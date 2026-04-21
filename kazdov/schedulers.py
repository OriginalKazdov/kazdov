"""Learning rate schedulers for training.

WSD (Warmup-Stable-Decay): better than cosine for small LMs, supports continued
training. From Hu et al. and used in MiniCPM, SmolLM training recipes.
"""

import torch
from typing import Callable


def make_wsd_lambda(warmup_steps: int, total_steps: int,
                     decay_frac: float = 0.2, final_lr_frac: float = 0.1) -> Callable:
    """Returns a lambda for torch.optim.lr_scheduler.LambdaLR.

    Phases:
      [0, warmup_steps): linear warmup from 0 to peak
      [warmup_steps, decay_start): stable at peak
      [decay_start, total_steps]: linear decay from peak to peak * final_lr_frac

    Args:
      warmup_steps: linear warmup duration
      total_steps: total training steps
      decay_frac: fraction of total_steps for decay phase (default 0.2 = last 20%)
      final_lr_frac: lr at end as fraction of peak (default 0.1)

    Returns:
      callable(step) -> lr_multiplier in [0, 1]
    """
    decay_start = total_steps - int(decay_frac * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        elif step < decay_start:
            return 1.0
        else:
            progress = (step - decay_start) / max(1, total_steps - decay_start)
            return 1.0 - (1.0 - final_lr_frac) * progress
    return lr_lambda


def make_cosine_lambda(warmup_steps: int, total_steps: int,
                        final_lr_frac: float = 0.1) -> Callable:
    """Cosine decay with warmup. Standard alternative to WSD."""
    import math
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return final_lr_frac + (1 - final_lr_frac) * 0.5 * (1 + math.cos(progress * math.pi))
    return lr_lambda


if __name__ == "__main__":
    # Quick visual test
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = 10000
    warmup = 500

    wsd = make_wsd_lambda(warmup, total, decay_frac=0.2, final_lr_frac=0.1)
    cos = make_cosine_lambda(warmup, total, final_lr_frac=0.1)

    steps = list(range(total))
    plt.figure(figsize=(10, 4))
    plt.plot(steps, [wsd(s) for s in steps], label="WSD", linewidth=2)
    plt.plot(steps, [cos(s) for s in steps], label="Cosine", linewidth=2, alpha=0.7)
    plt.xlabel("step")
    plt.ylabel("lr multiplier")
    plt.legend()
    plt.title(f"WSD vs Cosine (total={total}, warmup={warmup}, decay_frac=0.2)")
    plt.grid(True, alpha=0.3)
    plt.savefig("/tmp/wsd_vs_cosine.png", dpi=100, bbox_inches='tight')
    print("Saved /tmp/wsd_vs_cosine.png")
