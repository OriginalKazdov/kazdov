"""Plot training-loss curve, per-dataset eval-loss curves, and the
GPT-2-124M baseline horizontal lines for a Kazdov-LM training run.

Usage:
    python plot_training.py /path/to/metrics.jsonl
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# GPT-2 124M baseline (matched-tokenizer, 81,920 held-out tokens per dataset)
BASELINE = {
    "open_web_math": 3.57,
    "proof_pile_v1": 3.12,
    "mathpile":      3.04,
    "automath":      3.28,
    "numina_math":   2.58,
    "mean":          3.12,
}

DATASET_LABEL = {
    "open_web_math": "OpenWebMath",
    "proof_pile_v1": "proof-pile v1",
    "mathpile":      "MathPile",
    "automath":      "AutoMathText",
    "numina_math":   "NuminaMath",
    "mean":          "MEAN",
}

DATASET_COLOR = {
    "open_web_math": "#d62728",
    "proof_pile_v1": "#1f77b4",
    "mathpile":      "#2ca02c",
    "automath":      "#ff7f0e",
    "numina_math":   "#9467bd",
    "mean":          "#000000",
}


def load_metrics(path):
    train, eval_pts = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "eval" in r:
                eval_pts.append(r)
            elif "loss" in r:
                train.append(r)
    return train, eval_pts


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "metrics.jsonl"
    train, evals = load_metrics(path)
    print(f"loaded {len(train)} train logs, {len(evals)} eval points")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [1, 1.4]})

    # --- Top: training loss (smoothed) ------------------------------
    ax = axes[0]
    tok = np.array([r["tokens"] for r in train]) / 1e6
    loss = np.array([r["loss"] for r in train])
    # exponential moving average for readability
    alpha = 0.05
    sm = np.zeros_like(loss)
    sm[0] = loss[0]
    for i in range(1, len(loss)):
        sm[i] = alpha * loss[i] + (1 - alpha) * sm[i - 1]
    ax.plot(tok, loss, color="#888", alpha=0.25, lw=0.4, label="raw")
    ax.plot(tok, sm,   color="#1f77b4", lw=1.5, label="EMA(0.05)")
    ax.axhline(BASELINE["mean"], color="#d62728", ls="--", lw=1.0,
               label=f"GPT-2 124M baseline (mean {BASELINE['mean']})")
    ax.set_xlabel("Tokens seen (M)")
    ax.set_ylabel("Train LM loss")
    ax.set_title("Kazdov-α training loss (98M params, single 4090, bf16)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(2.0, 8.0)

    # --- Bottom: per-dataset eval loss ------------------------------
    ax = axes[1]
    eval_tok = np.array([r["tokens"] for r in evals]) / 1e6
    for ds in ["open_web_math", "proof_pile_v1", "mathpile", "automath",
                "numina_math", "mean"]:
        vals = np.array([r["eval"][ds] for r in evals])
        ax.plot(eval_tok, vals,
                color=DATASET_COLOR[ds],
                lw=1.5 if ds == "mean" else 1.0,
                marker="o", markersize=4,
                label=f"{DATASET_LABEL[ds]} (Kazdov)")
        # horizontal baseline
        ax.axhline(BASELINE[ds],
                   color=DATASET_COLOR[ds], ls="--", lw=0.8, alpha=0.4)

    # text labels for baselines
    xmax = eval_tok[-1]
    for ds in ["open_web_math", "proof_pile_v1", "mathpile", "automath",
                "numina_math", "mean"]:
        ax.text(xmax * 1.01, BASELINE[ds],
                f"GPT-2 baseline ({BASELINE[ds]})",
                color=DATASET_COLOR[ds], fontsize=7, va="center", alpha=0.7)

    ax.set_xlabel("Tokens seen (M)")
    ax.set_ylabel("Held-out LM loss")
    ax.set_title("Held-out loss per dataset vs. GPT-2-124M baseline (dashed)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, xmax * 1.18)

    plt.tight_layout()

    out_pdf = Path(path).parent / "training_curves.pdf"
    out_png = Path(path).parent / "training_curves.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"saved: {out_pdf}")
    print(f"saved: {out_png}")

    # --- Print summary table -----------------------------------------
    if evals:
        last = evals[-1]
        print("\n=== Latest eval (step {}, {:.1f}M tokens) ===".format(
            last["step"], last["tokens"] / 1e6))
        print(f"{'dataset':<16} {'Kazdov':>8} {'GPT-2 124M':>12} {'Δ (nats)':>10} {'PPL ratio':>10}")
        print("-" * 60)
        for ds in ["open_web_math", "proof_pile_v1", "mathpile", "automath",
                    "numina_math", "mean"]:
            k = last["eval"][ds]
            b = BASELINE[ds]
            delta = k - b
            ratio = float(np.exp(b - k))   # > 1 means we are better
            print(f"{ds:<16} {k:>8.3f} {b:>12.2f} {delta:>+10.3f} {ratio:>10.2f}x")


if __name__ == "__main__":
    main()
