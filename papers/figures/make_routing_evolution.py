"""Generate a 4-panel routing-usage heatmap showing how the per-block expert
distribution evolved across the 5B-token Kazdov-α run.

Pulls diagnostics.json from each of:
  results/                  (344M-token snapshot, named diagnostics.json)
  results/routing_diag_1B/  (1B-token snapshot)
  results/routing_diag_final/ (5B-token snapshot)

Output: papers/figures/routing_evolution.{png,pdf}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent.parent
SNAPSHOTS = [
    ("344M tokens",  REPO / "results" / "diagnostics.json"),
    ("1B tokens",    REPO / "results" / "routing_diag_1B" / "diagnostics.json"),
    ("4B tokens",    REPO / "results" / "routing_diag_4B" / "diagnostics.json"),
    ("5B tokens (final)", REPO / "results" / "routing_diag_final" / "diagnostics.json"),
]


def load(p):
    with open(p) as f:
        return json.load(f)


fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)

for ax, (label, path) in zip(axes, SNAPSHOTS):
    data = load(path)
    diag = data["per_block"]
    K = len(diag[0]["usage"])
    n_blocks = len(diag)
    mat = np.array([d["usage"] for d in diag])  # (n_blocks, K)

    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1.0)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"e{k}" for k in range(K)], fontsize=9)
    ax.set_yticks(range(n_blocks))
    if ax is axes[0]:
        ax.set_yticklabels([f"block {i}" for i in range(n_blocks)], fontsize=8)
    avg_e = data["avg_entropy"] / data["max_entropy"]
    ax.set_title(f"{label}\nentropy {avg_e*100:.1f}% / collapse {data['avg_collapse']:.2f}",
                 fontsize=10)
    for i in range(n_blocks):
        for j in range(K):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    color="white" if v < 0.45 else "black", fontsize=7)

# single colorbar on the right
cbar = fig.colorbar(im, ax=axes, fraction=0.014, pad=0.02)
cbar.set_label("mean routing weight")

fig.suptitle("Per-block MoBE expert routing across training (Kazdov-α)",
             fontsize=12, y=1.02)

out_png = REPO / "papers" / "figures" / "routing_evolution.png"
out_pdf = REPO / "papers" / "figures" / "routing_evolution.pdf"
plt.savefig(out_png, dpi=160, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"saved: {out_png}\nsaved: {out_pdf}")
