from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "figures"
F.mkdir(exist_ok=True)


def box(ax, x, y, w, h, text):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02", fill=False, linewidth=1.2
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)
    return p


def arrow(ax, a, b):
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=10, linewidth=1.0)
    )


fig, ax = plt.subplots(figsize=(10.4, 2.8))
ax.set_xlim(0, 10.45)
ax.set_ylim(0, 2.8)
ax.axis("off")
coords = [
    (0.2, 1.0, "Semantic NOT EXISTS\nquery"),
    (2.0, 1.0, "Freeze candidate\noutput $A_0$"),
    (3.8, 1.0, "Predeclare nested\n$A_1\\subset\\cdots\\subset A_K$"),
    (5.7, 1.0, "SRSWOR group\naudit"),
    (7.4, 1.0, "Exact within-group\nverification"),
    (9.0, 1.0, "Return largest\ncertified $A_k$"),
]
prev = None
for x, y, t in coords:
    box(ax, x, y, 1.25, 0.55, t)
    if prev is not None:
        arrow(ax, (prev + 1.25, 1.275), (x, 1.275))
    prev = x
ax.text(
    8.0,
    0.48,
    "proxy-ranked records\nwitness -> stop; none -> exhaust",
    fontsize=8.3,
    ha="center",
)
fig.tight_layout()
fig.savefig(F / "architecture.pdf", bbox_inches="tight")
fig.savefig(F / "architecture.png", dpi=220, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 2.5))
ax.set_xlim(0, 7.2)
ax.set_ylim(0, 2.5)
ax.axis("off")
for i in range(5):
    box(ax, 0.4 + i * 1.0, 1.15, 0.62, 0.5, "row " + str(i + 1))
ax.text(0.4, 1.9, "True group: EXISTS positive", fontsize=9)
ax.text(0.4, 0.5, "Semantic predicate misses one positive row", fontsize=9)
ax.text(5.7, 1.9, "NOT EXISTS output", fontsize=9, ha="center")
box(ax, 5.15, 1.05, 1.1, 0.7, "FALSE\nABSENCE")
arrow(ax, (4.98, 1.4), (5.15, 1.4))
ax.annotate(
    "missed positive",
    xy=(2.02, 1.4),
    xytext=(2.1, 0.75),
    arrowprops={"arrowstyle": "->"},
    fontsize=8,
)
fig.tight_layout()
fig.savefig(F / "failure_mechanism.pdf", bbox_inches="tight")
fig.savefig(F / "failure_mechanism.png", dpi=220, bbox_inches="tight")
plt.close(fig)
