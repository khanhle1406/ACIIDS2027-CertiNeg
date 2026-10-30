from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT.parent / "data/results" / "retention_audit_curve.csv")
fig, ax = plt.subplots(figsize=(6.25, 3.25))
for name, g in df.groupby("dataset", sort=False):
    g = g.sort_values("audit_fraction")
    line = ax.plot(
        100 * g.audit_fraction,
        100 * g.mean_retention,
        marker="o",
        label=f"{name}: certified retention",
    )[0]
    # K=5 oracle, the relevant ceiling for the curve's family.
    ax.axhline(
        100 * g.oracle_K5.iloc[0],
        linestyle="--",
        linewidth=1.1,
        label=f"{name}: K=5 oracle",
    )
ax.set_xlabel("Audited groups / candidate groups (%)")
ax.set_ylabel("Returned candidate output (%)")
ax.set_xlim(0, 70)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=7, loc="lower right", ncol=2)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "retention_budget.pdf", bbox_inches="tight")
fig.savefig(ROOT / "figures" / "retention_budget.png", dpi=220, bbox_inches="tight")
plt.close(fig)
