import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from certineg import group_inspection_cost, hypergeom_upper_rate
from run_experiments import SEED, make_candidate_population, make_nested_sets

OUT = ROOT.parent / "data/results"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(exist_ok=True)


def main() -> None:
    _sizes, _z, risk, groups = make_candidate_population()
    sets = make_nested_sets(risk, [0.3, 0.45, 0.6, 0.8, 1.0])
    A = sets[3]
    rng = np.random.default_rng(SEED + 707)
    # Proxy-signal ablation: same labels, fixed independent noise, signal changes only ordering efficiency.
    rows = []
    noise = [rng.normal(size=len(l)) for l, s in groups]
    for signal in [0.0, 0.5, 1.0, 2.0, 4.0]:
        costs = []
        for g in A:
            lab, _ = groups[int(g)]
            sc = noise[int(g)] + signal * lab
            costs.append(group_inspection_cost(lab, sc))
        rows.append(
            (signal, np.mean(costs), np.median(costs), np.percentile(costs, 95))
        )
    pd.DataFrame(
        rows,
        columns=["proxy_signal", "mean_label_cost_per_group", "median", "p95"],
    ).to_csv(OUT / "proxy_ablation.csv", index=False)

    # epsilon=0 exact sample burden under zero observed defects: smallest n that gets U=0.
    rows = []
    for N in [50, 100, 500, 1000]:
        for delta in [0.10, 0.05, 0.01]:
            min_n = None
            for n in range(N + 1):
                if hypergeom_upper_rate(N, n, 0, delta) == 0:
                    min_n = n
                    break
            rows.append((N, delta, min_n, min_n / N))
    pd.DataFrame(
        rows,
        columns=["N", "delta", "min_n_zero_defects", "sample_fraction"],
    ).to_csv(OUT / "epsilon_zero_burden.csv", index=False)

    # Group-size skew ablation: fix 12% positive groups, one witness, compare random expected vs proxy-ranked cost.
    rows = []
    for sigma in [0.1, 0.5, 1.0, 1.5]:
        rr = np.random.default_rng(SEED + int(sigma * 1000) + 13)
        N = 1500
        sz = np.clip(np.rint(rr.lognormal(3.3, sigma, N)).astype(int), 2, 800)
        zz = rr.binomial(1, 0.12, N)
        pc = []
        rc = []
        for m, d in zip(sz, zz):
            lab = np.zeros(m, dtype=int)
            if d:
                lab[rr.integers(0, m)] = 1
            sc = rr.normal(size=m) + 2.0 * lab
            pc.append(group_inspection_cost(lab, sc))
            rc.append(m if d == 0 else (m + 1) / 2)
        rows.append(
            (
                sigma,
                np.median(sz),
                np.percentile(sz, 95),
                np.mean(pc),
                np.mean(rc),
                100 * (1 - np.mean(pc) / np.mean(rc)),
            )
        )
    pd.DataFrame(
        rows,
        columns=[
            "lognormal_sigma",
            "median_group_size",
            "p95_group_size",
            "proxy_cost",
            "random_expected_cost",
            "saving_pct",
        ],
    ).to_csv(OUT / "skew_ablation.csv", index=False)

    p = pd.read_csv(OUT / "proxy_ablation.csv")
    plt.figure(figsize=(5.1, 3.1))
    plt.plot(p.proxy_signal, p.mean_label_cost_per_group, marker="o")
    plt.xlabel("Proxy signal separation")
    plt.ylabel("Mean trusted labels / audited group")
    plt.tight_layout()
    plt.savefig(FIG / "proxy_ablation.pdf", bbox_inches="tight")
    plt.savefig(FIG / "proxy_ablation.png", dpi=220, bbox_inches="tight")
    plt.close()

    e = pd.read_csv(OUT / "epsilon_zero_burden.csv")
    for delta, sub in e.groupby("delta"):
        plt.figure(figsize=(5.1, 3.1))
        plt.plot(sub.N, sub.sample_fraction, marker="o")
        plt.xlabel("Candidate groups N")
        plt.ylabel("Minimum audited fraction")
        plt.ylim(0.85, 1.01)
        plt.tight_layout()
        plt.savefig(FIG / f"epsilon_zero_delta_{delta}.pdf", bbox_inches="tight")
        plt.close()

    print("proxy ablation")
    print(pd.read_csv(OUT / "proxy_ablation.csv").to_string(index=False))
    print("\nepsilon0")
    print(pd.read_csv(OUT / "epsilon_zero_burden.csv").to_string(index=False))
    print("\nskew")
    print(pd.read_csv(OUT / "skew_ablation.csv").to_string(index=False))


if __name__ == "__main__":
    main()
