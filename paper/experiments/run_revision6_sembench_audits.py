import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "data/results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

from certineg import hypergeom_upper_rate

EPS = 0.10
DELTA = 0.05
R = 2000
SEED = 20261026
FRACS = np.array([0.30, 0.45, 0.60, 0.80, 1.00])
TARGETS = [
    "AIR BAGS",
    "ENGINE AND ENGINE COOLING",
    "VISIBILITY/WIPER",
    "EXTERIOR LIGHTING",
]


def safe_name(t):
    return "".join(c if c.isalnum() else "_" for c in t.lower()).strip("_")


def sizes(N):
    out = []
    prev = 0
    for f in FRACS:
        m = max(prev, min(N, round(float(f) * N)))
        out.append(m)
        prev = m
    return np.array(out, dtype=int)


def fixed(sample, z, ps):
    N = len(z)
    mark = np.zeros(N, np.int8)
    mark[sample] = 1
    cn = np.cumsum(mark)
    cx = np.cumsum(mark * z)
    k = -1
    for i, m in enumerate(ps):
        n = int(cn[m - 1])
        x = int(cx[m - 1])
        if hypergeom_upper_rate(int(m), n, x, DELTA) <= EPS:
            k = i
        else:
            break
    return k


def simulate(z, n, seed):
    N = len(z)
    ps = sizes(N)
    fdps = np.array([z[:m].mean() for m in ps])
    rng = np.random.default_rng(seed)
    ret = np.zeros(R)
    chosen = np.full(R, -1, dtype=int)
    vio = np.zeros(R, dtype=int)
    for r in range(R):
        s = rng.choice(N, size=min(n, N), replace=False)
        k = fixed(s, z, ps)
        chosen[r] = k
        if k >= 0:
            ret[r] = ps[k] / N
            vio[r] = int(fdps[k] > EPS)
    return ret, chosen, vio, fdps, ps


def main() -> None:
    meta = json.loads((OUT / "sembench_predicate_cache_meta.json").read_text())[
        "targets"
    ]
    rows = []
    freqs = []
    for i, t in enumerate(TARGETS):
        a = np.load(OUT / f"sembench_{safe_name(t)}_groups.npz")
        z = a["z"].astype(np.int8)
        N = len(z)
        n = round(0.35 * N)
        ret, ch, vio, fdps, ps = simulate(z, n, SEED + 50000 + i)
        oracle = max([ps[j] / N for j, v in enumerate(fdps) if v <= EPS], default=0.0)
        cs = np.cumsum(z) / np.arange(1, N + 1)
        ok = np.where((np.arange(1, N + 1) >= int(np.ceil(0.30 * N))) & (cs <= EPS))[0]
        cont = (ok[-1] + 1) / N if len(ok) else 0.0
        m = meta[t]
        rows.append(
            {
                "predicate": t,
                "f1": m["f1"],
                "auc": m["auc"],
                "candidate_groups": N,
                "candidate_fdp": float(z.mean()),
                "oracle_K5": oracle,
                "oracle_continuous": cont,
                "audit_n": n,
                "audit_fraction": n / N,
                "mean_retention": float(ret.mean()),
                "sd_retention_audit": float(ret.std(ddof=1)),
                "median_retention": float(np.median(ret)),
                "violation_count": int(vio.sum()),
                "violation_rate": float(vio.mean()),
                "census_cost": int(a["cost"].sum()),
            }
        )
        counts = {str(-1): int((ch < 0).sum())}
        for j, f in enumerate(FRACS):
            counts[str(float(f))] = int((ch == j).sum())
        freqs.append({"predicate": t, **counts})
    pd.DataFrame(rows).to_csv(OUT / "sembench_primary_predicates_r6.csv", index=False)
    pd.DataFrame(freqs).to_csv(
        OUT / "sembench_primary_predicate_frequencies_r6.csv", index=False
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nFREQ\n", pd.DataFrame(freqs).to_string(index=False))


if __name__ == "__main__":
    main()
