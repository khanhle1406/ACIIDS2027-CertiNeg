import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from certineg import exact_coverage_min


def test_exhaustive_exact_coverage_all_N_le_30():
    for N in range(1, 31):
        for n in range(N + 1):
            for alpha in (0.10, 0.05, 0.01):
                cov, _ = exact_coverage_min(N, n, alpha)
                assert cov + 1e-12 >= 1 - alpha, (N, n, alpha, cov)


if __name__ == "__main__":
    test_exhaustive_exact_coverage_all_N_le_30()
    print(
        "exhaustive exact-coverage tests passed for all N<=30, all n, alpha in {0.1,0.05,0.01}"
    )


def test_fixed_sequence_selected_set_guarantee_small_populations():
    import itertools
    import math

    import numpy as np
    from certineg import certify_nested_fixed_sequence

    delta = 0.10
    for N in range(2, 8):
        # Predeclared nested prefixes, deduplicated for small N.
        sizes = sorted({max(1, min(N, round(f * N))) for f in (0.25, 0.5, 0.75, 1.0)})
        sets = [np.arange(m, dtype=int) for m in sizes]
        for epsilon in (0.20, 0.50):
            for bits in itertools.product((0, 1), repeat=N):
                z = np.asarray(bits, dtype=int)
                for n in range(N + 1):
                    total = math.comb(N, n)
                    bad = 0
                    for sample_tuple in itertools.combinations(range(N), n):
                        sample = np.asarray(sample_tuple, dtype=int)
                        cert = certify_nested_fixed_sequence(
                            sets, sample, z, epsilon, delta
                        )
                        if cert.chosen_index is not None:
                            A = sets[cert.chosen_index]
                            if float(z[A].mean()) > epsilon:
                                bad += 1
                    assert bad / total <= delta + 1e-12, (
                        N,
                        epsilon,
                        bits,
                        n,
                        bad / total,
                    )


def test_simultaneous_and_weighted_small_populations() -> None:
    """Integrate the actual selected-set error over all SRSWOR samples."""
    import itertools
    import math

    import numpy as np
    from certineg import certify_nested_from_largest_sample, certify_nested_weighted

    for N in range(2, 7):
        sets = [np.arange(m) for m in sorted({1, N // 2, N})]
        for bits in itertools.product((0, 1), repeat=N):
            z = np.asarray(bits)
            for n in range(N + 1):
                violations = [0, 0]
                for sample in itertools.combinations(range(N), n):
                    s = np.asarray(sample, dtype=int)
                    results = [
                        certify_nested_from_largest_sample(sets, s, z, 0.5, 0.1),
                        certify_nested_weighted(
                            sets, s, z, 0.5, 0.1, range(1, len(sets) + 1)
                        ),
                    ]
                    for j, result in enumerate(results):
                        if result.chosen_index is not None:
                            violations[j] += int(
                                z[sets[result.chosen_index]].mean() > 0.5
                            )
                assert all(v / math.comb(N, n) <= 0.1 + 1e-12 for v in violations)


def test_fixed_sequence_stops_at_first_failure() -> None:
    import numpy as np
    from certineg import certify_nested_fixed_sequence

    sets = [np.arange(1), np.arange(10)]
    z = np.array([1] + [0] * 9)
    result = certify_nested_fixed_sequence(sets, np.arange(10), z, 0.2, 0.05)
    assert result.upper_rates == (1.0, 0.1)
    assert result.chosen_index is None
