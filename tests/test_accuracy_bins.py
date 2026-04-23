"""F-12: equal-count calibration bins — each of 5 bins gets ~N/5 samples.

With 100 synthetic probabilities, each bin should contain ~20 rows (±5).
Tests the bin-edge computation logic extracted from scripts/accuracy.py.
"""
import numpy as np
import pytest


def _compute_bins(probs, n_bins=5):
    """Replicate the F-12 equal-count bin logic from scripts/accuracy.py."""
    probs_arr = np.sort(probs)
    n = len(probs_arr)
    bin_edges = [float(probs_arr[int(i * n / n_bins)]) for i in range(n_bins)]
    bin_edges.append(1.0)
    return list(zip(bin_edges[:-1], bin_edges[1:]))


class TestEqualCountBins:
    def test_five_bins_produced(self):
        """F-12: exactly 5 bins returned."""
        probs = list(np.random.default_rng(0).uniform(0, 0.4, 100))
        bins = _compute_bins(probs)
        assert len(bins) == 5

    def test_each_bin_has_approx_equal_count(self):
        """F-12: each bin gets ~20 samples out of 100 (±5 tolerance)."""
        rng = np.random.default_rng(42)
        probs = list(rng.uniform(0.01, 0.35, 100))
        bins = _compute_bins(probs)

        counts = []
        for lo, hi in bins:
            count = sum(1 for p in probs if lo <= p < hi)
            counts.append(count)

        # With equal-count design every bin targets n/5 = 20
        target = len(probs) / 5
        for i, (count, (lo, hi)) in enumerate(zip(counts, bins)):
            assert abs(count - target) <= 5, (
                f"Bin {i} ({lo:.2%}-{hi:.2%}) has {count} samples, "
                f"expected ~{target:.0f} ±5"
            )

    def test_bins_cover_full_range(self):
        """F-12: bin edges span [min_prob, 1.0] with no gap."""
        probs = list(np.random.default_rng(7).uniform(0.01, 0.30, 100))
        bins = _compute_bins(probs)
        # Last edge is always 1.0
        assert bins[-1][1] == 1.0
        # First edge is the minimum probability in the sorted array
        assert bins[0][0] == pytest.approx(min(probs), abs=1e-9)
        # Bins are contiguous
        for (_, hi), (lo2, _) in zip(bins[:-1], bins[1:]):
            assert hi == pytest.approx(lo2, abs=1e-9)

    def test_works_with_skewed_distribution(self):
        """F-12: equal-count bins handle right-skewed (most probs near zero) distributions."""
        rng = np.random.default_rng(99)
        # Typical HK racing: most horses have low prob, a few have high prob
        probs = list(rng.beta(1, 8, 200))  # heavily right-skewed
        bins = _compute_bins(probs)

        assert len(bins) == 5
        counts = [sum(1 for p in probs if lo <= p < hi) for lo, hi in bins]
        target = len(probs) / 5
        for i, count in enumerate(counts):
            assert abs(count - target) <= 8, (
                f"Bin {i} has {count} samples, expected ~{target:.0f} ±8"
            )
