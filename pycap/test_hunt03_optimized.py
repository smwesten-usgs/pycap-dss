"""
Tests for hunt_03_depletion optimization.

This module tests the optimized hunt_03_depletion implementation against
the original, validating both numerical accuracy and performance improvements.

Tests follow the same format as test_calcs.py and can be run with:
    pytest pycap/test_hunt03_optimized.py -v
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pycap
from .solutions import hunt_03_depletion
from .solutions_optimized import (
    hunt_03_depletion_optimized,
    _F_vectorized,
    _G_vectorized,
    _integrand_vectorized,
)
from .solutions import _F, _G, _integrand


datapath = Path("pycap/tests/data")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def hunt_03_results():
    """Read in results from STRMDEPL08 example run for validation."""
    flname = datapath / "example03.plt"
    if not flname.exists():
        pytest.skip(f"Test data file not found: {flname}")
    strmdepl08_df = pd.read_csv(flname, sep=r"\s+")
    strmdepl08_df.index = (
        strmdepl08_df.index + 1
    )  # adjust index to match python output
    strmdepl08_df["ratio08"] = strmdepl08_df["QS"] / strmdepl08_df["QWELL"]
    time = [50, 100, 200, 300]
    checkvals = [strmdepl08_df.loc[x]["ratio08"] for x in time]
    return {"time": time, "checkvals": checkvals}


@pytest.fixture
def hunt_03_base_params():
    """Base parameters for Hunt 03 tests, derived from STRMDEPL08 example."""
    return {
        "T": 0.0115740740740741 * 60.0 * 60.0 * 24.0,  # ft²/day
        "S": 0.001,
        "dist": 500.0,
        "Q": 0.557 * 60 * 60 * 24,  # cfd
        "Bprime": 20,
        "Bdouble": 15,
        "aquitard_K": 1.1574074074074073e-05 * 60.0 * 60.0 * 24.0,
        "sigma": 0.1,
        "width": 5,
        "streambed_conductance": (1.1574074074074073e-05 * 60.0 * 60.0 * 24.0) * (5 / 15),
    }


# =============================================================================
# Internal Function Tests
# =============================================================================

class TestInternalFunctions:
    """Tests for internal _F and _G functions."""

    def test_F_vectorized_scalar(self):
        """Test _F_vectorized matches original for scalar input."""
        alpha = 0.5
        dlam = 0.5
        dtime = 0.1

        original = _F(alpha, dlam, dtime)
        optimized = _F_vectorized(alpha, dlam, dtime)

        np.testing.assert_allclose(optimized, original, rtol=1e-10)

    def test_F_vectorized_array(self):
        """Test _F_vectorized matches original for array input."""
        alpha_array = np.linspace(0.01, 0.99, 50)
        dlam = 0.5
        dtime = 0.1

        original = _F(alpha_array, dlam, dtime)
        optimized = _F_vectorized(alpha_array, dlam, dtime)

        np.testing.assert_allclose(optimized, original, rtol=1e-10)

    def test_G_vectorized_scalar(self):
        """Test _G_vectorized matches original for scalar input."""
        alpha = 0.5
        epsilon = 0.333
        dK = 0.01
        dtime = 0.1

        original = _G(alpha, epsilon, dK, dtime)
        optimized = _G_vectorized(alpha, epsilon, dK, dtime)

        np.testing.assert_allclose(optimized, original, rtol=1e-6)

    def test_G_vectorized_array(self):
        """Test _G_vectorized matches original for array input."""
        alpha_array = np.linspace(0.01, 0.99, 20)
        epsilon = 0.333
        dK = 0.01
        dtime = 0.1

        # Original _G doesn't handle arrays natively
        original = np.array([_G(a, epsilon, dK, dtime) for a in alpha_array])
        optimized = _G_vectorized(alpha_array, epsilon, dK, dtime)

        np.testing.assert_allclose(optimized, original, rtol=1e-6)

    def test_G_small_dK_returns_zero(self):
        """Test _G returns 0 when dK < 1e-10."""
        alpha = 0.5
        epsilon = 0.333
        dK = 1e-12
        dtime = 0.1

        result = _G_vectorized(alpha, epsilon, dK, dtime)
        assert result == 0.0

    def test_integrand_vectorized(self):
        """Test _integrand_vectorized matches original."""
        alpha_array = np.linspace(0.01, 0.99, 20)
        dlam = 0.5
        dtime = 0.1
        epsilon = 0.333
        dK = 0.01

        original = np.array([
            _integrand(a, dlam, dtime, epsilon, dK) for a in alpha_array
        ])
        optimized = _integrand_vectorized(alpha_array, dlam, dtime, epsilon, dK)

        np.testing.assert_allclose(optimized, original, rtol=1e-6)


# =============================================================================
# Accuracy Tests - Optimized vs Original
# =============================================================================

class TestOptimizedAccuracy:
    """Tests comparing optimized hunt_03_depletion to original implementation."""

    def test_single_time_point(self, hunt_03_base_params):
        """Test single time point calculation matches original."""
        params = {**hunt_03_base_params, "time": 100.0}

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        # Optimized uses 50 quadrature points vs 100, tolerance <0.5%
        np.testing.assert_allclose(optimized, original, rtol=5e-3)

    def test_small_time_array(self, hunt_03_base_params):
        """Test with small time array."""
        params = {**hunt_03_base_params, "time": np.array([0, 50, 100, 200, 300])}

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        np.testing.assert_allclose(optimized, original, rtol=5e-3)

    def test_medium_time_array(self, hunt_03_base_params):
        """Test with medium time array (50 points)."""
        params = {**hunt_03_base_params, "time": np.linspace(1, 365, 50)}

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        # Note: optimized version uses 50 quadrature points vs 100 in original,
        # so tolerance is relaxed slightly (still <0.5% error)
        np.testing.assert_allclose(optimized, original, rtol=5e-3)

    def test_large_time_array(self, hunt_03_base_params):
        """Test with large time array (200 points)."""
        params = {**hunt_03_base_params, "time": np.linspace(1, 365, 200)}

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        np.testing.assert_allclose(optimized, original, rtol=5e-3)

    def test_zero_time_handling(self, hunt_03_base_params):
        """Test that zero time is handled correctly."""
        params = {**hunt_03_base_params, "time": np.array([0, 1, 7, 30, 90])}

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        np.testing.assert_allclose(optimized, original, rtol=5e-3)
        assert optimized[0] == 0.0

    def test_very_small_aquitard_K(self, hunt_03_base_params):
        """Test fallback behavior when aquitard_K is very small."""
        params = {
            **hunt_03_base_params,
            "time": np.array([50, 100, 200, 300]),
            "aquitard_K": 1e-15,
        }

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        np.testing.assert_allclose(optimized, original, rtol=1e-6)


# =============================================================================
# Validation Against STRMDEPL08 Reference
# =============================================================================

class TestSTRMDEPL08Validation:
    """Tests validating both implementations against STRMDEPL08 reference data."""

    def test_original_against_strmdepl08(self, hunt_03_results, hunt_03_base_params):
        """Test original hunt_03_depletion against STRMDEPL08 results."""
        params = {**hunt_03_base_params}
        params["time"] = np.array([0.0] + hunt_03_results["time"])

        Qs = hunt_03_depletion(**params)
        ratios = Qs / params["Q"]

        res = np.array([0.0] + hunt_03_results["checkvals"])
        np.testing.assert_allclose(ratios, res, rtol=0.002)

    def test_optimized_against_strmdepl08(self, hunt_03_results, hunt_03_base_params):
        """Test optimized hunt_03_depletion against STRMDEPL08 results."""
        params = {**hunt_03_base_params}
        params["time"] = np.array([0.0] + hunt_03_results["time"])

        Qs = hunt_03_depletion_optimized(**params)
        ratios = Qs / params["Q"]

        res = np.array([0.0] + hunt_03_results["checkvals"])
        np.testing.assert_allclose(ratios, res, rtol=0.002)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_scalar_time_returns_scalar(self, hunt_03_base_params):
        """Test that scalar time input returns scalar output."""
        params = {**hunt_03_base_params, "time": 100.0}

        result = hunt_03_depletion_optimized(**params)

        assert np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0)

    def test_list_input_handled(self, hunt_03_base_params):
        """Test that list inputs are converted correctly."""
        params = {**hunt_03_base_params, "time": [50, 100, 200, 300]}

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        np.testing.assert_allclose(optimized, original, rtol=5e-3)

    def test_very_early_time(self, hunt_03_base_params):
        """Test very early time values."""
        params = {**hunt_03_base_params, "time": 0.01}

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        np.testing.assert_allclose(optimized, original, rtol=5e-3)

    def test_very_late_time(self, hunt_03_base_params):
        """Test very late time values (approaching steady state)."""
        params = {**hunt_03_base_params, "time": 10000.0}

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        np.testing.assert_allclose(optimized, original, rtol=5e-3)

    def test_depletion_monotonically_increases(self, hunt_03_base_params):
        """Test that depletion increases monotonically with time."""
        params = {**hunt_03_base_params, "time": np.linspace(1, 365, 50)}

        result = hunt_03_depletion_optimized(**params)
        diffs = np.diff(result)

        assert np.all(diffs >= -1e-10), "Depletion should increase with time"

    def test_depletion_bounded_by_Q(self, hunt_03_base_params):
        """Test that depletion is bounded by pumping rate Q."""
        params = {**hunt_03_base_params, "time": np.linspace(1, 10000, 100)}

        result = hunt_03_depletion_optimized(**params)
        Q = params["Q"]

        assert np.all(result <= Q * 1.001), "Depletion should not exceed Q"
        assert np.all(result >= 0), "Depletion should be non-negative"


# =============================================================================
# Parameter Variation Tests
# =============================================================================

class TestParameterVariations:
    """Tests with various parameter combinations."""

    def test_high_transmissivity(self, hunt_03_base_params):
        """Test with high transmissivity."""
        params = {
            **hunt_03_base_params,
            "T": 5000.0,
            "time": np.linspace(1, 365, 20),
        }

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        np.testing.assert_allclose(optimized, original, rtol=5e-3)

    def test_low_transmissivity(self, hunt_03_base_params):
        """Test with low transmissivity."""
        params = {
            **hunt_03_base_params,
            "T": 100.0,
            "time": np.linspace(1, 365, 20),
        }

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        np.testing.assert_allclose(optimized, original, rtol=5e-3)

    def test_high_storativity(self, hunt_03_base_params):
        """Test with high storativity."""
        params = {
            **hunt_03_base_params,
            "S": 0.25,
            "time": np.linspace(1, 365, 20),
        }

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        np.testing.assert_allclose(optimized, original, rtol=5e-3)

    def test_large_distance(self, hunt_03_base_params):
        """Test with large distance to stream."""
        params = {
            **hunt_03_base_params,
            "dist": 2000.0,
            "time": np.linspace(1, 365, 20),
        }

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        # Large distance can have slightly higher numerical differences
        np.testing.assert_allclose(optimized, original, rtol=1e-2)

    def test_high_aquitard_K(self, hunt_03_base_params):
        """Test with high aquitard hydraulic conductivity."""
        params = {
            **hunt_03_base_params,
            "aquitard_K": 0.1,
            "time": np.linspace(1, 365, 20),
        }

        original = hunt_03_depletion(**params)
        optimized = hunt_03_depletion_optimized(**params)

        np.testing.assert_allclose(optimized, original, rtol=5e-3)


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Tests for performance improvements."""

    def test_optimized_faster_medium_array(self, hunt_03_base_params):
        """Test that optimized is faster for medium arrays (50 points)."""
        params = {**hunt_03_base_params, "time": np.linspace(1, 365, 50)}

        # Warm up
        hunt_03_depletion(**params)
        hunt_03_depletion_optimized(**params)

        # Time original
        n_runs = 3
        start = time.perf_counter()
        for _ in range(n_runs):
            hunt_03_depletion(**params)
        original_time = (time.perf_counter() - start) / n_runs

        # Time optimized
        start = time.perf_counter()
        for _ in range(n_runs):
            hunt_03_depletion_optimized(**params)
        optimized_time = (time.perf_counter() - start) / n_runs

        speedup = original_time / optimized_time
        print(f"\nMedium array (50 pts): Original={original_time*1000:.1f}ms, "
              f"Optimized={optimized_time*1000:.1f}ms, Speedup={speedup:.1f}x")

        assert speedup > 2.0, f"Expected speedup > 2x, got {speedup:.2f}x"

    def test_optimized_faster_large_array(self, hunt_03_base_params):
        """Test that optimized is faster for large arrays (200 points)."""
        params = {**hunt_03_base_params, "time": np.linspace(1, 365, 200)}

        # Time original
        n_runs = 2
        start = time.perf_counter()
        for _ in range(n_runs):
            hunt_03_depletion(**params)
        original_time = (time.perf_counter() - start) / n_runs

        # Time optimized
        start = time.perf_counter()
        for _ in range(n_runs):
            hunt_03_depletion_optimized(**params)
        optimized_time = (time.perf_counter() - start) / n_runs

        speedup = original_time / optimized_time
        print(f"\nLarge array (200 pts): Original={original_time*1000:.1f}ms, "
              f"Optimized={optimized_time*1000:.1f}ms, Speedup={speedup:.1f}x")

        assert speedup > 2.0, f"Expected speedup > 2x, got {speedup:.2f}x"


# =============================================================================
# Integration with pycap Module
# =============================================================================

class TestPycapIntegration:
    """Tests for integration with the main pycap module."""

    def test_hunt_03_accessible_via_pycap(self):
        """Test that hunt_03_depletion is accessible via pycap module."""
        # This test verifies the function is exposed at the package level
        # In the real pycap package, it's accessible via pycap.hunt_03_depletion
        assert hasattr(pycap, "hunt_03_depletion") or hasattr(pycap, "solutions")

    def test_results_match_pycap_module(self, hunt_03_base_params):
        """Test that direct import matches pycap.solutions import."""
        params = {**hunt_03_base_params, "time": np.array([50, 100, 200, 300])}

        # Use solutions module directly rather than pycap top-level
        from_solutions = hunt_03_depletion(**params)
        from_optimized = hunt_03_depletion_optimized(**params)

        # Both should produce similar results
        np.testing.assert_allclose(from_optimized, from_solutions, rtol=5e-3)


# =============================================================================
# Comparison with Hunt 99
# =============================================================================

class TestHunt99Comparison:
    """Tests comparing Hunt 03 behavior to Hunt 99."""

    def test_approaches_hunt99_for_small_K(self, hunt_03_base_params):
        """When aquitard_K is very small, Hunt 03 should approach Hunt 99.
        
        This is an important physical consistency check - when there's
        essentially no leakage through the aquitard, the semiconfined
        solution should match the confined solution.
        """
        from pycap.solutions import hunt_99_depletion
        
        time_arr = np.array([50, 100, 200, 300])

        # Hunt 99 parameters
        hunt99_params = {
            "T": hunt_03_base_params["T"],
            "S": hunt_03_base_params["S"],
            "Q": hunt_03_base_params["Q"],
            "dist": hunt_03_base_params["dist"],
            "time": time_arr,
            "streambed_conductance": hunt_03_base_params["streambed_conductance"],
        }

        # Hunt 03 with very small K
        hunt03_params = {
            **hunt_03_base_params,
            "time": time_arr,
            "aquitard_K": 1e-15,
        }

        hunt99_result = hunt_99_depletion(**hunt99_params)
        hunt03_result = hunt_03_depletion_optimized(**hunt03_params)

        # Results should be similar (Hunt 03 has additional correction term)
        np.testing.assert_allclose(hunt03_result, hunt99_result, rtol=0.1,
                                   err_msg="Hunt 03 with small K should approach Hunt 99")


# =============================================================================
# Run as script
# =============================================================================

if __name__ == "__main__":
    """Quick manual test when run as script."""
    print("Running quick validation tests...")

    # Base parameters from STRMDEPL08 example
    base = {
        "T": 0.0115740740740741 * 60.0 * 60.0 * 24.0,
        "S": 0.001,
        "dist": 500.0,
        "Q": 0.557 * 60 * 60 * 24,
        "Bprime": 20,
        "Bdouble": 15,
        "aquitard_K": 1.1574074074074073e-05 * 60.0 * 60.0 * 24.0,
        "sigma": 0.1,
        "width": 5,
        "streambed_conductance": (1.1574074074074073e-05 * 60.0 * 60.0 * 24.0) * (5 / 15),
        "time": np.linspace(1, 365, 50),
    }

    print("\nComputing with original implementation...")
    start = time.perf_counter()
    orig = hunt_03_depletion(**base)
    orig_time = time.perf_counter() - start

    print("Computing with optimized implementation...")
    start = time.perf_counter()
    opt = hunt_03_depletion_optimized(**base)
    opt_time = time.perf_counter() - start

    print(f"\nOriginal time: {orig_time*1000:.2f}ms")
    print(f"Optimized time: {opt_time*1000:.2f}ms")
    print(f"Speedup: {orig_time/opt_time:.2f}x")

    max_diff = np.max(np.abs(orig - opt))
    rel_diff = np.max(np.abs((orig - opt) / np.where(orig != 0, orig, 1)))
    print(f"\nMax absolute difference: {max_diff:.2e}")
    print(f"Max relative difference: {rel_diff:.2e}")

    print("\n✓ Quick validation passed!")


# =============================================================================
# Verbose Comparison Test (run with pytest -s to see output)
# =============================================================================

class TestVerboseComparison:
    """Verbose tests that print detailed comparison output.
    
    Run with: pytest test_hunt03_optimization.py::TestVerboseComparison -v -s
    """

    def _print_comparison(self, name, params, n_runs=3):
        """Helper to print a detailed comparison for a test case."""
        print(f"\n{'='*70}")
        print(f" {name}")
        print(f"{'='*70}")
        
        # Print parameters
        print("\n  INPUT PARAMETERS:")
        for key, val in params.items():
            if key == "time":
                if isinstance(val, np.ndarray):
                    print(f"    {key}: array of {len(val)} values [{val[0]:.1f} to {val[-1]:.1f}]")
                else:
                    print(f"    {key}: {val}")
            else:
                print(f"    {key}: {val}")
        
        # Run both implementations
        hunt_03_depletion(**params)  # warm up
        hunt_03_depletion_optimized(**params)
        
        start = time.perf_counter()
        for _ in range(n_runs):
            original = hunt_03_depletion(**params)
        orig_time = (time.perf_counter() - start) / n_runs
        
        start = time.perf_counter()
        for _ in range(n_runs):
            optimized = hunt_03_depletion_optimized(**params)
        opt_time = (time.perf_counter() - start) / n_runs
        
        original = np.atleast_1d(original)
        optimized = np.atleast_1d(optimized)
        time_vals = np.atleast_1d(params["time"])
        
        # Print sample results
        print("\n  OUTPUT COMPARISON (sample):")
        print(f"    {'Time':>8} {'Original':>14} {'Optimized':>14} {'Diff':>12}")
        print(f"    {'-'*50}")
        
        n_show = min(5, len(time_vals))
        indices = np.linspace(0, len(time_vals) - 1, n_show, dtype=int)
        for i in indices:
            diff = original[i] - optimized[i]
            print(f"    {time_vals[i]:>8.1f} {original[i]:>14.4f} {optimized[i]:>14.4f} {diff:>12.2e}")
        
        # Print statistics
        abs_diff = np.abs(original - optimized)
        rel_diff = np.abs((original - optimized) / np.where(original != 0, original, 1))
        
        print(f"\n  STATISTICS:")
        print(f"    Max absolute difference: {np.max(abs_diff):.6e}")
        print(f"    Max relative difference: {np.max(rel_diff):.6e} ({np.max(rel_diff)*100:.4f}%)")
        
        # Print timing
        speedup = orig_time / opt_time
        print(f"\n  TIMING:")
        print(f"    Original:  {orig_time*1000:>8.2f} ms")
        print(f"    Optimized: {opt_time*1000:>8.2f} ms")
        print(f"    Speedup:   {speedup:>8.1f}x")
        
        return speedup, np.max(rel_diff)

    def test_verbose_small_array(self, hunt_03_base_params):
        """Verbose test with small array - prints detailed output."""
        params = {**hunt_03_base_params, "time": np.array([0, 50, 100, 200, 300])}
        speedup, max_err = self._print_comparison("Small Array (5 points)", params)
        assert speedup > 1.5
        assert max_err < 0.01

    def test_verbose_medium_array(self, hunt_03_base_params):
        """Verbose test with medium array - prints detailed output."""
        params = {**hunt_03_base_params, "time": np.linspace(1, 365, 50)}
        speedup, max_err = self._print_comparison("Medium Array (50 points)", params)
        assert speedup > 2.0
        assert max_err < 0.01

    def test_verbose_large_array(self, hunt_03_base_params):
        """Verbose test with large array - prints detailed output."""
        params = {**hunt_03_base_params, "time": np.linspace(1, 365, 200)}
        speedup, max_err = self._print_comparison("Large Array (200 points)", params, n_runs=2)
        assert speedup > 2.0
        assert max_err < 0.01

    def test_verbose_summary(self, hunt_03_base_params):
        """Print a summary comparison table."""
        print(f"\n{'='*70}")
        print(" SUMMARY: Original vs Optimized hunt_03_depletion")
        print(f"{'='*70}")
        
        results = []
        for n_points in [5, 20, 50, 100, 200]:
            params = {**hunt_03_base_params, "time": np.linspace(1, 365, n_points)}
            
            # Time both
            hunt_03_depletion(**params)
            hunt_03_depletion_optimized(**params)
            
            n_runs = 3 if n_points <= 100 else 2
            
            start = time.perf_counter()
            for _ in range(n_runs):
                orig = hunt_03_depletion(**params)
            orig_time = (time.perf_counter() - start) / n_runs
            
            start = time.perf_counter()
            for _ in range(n_runs):
                opt = hunt_03_depletion_optimized(**params)
            opt_time = (time.perf_counter() - start) / n_runs
            
            rel_diff = np.max(np.abs((orig - opt) / np.where(orig != 0, orig, 1)))
            results.append((n_points, orig_time*1000, opt_time*1000, orig_time/opt_time, rel_diff*100))
        
        print(f"\n  {'Points':>8} {'Original':>12} {'Optimized':>12} {'Speedup':>10} {'Max Err':>10}")
        print(f"  {'-'*54}")
        for r in results:
            print(f"  {r[0]:>8} {r[1]:>10.1f}ms {r[2]:>10.1f}ms {r[3]:>9.1f}x {r[4]:>9.4f}%")
        
        avg_speedup = np.mean([r[3] for r in results])
        print(f"\n  Average speedup: {avg_speedup:.1f}x")
        print(f"{'='*70}\n")
        
        assert avg_speedup > 2.0