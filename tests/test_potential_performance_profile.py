"""Check the plotted energy profiles without rerunning the particle benchmark."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.integrate import quad


def load_benchmark():
    path = Path(__file__).with_name("example_potential_performance.py")
    spec = importlib.util.spec_from_file_location("potential_performance_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_radial_profiles_use_common_analytic_potential_normalization():
    benchmark = load_benchmark()
    radii = np.geomspace(0.05 * benchmark.PLUMMER_SCALE_KPC,
                        benchmark.PLUMMER_TRUNCATION_KPC, benchmark.ERROR_PROBE_COUNT)
    truth, _ = benchmark.analytic_plummer(radii)
    phi_min = benchmark.analytic_plummer(np.zeros(1))[0][0]
    rows = []
    for n, offset in [(10000, 0.002), (1000, 0.2)]:
        rows.append({
            "n_particles": n,
            "softening_support_kpc": 0.01,
            "native": {"potential_probe": (truth + offset * phi_min).tolist()},
            "agama": {"potential_probe": (truth - offset * phi_min).tolist()},
        })
    profile = benchmark.radial_energy_profile(rows)
    assert profile["n_particles"] == 10000
    np.testing.assert_array_equal(profile["radii_kpc"], radii)
    np.testing.assert_allclose(profile["analytic"], truth / phi_min)
    np.testing.assert_allclose(profile["native"], truth / phi_min + 0.002)
    np.testing.assert_allclose(profile["agama"], truth / phi_min - 0.002)
    assert profile["phi_min_km2_s2"] == phi_min < 0


def test_analytic_central_potential_matches_shell_integral():
    benchmark = load_benchmark()
    a = benchmark.PLUMMER_SCALE_KPC
    rt = benchmark.PLUMMER_TRUNCATION_KPC
    enclosed_fraction = rt**3 / (rt**2 + a*a)**1.5
    rho0 = 3 * benchmark.PLUMMER_TOTAL_MASS_MSUN / (4 * np.pi * a**3 * enclosed_fraction)
    integral = quad(lambda r: rho0 * r * (1 + (r/a)**2)**-2.5,
                    0, rt, epsabs=0, epsrel=1e-12)[0]
    expected = -4 * np.pi * 4.30091727067736e-6 * integral
    actual = benchmark.analytic_plummer(np.zeros(1))[0][0]
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
