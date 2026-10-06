"""Accuracy references kept separate from the performance-only benchmark."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.integrate import quad

SCRIPT = Path(__file__).with_name("example_potential_grid_convergence.py")
spec = importlib.util.spec_from_file_location("plummer_accuracy_reference", SCRIPT)
assert spec is not None and spec.loader is not None
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def test_analytic_central_potential_matches_shell_integral():
    scale = reference.PLUMMER_SCALE_KPC
    truncation = reference.PLUMMER_TRUNCATION_KPC
    fraction = truncation**3 / (truncation**2 + scale**2)**1.5
    rho0 = 3 * reference.PLUMMER_TOTAL_MASS_MSUN / (4 * np.pi * scale**3 * fraction)
    integral = quad(lambda r: rho0 * r * (1 + (r/scale)**2)**-2.5,
                    0, truncation, epsabs=0, epsrel=1e-12)[0]
    expected = -4 * np.pi * 4.30091727067736e-6 * integral
    actual = reference.analytic_plummer(np.zeros(1))[0][0]
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_analytic_boundary_matches_kepler_exterior():
    radii = np.array([reference.PLUMMER_TRUNCATION_KPC, 20.])
    phi, force = reference.analytic_plummer(radii)
    gm = 4.30091727067736e-6 * reference.PLUMMER_TOTAL_MASS_MSUN
    np.testing.assert_allclose(phi, -gm/radii, rtol=1e-14)
    np.testing.assert_allclose(force, -gm/radii**2, rtol=1e-14)
