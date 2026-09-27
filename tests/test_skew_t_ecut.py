import numpy as np
from scipy.optimize import brentq

from kinematic_decompose.mixture import util


def test_get_ecut_skewt_returns_finite_boundary_for_two_component_sample():
    rng = np.random.default_rng(42)
    energies = np.concatenate([
        rng.normal(-0.78, 0.05, 600),
        rng.normal(-0.40, 0.08, 600),
    ])
    masses = np.ones(energies.size)

    cut = util.get_Ecut_skewt(energies, masses)

    assert np.isfinite(cut)
    assert np.quantile(energies, 0.01) <= cut <= np.quantile(energies, 0.99)


def test_skew_t_fit_applies_finite_energy_mask_to_masses():
    rng = np.random.default_rng(19)
    energies = np.concatenate([
        rng.normal(-0.78, 0.05, 300),
        rng.normal(-0.40, 0.08, 300),
    ])
    masses = np.ones(energies.size)
    energies[17] = np.nan

    _, cut, _ = util._fit_skew_t_with_params(energies, masses)

    assert np.isfinite(cut)


def _component_parameters(params):
    w1, m1, s1, a1, inv_nu1, m2, s2, a2, inv_nu2 = params
    nu1 = 1.0 / max(inv_nu1, 1e-6)
    nu2 = 1.0 / max(inv_nu2, 1e-6)
    mode1 = util._skew_t_mode(m1, s1, a1, nu1)
    mode2 = util._skew_t_mode(m2, s2, a2, nu2)
    return w1, nu1, nu2, mode1, mode2, m1, s1, a1, m2, s2, a2


def test_resolvable_components_return_the_fitted_crossing():
    """sep >= d_min: the returned cut is the fitted component-density crossing."""
    rng = np.random.default_rng(7)
    energies = np.concatenate([
        rng.normal(-0.80, 0.06, 800),
        rng.normal(-0.30, 0.06, 800),
    ])
    params, cut, separation = util._fit_skew_t_with_params(
        energies, np.ones(energies.size)
    )

    assert separation >= 1.0, "synthetic components are well separated"

    w1, nu1, nu2, mode1, mode2, m1, s1, a1, m2, s2, a2 = _component_parameters(params)

    def difference(x):
        return (
            w1 * util._skew_t_pdf(x, m1, s1, a1, nu1)
            - (1.0 - w1) * util._skew_t_pdf(x, m2, s2, a2, nu2)
        )

    lo, hi = min(mode1, mode2), max(mode1, mode2)
    assert difference(lo) * difference(hi) <= 0, "components must cross between modes"
    expected = float(brentq(difference, lo, hi))

    assert np.isclose(cut, expected, rtol=0.0, atol=1e-9)


def test_unresolvable_components_fall_back_to_get_ecut():
    """sep < d_min: indistinguishable components, so get_Ecut decides."""
    rng = np.random.default_rng(11)
    energies = np.concatenate([
        rng.normal(-0.60, 0.12, 800),
        rng.normal(-0.50, 0.12, 800),
    ])
    masses = np.ones(energies.size)

    params, cut, separation = util._fit_skew_t_with_params(energies, masses)

    assert separation < 1.0, "overlapping synthetic components are unresolvable"

    expected = util.get_Ecut(energies, masses, M_bin=100, m_bin=25, Mmin=0.1)
    assert expected is not None and expected != 0.0
    assert cut == float(expected)
