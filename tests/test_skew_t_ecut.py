import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import gaussian_kde
from types import SimpleNamespace

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


def _two_peak_sample():
    """Gaussian + Gaussian, restricted to the physical energy interval."""
    rng = np.random.default_rng(21)
    energies = np.concatenate([
        rng.normal(-0.78, 0.05, 700),
        rng.normal(-0.35, 0.08, 900),
    ])
    return energies[(energies >= -1.0) & (energies <= 0.0)]


def test_fit_uses_binned_twice_scott_kde_and_scaled_trf(monkeypatch):
    """Binned convolution keeps the KDE bandwidth and the scaled TRF contract."""
    energies = _two_peak_sample()
    original = util.least_squares
    captured = {}

    def capture(fun, x0, **kwargs):
        closure = dict(zip(fun.__code__.co_freevars,
                           [cell.cell_contents for cell in fun.__closure__]))
        captured.update(closure=closure, start=x0.copy(), kwargs=kwargs)
        result = original(fun, x0, **kwargs)
        captured['result'] = result
        return result

    monkeypatch.setattr(util, 'least_squares', capture)
    params, cut, _ = util._fit_skew_t_with_params(energies)
    assert params is not None and np.isfinite(cut)
    assert captured['result'].success
    closure = captured['closure']
    x = closure['xc']
    kde = gaussian_kde(energies, bw_method=lambda obj: 2.0 * obj.scotts_factor())
    expected = kde(x)
    expected /= expected.sum() * (x[1] - x[0])
    np.testing.assert_allclose(closure['h_sm'], expected, rtol=0,
                               atol=2e-4 * expected.max())
    binned = util._binned_gaussian_density(energies, x)
    binned /= binned.sum() * (x[1] - x[0])
    np.testing.assert_allclose(closure['h_sm'], binned, rtol=1e-12)
    kwargs = captured['kwargs']
    assert kwargs['method'] == 'trf'
    assert kwargs['jac'] == '2-point'
    assert kwargs['loss'] == 'linear'
    assert kwargs['max_nfev'] == 600
    start = captured['start']
    np.testing.assert_allclose(kwargs['x_scale'],
                               [1, start[2], start[2], 1, start[4],
                                start[6], start[6], 1, start[8]])
    lower, upper = kwargs['bounds']
    assert np.all(start >= lower) and np.all(start <= upper)


def test_nonconverged_fit_is_not_reported_as_success(monkeypatch):
    def fail(fun, x0, **kwargs):
        return SimpleNamespace(success=False, x=x0, fun=fun(x0),
                               message='budget exhausted')

    monkeypatch.setattr(util, 'least_squares', fail)
    with pytest.warns(RuntimeWarning, match='did not converge'):
        assert util._fit_skew_t_with_params(_two_peak_sample()) == (None, None, None)


def test_solver_exception_does_not_return_initial_guess(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('solver failure')

    monkeypatch.setattr(util, 'least_squares', fail)
    with pytest.warns(RuntimeWarning, match='solver failure'):
        assert util.get_Ecut_skewt(_two_peak_sample()) is None


@pytest.mark.parametrize('energies', [np.ones(100), np.full(49, -0.5),
                                     np.array([np.nan, np.inf, -np.inf])])
def test_unusable_samples_do_not_enter_optimizer(energies):
    assert util._fit_skew_t_with_params(energies) == (None, None, None)


@pytest.mark.parametrize('energies', [
    _two_peak_sample(),
    np.concatenate([np.full(900, -1.0), np.full(100, -0.02)]),
])
def test_binned_density_matches_direct_kde_at_edges_and_in_tails(energies):
    """Do not reflect endpoint particles or discard the density outside them."""
    h = 2.0 * energies.std(ddof=1) * len(energies) ** (-1.0 / 5.0)
    x = np.linspace(energies.min() - 6.0 * h, energies.max() + 6.0 * h, 3001)
    expected = gaussian_kde(energies, bw_method=lambda k: 2.0 * k.scotts_factor())(x)
    density = util._binned_gaussian_density(energies, x)
    np.testing.assert_allclose(density, expected, rtol=0, atol=2e-4 * expected.max())
    assert np.all(np.isfinite(density)) and np.all(density >= 0)
    assert np.isclose(np.trapezoid(density, x), 1.0, atol=2e-5)


def test_binned_density_is_affine_equivariant():
    """Changing the energy unit cannot change the smoothing scale in sigma units."""
    energies = _two_peak_sample()
    x = np.linspace(-1.1, 0.1, 500)
    density = util._binned_gaussian_density(energies, x)
    transformed = util._binned_gaussian_density(3.0 * energies + 0.2, 3.0 * x + 0.2)
    np.testing.assert_allclose(transformed * 3.0, density, rtol=1e-11, atol=1e-12)
