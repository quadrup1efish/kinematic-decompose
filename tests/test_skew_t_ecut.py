import numpy as np

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
