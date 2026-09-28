"""Regression tests for the fitted-source softened Multipole route."""
import numpy as np

from kinematic_decompose.potential import Potential


def _build(pos, mass, softening, *, symmetry="a", lmax=4):
    return Potential(
        type="Multipole", particles=(pos, mass), softening=softening,
        symmetry=symmetry, lmax=lmax, mmax=lmax,
        gridSizeR=30, rmin=0.01, rmax=20.0,
    )


def test_sparse_source_keeps_original_particle_kernel():
    """A poorly sampled radial/harmonic fit must not alter sparse gas forces."""
    rng = np.random.default_rng(10)
    positions = rng.normal(size=(350, 3))
    masses = rng.uniform(0.5, 1.5, len(positions))
    queries = rng.normal(size=(25, 3))
    scalar = _build(positions, masses, 0.8)
    per_particle = _build(positions, masses, np.full(len(masses), 0.8))
    np.testing.assert_array_equal(scalar.potential(queries), per_particle.potential(queries))
    np.testing.assert_array_equal(scalar.force(queries), per_particle.force(queries))


def test_fitted_convolution_retains_mass_at_origin():
    """A central massive particle is added analytically, not dropped by log-r fitting."""
    rng = np.random.default_rng(16)
    positions = rng.normal(size=(1600, 3))
    masses = rng.uniform(0.5, 1.5, len(positions))
    center_mass = 10.0
    points = np.array([[0., 0., 0.], [0.1, 0., 0.], [0.5, 0., 0.], [3., 0., 0.]])
    other = _build(positions, masses, 0.8)
    combined = _build(np.vstack((positions, np.zeros((1, 3)))),
                      np.append(masses, center_mass), 0.8)
    central = _build(np.zeros((1, 3)), np.array([center_mass]), 0.8)
    np.testing.assert_allclose(combined.potential(points) - other.potential(points),
                               central.potential(points), rtol=2e-6, atol=1e-12)
    # Interpolation of the l=0 potential uses log scaling, so evaluation of
    # separately interpolated components is not exactly additive off-grid.
    np.testing.assert_allclose(combined.force(points) - other.force(points),
                               central.force(points), rtol=5e-5, atol=1e-12)
