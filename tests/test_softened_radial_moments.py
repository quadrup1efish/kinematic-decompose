import numpy as np

from kinematic_decompose import potential as native_potential
from kinematic_decompose.potential import Potential


def test_narrow_softened_monopole_is_resolved_independently_of_radial_grid():
    """A compact kernel narrower than every radial cell must retain its mass."""
    particle_position = np.array([[0.3, 0.0, 0.0]])
    particle_mass = np.array([1.0])
    rmin, rmax = 0.01, 2.0

    for grid_size in (30, 60):
        radii = np.exp(np.linspace(np.log(rmin), np.log(rmax), grid_size))
        exterior = radii > 0.6
        query_points = np.column_stack((radii[exterior], np.zeros((exterior.sum(), 2))))
        expected = -native_potential.G / radii[exterior]

        for support in (3e-4, 3e-5):
            potential = Potential(
                type="Multipole",
                particles=(particle_position, particle_mass),
                softening=support,
                symmetry="s",
                lmax=0,
                mmax=0,
                gridSizeR=grid_size,
                rmin=rmin,
                rmax=rmax,
            )
            actual = np.asarray(potential.potential(query_points)).ravel()
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)


def test_high_order_origin_overlapping_kernel_matches_exterior_multipole_series():
    """Small-a/h angular moments must not blow up near the radial origin."""
    for particle_radius, support in ((1e-3, 0.8), (0.1, 0.8)):
        grid_size, rmin, rmax = 80, 1e-4, 20.0
        radii = np.exp(np.linspace(np.log(rmin), np.log(rmax), grid_size))
        radius = radii[70]
        potential = Potential(
            type="Multipole",
            particles=(np.array([[particle_radius, 0.0, 0.0]]), np.array([1.0])),
            softening=support,
            symmetry="n",
            lmax=8,
            mmax=8,
            gridSizeR=grid_size,
            rmin=rmin,
            rmax=rmax,
        )

        actual = float(np.asarray(potential.potential([[radius, 0.0, 0.0]])).ravel()[0])
        expected = -native_potential.G / radius * sum((particle_radius / radius) ** ell for ell in range(9))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)


def test_softened_lmax2_exterior_field_matches_quadrupole_series():
    """The lmax=2 interpolation path must retain the full quadrupole field."""
    particle_radius, support, radius, angle = 1.0, 0.1, 5.0, 0.73
    potential = Potential(
        type="Multipole",
        particles=(np.array([[particle_radius, 0.0, 0.0]]), np.array([1.0])),
        softening=support,
        symmetry="n",
        lmax=2,
        mmax=2,
        gridSizeR=80,
        rmin=0.01,
        rmax=10.0,
    )
    point = np.array([[radius * np.cos(angle), radius * np.sin(angle), 0.0]])
    cosine = np.cos(angle)
    p2 = 0.5 * (3.0 * cosine**2 - 1.0)
    expected = -native_potential.G / radius * (
        1.0 + (particle_radius / radius) * cosine
        + (particle_radius / radius) ** 2 * p2
    )

    actual = float(np.asarray(potential.potential(point)).ravel()[0])

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)
