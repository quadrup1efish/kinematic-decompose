"""Density API, Poisson consistency, serialization and unit regressions."""
import warnings

import numpy as np
import pytest

from kinematic_decompose import potential as backend


@pytest.fixture(autouse=True)
def reset_backend():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        backend.setUnits()
        backend.set_num_threads(1)
        yield
        backend.setUnits()
        backend.set_num_threads(1)


@pytest.fixture
def model():
    rng = np.random.default_rng(671)
    return backend.Potential(
        particles=(rng.normal(size=(128, 3)), rng.uniform(0.5, 1.5, 128)),
        softening=0.4, symmetry="a", lmax=4, mmax=0,
        rmin=0.02, rmax=20, gridSizeR=60,
    )


def test_density_input_shapes(model):
    point = [0.4, -0.3, 0.7]
    scalar = model.density(point)
    assert isinstance(scalar, float)
    assert np.isfinite(scalar)
    assert scalar == model.density([point])[0]
    assert model.density(np.empty((0, 3))).shape == (0,)
    points = np.arange(30, dtype=np.float32).reshape(5, 6)[:, ::2] / 10
    np.testing.assert_array_equal(model.density(points), model.density(points.copy()))
    np.testing.assert_array_equal(model.density(points.tolist()), model.density(points))


@pytest.mark.parametrize("points", [0.0, [], [1, 2], [1, 2, 3, 4], [[1, 2]], np.zeros((2, 3, 1))])
def test_density_rejects_invalid_shapes(model, points):
    with pytest.raises(ValueError, match="coordinates must have shape"):
        model.density(points)


def test_density_composite_and_roundtrip(model, tmp_path):
    points = np.array([[0, 0, 0], [0, 0, 1], [0.3, 0.4, 0.5], [40, -20, 5]])
    density = model.density(points)
    # Preserve upstream's exact-origin extrapolation semantics (which may
    # produce NaN); do not silently clip or replace the numerical result.
    assert np.isfinite(density[1:]).all()
    composite = backend.Potential(model, model)
    np.testing.assert_array_equal(composite.density(points), 2 * density)
    for label, pot in [("single", model), ("composite", composite)]:
        path = tmp_path / (label + ".ini")
        pot.export(path)
        np.testing.assert_allclose(backend.Potential(path).density(points), pot.density(points),
                                   rtol=2e-8, atol=1e-12)


def test_density_poisson_equation(model):
    points = np.array([[0.4, 0.3, 0.7], [1.3, -0.5, 0.8], [2, 1, -0.3]])
    step = 1e-5
    divergence = np.zeros(len(points))
    for axis in range(3):
        offset = np.eye(3)[axis] * step
        divergence += (model.force(points + offset)[:, axis]
                       - model.force(points - offset)[:, axis]) / (2 * step)
    np.testing.assert_allclose(model.density(points), -divergence / (4 * np.pi * backend.G),
                               rtol=2e-6, atol=1e-9)


def test_density_unit_scaling(model):
    points = np.array([[0.4, 0.3, 0.7], [1.3, -0.5, 0.8]])
    # Keep the same physical native potential and locations, changing only
    # external units. Changing units after construction is deliberate here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        backend.setUnits(length=2.5, mass=3e9, velocity=170)
    # Independently verify scaling via Poisson's equation in external units;
    # this does not rely on a separately fitted particle density.
    step = 1e-5
    divergence = np.zeros(len(points))
    for axis in range(3):
        offset = np.eye(3)[axis] * step
        divergence += (model.force(points + offset)[:, axis]
                       - model.force(points - offset)[:, axis]) / (2 * step)
    actual = model.density(points)
    np.testing.assert_allclose(actual, -divergence / (4 * np.pi * backend.G), rtol=2e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        backend.setUnits(length=2.5, mass=21e9, velocity=170)
    np.testing.assert_allclose(model.density(points), actual / 7, rtol=2e-15)


def test_density_thread_determinism(model):
    points = np.random.default_rng(82).normal(size=(8192, 3))
    serial = model.density(points)
    backend.set_num_threads(2)
    np.testing.assert_array_equal(model.density(points), serial)


@pytest.mark.parametrize("symmetry", ["n", "t"])
@pytest.mark.parametrize("z", [1.0, -1.0])
@pytest.mark.parametrize("softening", [None, 0.4])
def test_density_nonaxisymmetric_exact_axis_poisson(symmetry, z, softening, tmp_path):
    positions = np.array([[0.2, -0.1, 0.4], [1.2, 0.8, -0.5],
                          [-0.7, 0.3, 1.1], [0.05, 0.2, 0.15]])
    model = backend.Potential(particles=(positions, np.array([1.0, 2.0, 0.7, 0.2])),
                              softening=softening, symmetry=symmetry, lmax=4, mmax=4,
                              rmin=0.02, rmax=20, gridSizeR=60)
    point = np.array([[0.0, 0.0, z]])
    step = 1e-5
    divergence = sum(
        (model.force(point + np.eye(3)[axis] * step)[0, axis]
         - model.force(point - np.eye(3)[axis] * step)[0, axis]) / (2 * step)
        for axis in range(3)
    )
    np.testing.assert_allclose(model.density(point), -divergence / (4 * np.pi * backend.G),
                               rtol=1e-2, atol=1e-5)

    # The analytic axis limit must not depend on an arbitrary azimuth, or jump
    # inside the numerical axis branch. No shifted coordinate is used as truth.
    radius = abs(z) * np.cbrt(np.finfo(float).eps) / 2
    near_axis = np.array([[radius, 0, z], [-radius, 0, z],
                          [0, radius, z], [0, -radius, z]])
    np.testing.assert_allclose(model.density(near_axis), model.density(point)[0],
                               rtol=1e-12, atol=1e-12)
    # Round trips and composite dispatch must use the same repaired path.
    exported = tmp_path / "nonaxisymmetric.ini"
    model.export(exported)
    loaded = backend.Potential(exported)
    np.testing.assert_allclose(loaded.density(point), model.density(point),
                               rtol=2e-8, atol=1e-12)
    np.testing.assert_allclose(loaded.force(point), model.force(point),
                               rtol=2e-8, atol=1e-12)
    composite = backend.Potential(model, loaded)
    np.testing.assert_allclose(composite.force(point), 2 * model.force(point),
                               rtol=2e-8, atol=1e-12)
    np.testing.assert_allclose(composite.density(point), 2 * model.density(point),
                               rtol=2e-8, atol=1e-12)


@pytest.mark.parametrize("z", [1.0, -1.0, 40.0, -40.0])
def test_nonaxisymmetric_axis_force_matches_potential_gradient(z):
    positions = np.array([[0.2, -0.1, 0.4], [1.2, 0.8, -0.5],
                          [-0.7, 0.3, 1.1], [0.05, 0.2, 0.15]])
    model = backend.Potential(particles=(positions, np.array([1.0, 2.0, 0.7, 0.2])),
                              softening=0.4, symmetry="n", lmax=4, mmax=4,
                              rmin=0.02, rmax=20, gridSizeR=60)
    point = np.array([[0.0, 0.0, z]])
    step = abs(z) * 1e-6
    expected = np.array([
        -(model.potential(point + np.eye(3)[axis] * step)[0]
          - model.potential(point - np.eye(3)[axis] * step)[0]) / (2 * step)
        for axis in range(3)
    ])
    np.testing.assert_allclose(model.force(point)[0], expected, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize("symmetry", ["n", "t"])
@pytest.mark.parametrize("z", [1.0, -1.0])
@pytest.mark.parametrize("softening", [None, 0.4])
def test_near_axis_density_branch_boundary(symmetry, z, softening):
    positions = np.array([[0.2, -0.1, 0.4], [1.2, 0.8, -0.5],
                          [-0.7, 0.3, 1.1], [0.05, 0.2, 0.15]])
    model = backend.Potential(particles=(positions, np.array([1.0, 2.0, 0.7, 0.2])),
                              softening=softening, symmetry=symmetry, lmax=4, mmax=4,
                              rmin=0.02, rmax=20, gridSizeR=60)
    axis_density = model.density([0, 0, z])
    boundary = abs(z) * np.cbrt(np.finfo(float).eps)
    # Avoid an ambiguous equality at the floating-point threshold. Probe
    # both sides, along cardinal and diagonal azimuths on both polar axes.
    azimuths = np.arange(8) * np.pi / 4
    radial_factors = np.array([0.5, 1 - 1e-3, 1 + 1e-3, 2.0])
    radii = boundary * radial_factors[:, None]
    x, y = radii * np.cos(azimuths), radii * np.sin(azimuths)
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z)))
    density = model.density(points).reshape(len(radial_factors), len(azimuths))
    assert np.isfinite(density).all()
    np.testing.assert_allclose(density[:2], axis_density, rtol=1e-12, atol=1e-12)
    # Normalize the allowed transition error by the Hessian scale rather
    # than density, which can vanish by physical cancellation. The bound
    # scales with the epsilon^(1/3) axis-limit truncation, not a density floor.
    step = abs(z) * 1e-5
    axis = np.array([[0.0, 0.0, z]])
    force_derivatives = np.array([
        (model.force(axis + np.eye(3)[j] * step)[0]
         - model.force(axis - np.eye(3)[j] * step)[0]) / (2 * step)
        for j in range(3)
    ])
    curvature_scale = np.max(np.abs(force_derivatives)) / (4 * np.pi * backend.G)
    transition_tolerance = 32 * np.cbrt(np.finfo(float).eps) * curvature_scale
    np.testing.assert_allclose(density[2:], axis_density, rtol=0, atol=transition_tolerance)
    np.testing.assert_allclose(density[2], density[1], rtol=0, atol=transition_tolerance)
