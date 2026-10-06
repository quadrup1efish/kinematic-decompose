import numpy as np
import pytest
from types import SimpleNamespace
from typing import cast

import pynbody
from pynbody.array import SimArray

from kinematic_decompose.gravity.kinematic_solver import (
    _circular_velocity_from_radial_force,
)
from kinematic_decompose import pipeline
from kinematic_decompose.gravity import kinematic_solver as solver
from kinematic_decompose.PyTNG import derived_array  # noqa: F401
from kinematic_decompose.potential import Potential


def test_circular_velocity_uses_inward_force_and_rejects_outward_force():
    radii = np.array([3.0, 3.0])
    radial_force = np.array([-2.0, 2.0])

    speeds = _circular_velocity_from_radial_force(radii, radial_force)

    np.testing.assert_allclose(speeds[0], np.sqrt(6.0))
    assert np.isnan(speeds[1])


def test_pipeline_uses_potential_fallback_when_skew_t_cut_is_none(monkeypatch):
    class FakeScaler:
        def fit_transform(self, values):
            return values

        def transform(self, value, columns=None):
            return value

        def inverse_transform_GMM(self, model):
            return model

    class FakeAutoGMM:
        def __init__(self, random_state):
            self.best_model = object()

        def fit(self, values, **kwargs):
            return self

    n = 120
    galaxy = SimpleNamespace(
        s={
            "eoemin": np.linspace(-0.9, -0.1, n),
            "jzojc": np.zeros(n),
            "jpojc": np.zeros(n),
            "mass": np.ones(n),
            "e": np.linspace(-1.0, -0.2, n),
        },
        properties={"eps": 0.1},
    )
    potential = SimpleNamespace(
        potential=lambda points: -1.0 / (np.linalg.norm(points, axis=1) + 1.0)
    )

    monkeypatch.setattr(pipeline.util, "JEHistogram", lambda *args, **kwargs: (np.ones(n, bool), None))
    monkeypatch.setattr(pipeline.util, "get_Ecut_skewt", lambda *args: None)
    monkeypatch.setattr(pipeline.preprocessing, "RobustScaler", FakeScaler)
    monkeypatch.setattr(pipeline, "AutoGaussianMixtureModel", FakeAutoGMM)

    _, _, cut, _ = pipeline.train_auto_gaussian_mixture_model(galaxy, potential)

    assert np.isfinite(cut)


def test_jehistogram_includes_maximum_energy_in_last_bin():
    spheroid, disk = pipeline.util.JEHistogram(
        np.array([0.0, 0.5, 1.0]), np.zeros(3), n_E=2, n_eps=4
    )

    assert spheroid[-1]
    assert not disk[-1]


def test_robust_scaler_ignores_nan_for_scale_and_handles_constant_feature():
    values = np.array([[1.0, 5.0], [np.nan, 5.0], [3.0, 5.0]])
    scaler = pipeline.preprocessing.RobustScaler()

    transformed = scaler.fit_transform(values)

    np.testing.assert_allclose(transformed[[0, 2], 0], [-1.0, 1.0])
    assert np.isnan(transformed[1, 0])
    np.testing.assert_array_equal(transformed[:, 1], np.zeros(3))
    assert np.all(scaler.scale_ == np.array([1.0, 1.0]))


def test_robust_scaler_rejects_all_nan_feature():
    with pytest.raises(ValueError, match="all-NaN feature"):
        pipeline.preprocessing.RobustScaler().fit(
            np.array([[np.nan, 1.0], [np.nan, 2.0]])
        )


def test_require_bulge_halo_preserves_soft_probabilities_for_unchanged_labels():
    class FakeGMM:
        means_ = np.array([[-0.4, 0.0], [0.0, 0.9]])

        def _estimate_log_prob_resp(self, values):
            return None, np.log(np.array([[0.9, 0.1], [0.2, 0.8]]))

    galaxy = SimpleNamespace(
        s={"eoemin": np.array([-0.8, -0.2]), "jzojc": np.zeros(2)}
    )

    result = pipeline.util.decompose(
        np.zeros((2, 2)), galaxy, FakeGMM(), eoemin_cut=-0.5,
        jzojc_cut=0.5, predict_method="hard", require_bulge_halo=True,
    )

    np.testing.assert_array_equal(result.s["label"], [2, 0])
    np.testing.assert_array_equal(result.s["prob"][0], [0, 0, 1, 0, 0])
    np.testing.assert_allclose(result.s["prob"][1], [0.8, 0, 0, 0.2, 0])


# Azimuthally averaged circular references.

class FourierPotential:
    """Analytic planar harmonic field, with a negative energy offset."""

    def __init__(self, order=4, phase=0.0, amplitude=0.2):
        self.order = order
        self.phase = phase
        self.amplitude = amplitude

    def potential(self, points):
        radius = np.linalg.norm(points[:, :2], axis=1)
        angle = np.arctan2(points[:, 1], points[:, 0]) - self.phase
        return -10 + 0.5 * radius**2 * (1 + self.amplitude * np.cos(self.order * angle))

    def force(self, points):
        radius = np.linalg.norm(points[:, :2], axis=1)
        angle = np.arctan2(points[:, 1], points[:, 0])
        radial = -radius * (1 + self.amplitude * np.cos(self.order * (angle - self.phase)))
        tangential = 0.5 * self.amplitude * self.order * radius * np.sin(self.order * (angle - self.phase))
        return np.column_stack((radial * np.cos(angle) - tangential * np.sin(angle),
                                radial * np.sin(angle) + tangential * np.cos(angle),
                                np.zeros_like(radius)))


@pytest.mark.parametrize('order', [1, 2, 3, 4])
@pytest.mark.parametrize('phase', [0.0, 0.137, 0.8])
def test_ring_average_eliminates_default_fourier_orders(order, phase):
    radius = np.geomspace(0.01, 3, 30)
    phi, force = solver._azimuthally_averaged_midplane(FourierPotential(order, phase), radius)
    np.testing.assert_allclose(phi, -10 + 0.5 * radius**2, atol=2e-15)
    np.testing.assert_allclose(force, -radius, rtol=1e-14)


def test_four_point_average_aliases_m4_but_eight_does_not():
    radius = np.array([1.0, 2.0])
    model = FourierPotential()
    _, four = solver._azimuthally_averaged_midplane(model, radius, azimuth_samples=4)
    _, eight = solver._azimuthally_averaged_midplane(model, radius, azimuth_samples=8)
    np.testing.assert_allclose(four, -1.2 * radius)
    np.testing.assert_allclose(eight, -radius)


def test_axisymmetric_average_preserves_legacy_reference():
    radius = np.geomspace(0.01, 3, 30)
    model = FourierPotential(amplitude=0)
    single = solver._azimuthally_averaged_midplane(model, radius, azimuth_samples=1)
    averaged = solver._azimuthally_averaged_midplane(model, radius)
    for old, new in zip(single, averaged):
        np.testing.assert_allclose(new, old, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize('samples', [0, -1, 2.5])
def test_invalid_azimuth_samples_rejected(samples):
    with pytest.raises(ValueError, match='positive integer'):
        solver._azimuthally_averaged_midplane(FourierPotential(), np.array([1.0]), samples)


@pytest.mark.parametrize('radii', [[0.0], [-1.0], [np.nan], [[1.0]]])
def test_invalid_reference_radii_rejected(radii):
    with pytest.raises(ValueError, match='positive 1D'):
        solver._azimuthally_averaged_midplane(FourierPotential(), radii)


def test_default_component_contract_and_legacy_override(monkeypatch):
    family = {'pos': np.ones((12, 3)), 'mass': np.ones(12)}
    galaxy = SimpleNamespace(dm=family, gas=family, s=family,
                             properties={'eps': 0.1, 'mDM': 2.0}, R_vir=10.0)
    calls = []
    monkeypatch.setattr(solver, 'create_multipole_potential',
                        lambda *args, **kwargs: calls.append(kwargs) or object())
    monkeypatch.setattr(solver, 'Potential', lambda *parts: parts)
    solver.construct_galaxy_potential_model(galaxy)
    assert [v['symmetry'] for v in calls] == ['s', 'a', 'n']
    assert [v['lmax'] for v in calls] == [0, 4, 4]
    assert all(v['gridsizeR'] == 30 for v in calls)
    calls.clear()
    solver.construct_galaxy_potential_model(galaxy, stellar_symmetry='a')
    assert [v['symmetry'] for v in calls] == ['s', 'a', 'a']


def test_particle_potentials_remain_unaveraged():
    galaxy = pynbody.new(star=8)
    angles = 2 * np.pi * np.arange(8) / 8
    galaxy.s['pos'] = SimArray(np.column_stack((np.cos(angles), np.sin(angles), np.zeros(8))), 'kpc')
    galaxy.s['vel'] = SimArray(np.column_stack((-np.sin(angles), np.cos(angles), np.zeros(8))), 'km s^-1')
    galaxy.s['mass'] = SimArray(np.ones(8), 'Msol')
    model = FourierPotential()
    expected = model.potential(np.asarray(galaxy.s['pos']))
    solver.calculate_kinematic_param(galaxy, cast(Potential, model))
    np.testing.assert_array_equal(np.asarray(galaxy.s['phi']), expected)
    assert np.ptp(expected) > 0
    assert np.isfinite(galaxy.s['jc']).all()


def test_native_ring_average_is_rotation_invariant():
    rng = np.random.default_rng(204)
    positions = rng.normal(size=(180, 3)) * [2, 0.6, 0.3]
    model = Potential(particles=(positions, np.ones(180)), softening=0.2,
                      symmetry='n', lmax=4, mmax=4, gridSizeR=30)
    class Rotated:
        def __init__(self, angle):
            c, s = np.cos(angle), np.sin(angle)
            self.rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        def potential(self, points):
            return model.potential(points @ self.rotation.T)
        def force(self, points):
            return model.force(points @ self.rotation.T) @ self.rotation
    radii = np.geomspace(0.02, 5, 50)
    reference = solver._azimuthally_averaged_midplane(model, radii)
    rotated = solver._azimuthally_averaged_midplane(Rotated(0.317), radii)
    for before, after in zip(reference, rotated):
        np.testing.assert_allclose(after, before, rtol=2e-13, atol=1e-12)
