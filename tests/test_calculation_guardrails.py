import numpy as np
import pytest
from types import SimpleNamespace

from kinematic_decompose.gravity.kinematic_solver import (
    _circular_velocity_from_radial_force,
)
from kinematic_decompose import pipeline


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
