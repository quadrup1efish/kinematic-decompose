import numpy as np
import pynbody
from pynbody.array import SimArray
from pynbody import units

from kinematic_decompose.PyTNG.snapshot_loader import Snapshot


def _snapshot_with_positions(positions, boxsize=None):
    container = pynbody.new(star=len(positions))
    container.star["pos"] = SimArray(np.asarray(positions, dtype=float), units.kpc)
    if boxsize is not None:
        container.properties["boxsize"] = SimArray(float(boxsize), units.kpc)
    snapshot = Snapshot.__new__(Snapshot)
    snapshot.container = container
    snapshot.group_catalog = {}
    return snapshot


def test_explicit_center_wraps_particles_across_periodic_boundary():
    snapshot = _snapshot_with_positions(
        [[0.2, 5.0, 5.0], [9.7, 5.0, 5.0], [0.1, 5.2, 5.0]], boxsize=10.0
    )

    snapshot.center(cen=SimArray([9.8, 5.0, 5.0], units.kpc), with_velocity=False)

    positions = np.asarray(snapshot.container.star["pos"])
    np.testing.assert_allclose(positions[:, 0], [0.4, -0.1, 0.3], atol=1e-6)
    assert np.max(np.linalg.norm(positions, axis=1)) < 1.0
    assert np.isclose(np.linalg.norm(positions[0] - positions[1]), 0.5)


def test_explicit_center_keeps_nonperiodic_translation_when_boxsize_missing():
    snapshot = _snapshot_with_positions([[12.0, 0.0, 0.0]])

    snapshot.center(cen=SimArray([10.0, 0.0, 0.0], units.kpc), with_velocity=False)

    np.testing.assert_allclose(snapshot.container.star["pos"], [[2.0, 0.0, 0.0]])


def test_automatic_center_still_uses_pynbody_wrap(monkeypatch):
    snapshot = _snapshot_with_positions([[1.0, 0.0, 0.0]], boxsize=10.0)
    calls = {}

    def fake_center(container, **kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(pynbody.analysis, "center", fake_center)
    snapshot.center(cen=None, with_velocity=False)

    assert calls["wrap"] is True
    assert calls["with_velocity"] is False
