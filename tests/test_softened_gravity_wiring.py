import numpy as np

from kinematic_decompose.gravity import kinematic_solver
from kinematic_decompose.gravity.kinematic_solver import create_multipole_potential


def test_create_multipole_potential_uses_native_softening():
    potential = create_multipole_potential(
        positions=np.array([[0.0, 0.0, 0.0]]),
        masses=np.array([1.0e10]),
        softening=0.8,
        symmetry="s",
        rmin=0.05,
        rmax=100.0,
        lmax=0,
        gridsizeR=30,
    )
    values = potential.potential(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    assert np.all(np.isfinite(values))
    assert values[0] < values[1] < 0


def test_create_multipole_potential_rejects_nonpositive_softening():
    with np.testing.assert_raises_regex(ValueError, "positive"):
        create_multipole_potential(
            positions=np.array([[0.0, 0.0, 0.0]]),
            masses=np.array([1.0e10]),
            softening=0.0,
        )


def test_galaxy_constructor_uses_one_kernel_scale_for_all_components(monkeypatch):
    class Family(dict):
        pass

    class Galaxy:
        properties = {"eps": 0.288, "mDM": 1.0e10}
        R_vir = 100.0
        dm = Family(pos=np.ones((10, 3)))
        gas = Family(pos=np.ones((10, 3)), mass=np.ones(10))
        s = Family(pos=np.ones((10, 3)), mass=np.ones(10))

    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(kinematic_solver, "create_multipole_potential", capture)
    monkeypatch.setattr(kinematic_solver, "Potential", lambda *parts: parts)
    kinematic_solver.construct_galaxy_potential_model(Galaxy())

    assert len(calls) == 3
    assert {call["softening"] for call in calls} == {2.8 * 0.288}
    assert {call["rmin"] for call in calls} == {0}
