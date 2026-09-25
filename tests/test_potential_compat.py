import numpy as np
import pytest

agama = pytest.importorskip("agama")
from kinematic_decompose import potential as native_potential
from kinematic_decompose.potential import Potential


@pytest.mark.parametrize("symmetry,lmax,mmax", [("s", 0, 0), ("a", 4, 0), ("n", 4, 4)])
def test_multipole_matches_agama(symmetry, lmax, mmax):
    agama.setUnits()
    native_potential.setUnits()
    positions = np.array([[0.2, -0.1, 0.4], [1.2, 0.8, -0.5], [-0.7, 0.3, 1.1]])
    masses = np.array([1.0, 2.0, 0.7])
    points = np.array([[0.1, 0.2, -0.3], [1.5, 0.1, 0.8], [4.0, -2.0, 1.0]])
    args = dict(type="Multipole", particles=(positions, masses), softening=0.45,
                symmetry=symmetry, lmax=lmax, mmax=mmax,
                rmin=0, rmax=20, gridSizeR=40)

    expected = agama.Potential(**args)
    actual = Potential(**args)
    np.testing.assert_allclose(actual.potential(points), expected.potential(points), rtol=2e-11, atol=1e-12)
    np.testing.assert_allclose(actual.force(points), expected.force(points), rtol=2e-10, atol=1e-11)


def test_agama_ini_roundtrip_and_composite(tmp_path):
    agama.setUnits()
    positions = np.array([[0.0, 0.0, 0.0], [1.5, -0.2, 0.4]])
    masses = np.array([1.2, 2.3])
    points = np.array([[0.2, 0.1, 0.3], [2.0, 0.0, -0.5], [5.0, 1.0, 0.2]])
    args = dict(type="Multipole", particles=(positions, masses), softening=0.3,
                symmetry="s", lmax=0, mmax=0, rmin=0, rmax=20, gridSizeR=40)

    native = Potential(**args)
    path = tmp_path / "multipole.ini"
    native.export(path)
    loaded_by_agama = agama.Potential(str(path))
    loaded_native = Potential(path)
    np.testing.assert_allclose(loaded_native.potential(points), native.potential(points), rtol=2e-10, atol=1e-11)
    np.testing.assert_allclose(loaded_by_agama.potential(points), native.potential(points), rtol=2e-10, atol=1e-11)

    agama_path = tmp_path / "agama-written.ini"
    agama_potential = agama.Potential(**args)
    agama_potential.export(str(agama_path))
    loaded_agama_export = Potential(agama_path)
    np.testing.assert_allclose(loaded_agama_export.potential(points), agama_potential.potential(points), rtol=2e-10, atol=1e-11)

    combined = Potential(native, loaded_native)
    np.testing.assert_allclose(combined.potential(points), 2 * native.potential(points), rtol=2e-10, atol=1e-11)
    combined_path = tmp_path / "composite.ini"
    combined.export(combined_path)
    roundtrip = agama.Potential(str(combined_path))
    np.testing.assert_allclose(roundtrip.potential(points), combined.potential(points), rtol=2e-10, atol=1e-11)


def test_set_units_matches_agama(tmp_path):
    agama.setUnits(length=1, mass=1, velocity=1)
    native_potential.setUnits(length=1, mass=1, velocity=1)
    assert native_potential.G == pytest.approx(agama.G, rel=2e-15)
    assert native_potential.getUnits() == pytest.approx(agama.getUnits())

    positions = np.array([[0., 0., 0.], [4., 0., 0.]])
    masses = np.array([1., 2.])
    points = np.array([[0., 0., 0.], [1., 0., 0.], [10., 0., 0.]])
    args = dict(type="Multipole", particles=(positions, masses), softening=1.,
                symmetry="s", lmax=0, rmin=0, rmax=100., gridSizeR=80)
    expected = agama.Potential(**args)
    actual = Potential(**args)
    np.testing.assert_allclose(actual.potential(points), expected.potential(points), rtol=2e-11)
    np.testing.assert_allclose(actual.force(points), expected.force(points), rtol=2e-10, atol=1e-14)

    path = tmp_path / "unit-scaled.ini"
    actual.export(path)
    loaded = Potential(path)
    loaded_by_agama = agama.Potential(str(path))
    np.testing.assert_allclose(loaded.potential(points), expected.potential(points), rtol=2e-10)
    np.testing.assert_allclose(loaded_by_agama.potential(points), expected.potential(points), rtol=2e-10)

    agama.setUnits()
    native_potential.setUnits()
