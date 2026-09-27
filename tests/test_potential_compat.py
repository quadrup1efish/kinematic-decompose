import json
import subprocess
import sys
import warnings
from importlib.metadata import PackageNotFoundError, distribution

import numpy as np
import pytest

try:
    distribution("agama")
except PackageNotFoundError:
    pytest.skip("install the agama-compat extra to run cross-backend tests", allow_module_level=True)

from kinematic_decompose import potential as native_potential
from kinematic_decompose.potential import Potential


_AGAMA_WORKER = r"""
import json, sys
import numpy as np
import agama

cfg = json.loads(sys.argv[1])
agama.setUnits(**cfg.get("units", {}))
if "ini" in cfg:
    potential = agama.Potential(cfg["ini"])
else:
    args = cfg["args"]
    positions, masses = args.pop("particles")
    args["particles"] = (np.asarray(positions, dtype=float), np.asarray(masses, dtype=float))
    potential = agama.Potential(**args)
if "export" in cfg:
    potential.export(cfg["export"])
points = np.asarray(cfg["points"], dtype=float)
print(json.dumps({
    "potential": np.asarray(potential.potential(points)).tolist(),
    "force": np.asarray(potential.force(points)).tolist(),
    "G": float(agama.G),
    "units": {key: float(value) for key, value in agama.getUnits().items()},
}))
"""


def _agama_values(points, *, args=None, ini=None, export=None, units=None):
    config = {"points": np.asarray(points).tolist(), "units": units or {}}
    if args is not None:
        config["args"] = args
    if ini is not None:
        config["ini"] = str(ini)
    if export is not None:
        config["export"] = str(export)
    result = subprocess.run(
        [sys.executable, "-c", _AGAMA_WORKER, json.dumps(config)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def _set_native_units(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        native_potential.setUnits(**kwargs)


@pytest.mark.parametrize("symmetry,lmax,mmax", [("s", 0, 0), ("a", 4, 0), ("n", 4, 4)])
def test_multipole_matches_agama(symmetry, lmax, mmax):
    _set_native_units()
    positions = np.array([
        [0.2, -0.1, 0.4], [1.2, 0.8, -0.5], [-0.7, 0.3, 1.1], [0.05, 0.2, 0.15],
    ])
    masses = np.array([1.0, 2.0, 0.7, 0.2])
    points = np.array([[0.1, 0.2, -0.3], [1.5, 0.1, 0.8], [4.0, -2.0, 1.0]])
    args = dict(type="Multipole", particles=(positions, masses),
                symmetry=symmetry, lmax=lmax, mmax=mmax,
                rmin=1e-4, rmax=20, gridSizeR=80)

    actual = Potential(**args)
    expected = _agama_values(points, args={**args, "particles": (positions.tolist(), masses.tolist())})
    # The standalone Agama wheel and this in-tree source can differ slightly in
    # spline fitting details; compare at a tolerance tighter than the grid error.
    np.testing.assert_allclose(actual.potential(points), expected["potential"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(actual.force(points), expected["force"], rtol=1e-4, atol=1e-6)


def test_agama_ini_roundtrip_and_composite(tmp_path):
    _set_native_units()
    positions = np.array([[0.0, 0.0, 0.0], [1.5, -0.2, 0.4]])
    masses = np.array([1.2, 2.3])
    points = np.array([[0.2, 0.1, 0.3], [2.0, 0.0, -0.5], [5.0, 1.0, 0.2]])
    args = dict(type="Multipole", particles=(positions, masses), softening=0.3,
                symmetry="s", lmax=0, mmax=0, rmin=0, rmax=20, gridSizeR=40)

    native = Potential(**args)
    path = tmp_path / "multipole.ini"
    native.export(path)
    loaded_native = Potential(path)
    np.testing.assert_allclose(loaded_native.potential(points), native.potential(points), rtol=2e-10, atol=1e-11)
    loaded_by_agama = _agama_values(points, ini=path)
    np.testing.assert_allclose(loaded_by_agama["potential"], native.potential(points), rtol=1e-9, atol=2e-9)
    np.testing.assert_allclose(loaded_by_agama["force"], native.force(points), rtol=2e-8, atol=2e-9)

    agama_path = tmp_path / "agama-written.ini"
    agama_positions = np.array([[0.2, 0.0, 0.0], [1.5, -0.2, 0.4]])
    agama_args = dict(type="Multipole", particles=(agama_positions.tolist(), masses.tolist()),
                      symmetry="s", lmax=0, mmax=0, rmin=1e-4, rmax=20, gridSizeR=40)
    expected_agama = _agama_values(points, args=agama_args, export=agama_path)
    loaded_agama_export = Potential(agama_path)
    np.testing.assert_allclose(loaded_agama_export.potential(points), expected_agama["potential"],
                               rtol=1e-8, atol=1e-10)

    combined = Potential(native, loaded_native)
    np.testing.assert_allclose(combined.potential(points), 2 * native.potential(points), rtol=2e-10, atol=1e-11)
    combined_path = tmp_path / "composite.ini"
    combined.export(combined_path)
    roundtrip = _agama_values(points, ini=combined_path)
    np.testing.assert_allclose(roundtrip["potential"], combined.potential(points), rtol=1e-9, atol=2e-9)


def test_set_units_matches_agama(tmp_path):
    unit_args = dict(length=1, mass=1, velocity=1)
    _set_native_units(**unit_args)
    positions = np.array([
        [0.2, -0.1, 0.4], [1.2, 0.8, -0.5], [-0.7, 0.3, 1.1], [0.05, 0.2, 0.15],
    ])
    masses = np.array([1.0, 2.0, 0.7, 0.2])
    points = np.array([[0.1, 0.2, -0.3], [1.5, 0.1, 0.8], [4.0, -2.0, 1.0]])
    args = dict(type="Multipole", particles=(positions, masses),
                symmetry="s", lmax=0, rmin=1e-4, rmax=20., gridSizeR=80)
    expected = _agama_values(points, args={**args, "particles": (positions.tolist(), masses.tolist())},
                             units=unit_args)
    actual = Potential(**args)
    assert native_potential.G == pytest.approx(expected["G"], rel=2e-15)
    assert native_potential.getUnits() == pytest.approx(expected["units"])
    np.testing.assert_allclose(actual.potential(points), expected["potential"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(actual.force(points), expected["force"], rtol=1e-4, atol=1e-6)

    path = tmp_path / "unit-scaled.ini"
    actual.export(path)
    loaded = Potential(path)
    loaded_by_agama = _agama_values(points, ini=path, units=unit_args)
    np.testing.assert_allclose(loaded.potential(points), actual.potential(points), rtol=2e-10)
    np.testing.assert_allclose(loaded_by_agama["potential"], expected["potential"], rtol=2e-8, atol=2e-9)
    _set_native_units()
