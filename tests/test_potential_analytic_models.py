"""Permanent analytic-model regressions; Agama references run out of process."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import quad

from kinematic_decompose import potential as backend

SCRIPT = Path(__file__).with_name("example_potential_analytic.py")
spec = importlib.util.spec_from_file_location("analytic_benchmark", SCRIPT)
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

PLUMMER_SCRIPT = Path(__file__).with_name("example_potential_grid_convergence.py")
plummer_spec = importlib.util.spec_from_file_location("plummer_accuracy_reference", PLUMMER_SCRIPT)
assert plummer_spec is not None and plummer_spec.loader is not None
plummer_reference = importlib.util.module_from_spec(plummer_spec)
plummer_spec.loader.exec_module(plummer_reference)


@pytest.mark.parametrize("model", ["MiyamotoNagai", "Ferrers"])
def test_positive_quadrature_and_explicit_direction_probes(model):
    xyz, volume = benchmark.source_nodes(model, 12)
    assert np.isfinite(xyz).all() and np.isfinite(volume).all()
    assert np.all(volume > 0) and np.all(np.linalg.norm(xyz, axis=1) > 0)
    radii, points = benchmark.probe_points(model)
    axes = np.eye(3)[[0, 2]] if model == "MiyamotoNagai" else np.eye(3)
    for i, axis in enumerate(axes):
        np.testing.assert_array_equal(points[i*64:(i+1)*64], radii[:, None] * axis)
    assert len(points) == len(axes) * 64 + 384
    if model == "Ferrers":
        # Full volume of the compact ellipsoid, independently of its density.
        np.testing.assert_allclose(volume.sum(), 4*np.pi/3 * 3 * 1.95 * 1.05, rtol=1e-13)
        assert np.all(np.sum((xyz / [3., 1.95, 1.05])**2, axis=1) < 1)


def test_plot_has_one_axes_and_only_analytic_and_softened_lines(tmp_path):
    results = {}
    for model, n_axes in [("MiyamotoNagai", 2), ("Ferrers", 3)]:
        radii, _ = benchmark.probe_points(model)
        profiles = np.concatenate([1 / (1 + radii / (i + 1)) for i in range(n_axes)])
        results[model] = {"rows": [{"radii_kpc": radii, "analytic": profiles,
                                   "n_nodes": 100, "configurations": {
                                       "native_softened": {"normalized_phi": profiles * .99}}}]}
    fig = benchmark.plot_results(results, tmp_path / "profiles.png")
    assert len(fig.axes) == 1
    assert fig.get_figwidth() == pytest.approx(fig.get_figheight())
    ax = fig.axes[0]
    assert ax.get_box_aspect() == pytest.approx(1)
    assert len(ax.lines) == 10
    assert sum(line.get_label().endswith(": analytic") for line in ax.lines) == 5
    assert sum(line.get_label().endswith(": softened") for line in ax.lines) == 5
    assert not any("agama" in line.get_label().lower() or "unsoftened" in line.get_label()
                   for line in ax.lines)
    assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
    assert not ax.texts  # No particle/source-count annotations.
    assert ax.get_xlim() == pytest.approx((.1, 10))
    assert ax.get_xlabel() == r"$r\,(\mathrm{kpc})$"
    legends = [ax.get_legend(), *ax.artists]
    assert all(legend is not None and all(text.get_fontsize() == pytest.approx(12.5)
                                           for text in legend.get_texts())
               for legend in legends)
    legend_text = [text.get_text() for legend in legends for text in legend.get_texts()]
    assert "Multi-particle softened" in legend_text
    assert all("equatorial" not in text for text in legend_text)


@pytest.fixture(scope="module", params=["MiyamotoNagai", "Ferrers"])
def reference(request, tmp_path_factory):
    python = os.environ.get("AGAMA_PYTHON", sys.executable)
    env = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[variable] = "1"
    available = subprocess.run([python, "-c", "import agama; assert callable(agama.Potential)"],
                               env=env, capture_output=True)
    if available.returncode:
        pytest.skip("Set AGAMA_PYTHON to an interpreter with Agama analytic models")
    path = tmp_path_factory.mktemp("analytic-field") / "reference.npz"
    subprocess.run([python, str(SCRIPT), "--worker", "reference", "--model", request.param,
                    "--order", "24", "--output", str(path)], env=env, check=True)
    with np.load(path) as data:
        return request.param, {key: data[key].copy() for key in data.files}


def test_reference_normalization_and_physical_mass(reference):
    model, data = reference
    assert data["phi_min"] < 0 and np.isfinite(data["truth"]).all()
    assert np.all(data["truth"] >= data["phi_min"])
    assert np.all(data["masses"] > 0)
    if model == "MiyamotoNagai":
        q = data["probes"]
        exact = -float(data["gravity"]) * 1e10 / np.sqrt(
            q[:, 0]**2 + q[:, 1]**2 + (3 + np.sqrt(q[:, 2]**2 + .8**2))**2)
        np.testing.assert_allclose(data["truth"], exact, rtol=2e-14)
        np.testing.assert_allclose(data["phi_min"], -float(data["gravity"])*1e10/3.8, rtol=2e-14)
        assert .96e10 < data["masses"].sum() < .98e10  # Unrenormalized finite domain.
    else:
        np.testing.assert_allclose(data["masses"].sum(), 1e10, rtol=1e-13)
        assert not np.allclose(data["truth"][:64], data["truth"][64:128])
        assert not np.allclose(data["truth"][64:128], data["truth"][128:192])


@pytest.mark.parametrize("softening", [None, .02])
@pytest.mark.parametrize("grid,lmax,budget", [(30, 4, .02), (80, 8, .005)])
def test_native_field_against_independent_agama_reference(reference, softening, grid, lmax, budget):
    model, data = reference
    backend.setUnits(length=1, mass=1, velocity=1)
    kwargs = dict(type="Multipole", particles=(data["positions"], data["masses"]),
                  symmetry="a" if model == "MiyamotoNagai" else "t", lmax=lmax,
                  mmax=lmax, gridSizeR=grid,
                  rmin=.001, rmax=np.linalg.norm(data["positions"], axis=1).max()*1.01)
    if softening is not None:
        kwargs["softening"] = softening
    try:
        potential = backend.Potential(**kwargs)
        phi = potential.potential(data["probes"])
        assert np.isfinite(phi).all()
        # Explicit resolution-dependent regression guards, not claims that the
        # requested coarse model meets the old high-resolution 0.5% budget.
        assert benchmark.max_error(phi, data["truth"], float(data["phi_min"])) < budget
        force = potential.force(data["probes"])
        assert np.isfinite(force).all()
    finally:
        backend.setUnits()


# Independent truncated-Plummer accuracy references.

def test_analytic_central_potential_matches_shell_integral():
    scale = plummer_reference.PLUMMER_SCALE_KPC
    truncation = plummer_reference.PLUMMER_TRUNCATION_KPC
    fraction = truncation**3 / (truncation**2 + scale**2)**1.5
    rho0 = 3 * plummer_reference.PLUMMER_TOTAL_MASS_MSUN / (4 * np.pi * scale**3 * fraction)
    integral = quad(lambda r: rho0 * r * (1 + (r/scale)**2)**-2.5,
                    0, truncation, epsabs=0, epsrel=1e-12)[0]
    expected = -4 * np.pi * 4.30091727067736e-6 * integral
    actual = plummer_reference.analytic_plummer(np.zeros(1))[0][0]
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_analytic_boundary_matches_kepler_exterior():
    radii = np.array([plummer_reference.PLUMMER_TRUNCATION_KPC, 20.])
    phi, force = plummer_reference.analytic_plummer(radii)
    gm = 4.30091727067736e-6 * plummer_reference.PLUMMER_TOTAL_MASS_MSUN
    np.testing.assert_allclose(phi, -gm/radii, rtol=1e-14)
    np.testing.assert_allclose(force, -gm/radii**2, rtol=1e-14)
