"""Permanent analytic-model regressions; Agama references run out of process."""
import importlib.util
import hashlib
import json
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


@pytest.mark.parametrize("include_particles", [False, True])
def test_profile_error_layout_and_individual_normalization(tmp_path, include_particles):
    results = {}
    for model, n_axes in [("MiyamotoNagai", 2), ("Ferrers", 3)]:
        radii, _ = benchmark.probe_points(model)
        profiles = np.concatenate([1 / (1 + radii / (i + 1)) for i in range(n_axes)])
        results[model] = {"rows": [{"radii_kpc": radii, "analytic": profiles, "phi_min_km2_s2": -100.,
                                   "n_nodes": 100, "configurations": {
                                       "native_softened": {"normalized_phi": profiles * .99}}}]}
    particles = None
    reference = -100 / (1 + radii)
    if include_particles:
        particles = dict(radii_kpc=radii, reference_phi=reference, reference_minimum=-100.,
                         profiles={"softened": reference + 1, "unsoftened": reference + 5})
    fig = benchmark.plot_results(results, tmp_path / "profiles.png", particles)
    columns = 2 if include_particles else 1
    assert len(fig.axes) == 2 * columns
    assert all(panel.get_xlim() == pytest.approx((.1, 10)) for panel in fig.axes)
    assert fig.get_figwidth() == pytest.approx(columns * fig.get_figheight())
    ax = fig.axes[0]
    assert len(ax.lines) == 10
    for line in ax.lines:
        assert np.max(line.get_ydata()) == pytest.approx(1.)
        assert np.all(np.asarray(line.get_ydata()) <= 1.)
    assert sum(line.get_label().endswith(": analytic") for line in ax.lines) == 5
    assert sum(line.get_label().endswith(": softened") for line in ax.lines) == 5
    assert not any("agama" in line.get_label().lower() or "unsoftened" in line.get_label()
                   for line in ax.lines)
    assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
    assert not ax.texts  # No particle/source-count annotations.
    assert ax.get_xlim() == pytest.approx((.1, 10))
    assert ax.get_xlabel() == ""
    residual_ax = fig.axes[columns]
    assert residual_ax.get_xlabel() == r"$r\,(\mathrm{kpc})$"
    assert residual_ax.get_ylabel() == "Error [%]"
    assert ax.get_ylabel() == r"$\Phi/\Phi_{\min}$"
    assert ax.yaxis.label.get_fontsize() == 28
    assert ax.lines[0].get_linewidth() == 4
    assert residual_ax.lines[0].get_linewidth() == 3
    for line in residual_ax.lines[:-1]:
        np.testing.assert_allclose(line.get_ydata(), 1., atol=2e-14)
    assert ax.get_position().width == pytest.approx(residual_ax.get_position().width)
    assert ax.get_position().y0 == pytest.approx(residual_ax.get_position().y1)
    legends = [ax.get_legend(), *ax.artists]
    assert all(legend is not None and all(text.get_fontsize() == pytest.approx(20)
                                           for text in legend.get_texts())
               for legend in legends)
    legend_text = [text.get_text() for legend in legends for text in legend.get_texts()]
    assert "Softened" in legend_text
    assert all("equatorial" not in text for text in legend_text)
    assert (tmp_path / "profiles.pdf").is_file()
    if include_particles:
        top, bottom = fig.axes[1], fig.axes[3]
        assert [text.get_text() for text in top.get_legend().get_texts()] == [
            "Direct", "Softened", "Unsoftened"]
        np.testing.assert_allclose(top.lines[1].get_ydata(), (reference + 1) / np.min(reference + 1))
        for line in top.lines:
            assert np.max(line.get_ydata()) == pytest.approx(1.)
            assert np.all(np.asarray(line.get_ydata()) <= 1.)
        np.testing.assert_allclose(bottom.lines[0].get_ydata(), 100 / np.abs(reference))
        assert top.get_ylabel() == bottom.get_ylabel() == ""
        assert ax.get_shared_y_axes().joined(ax, top)
        assert residual_ax.get_shared_y_axes().joined(residual_ax, bottom)
        assert top.get_ylim() == pytest.approx(ax.get_ylim())
        assert bottom.get_ylim() == pytest.approx(residual_ax.get_ylim())
        assert not top.get_yticklabels() and not bottom.get_yticklabels()
        assert bottom.get_position().width == pytest.approx(residual_ax.get_position().width)


def test_reference_normalization_preserves_absolute_offset():
    reference = np.array([-100., -50.])
    normalized, residual = benchmark.profile_comparison(reference + 10, reference, -100.)
    np.testing.assert_allclose(normalized, [.9, .4])
    np.testing.assert_allclose(residual, [10, 20])
    with pytest.raises(ValueError):
        benchmark.profile_comparison(reference, np.zeros(2), -100.)


def test_independent_spline_kernel_limits_and_shell_integral():
    h = .4
    distances = np.array([0., .5*h, h, 2*h])
    np.testing.assert_allclose(benchmark.spline_potential_kernel(distances, h),
                               np.array([2.8, 28/15, 1., .5]) / h, rtol=1e-14)
    # Independent spherical-shell integration of the normalized spline density;
    # a finite-order off-center Multipole is not an exact single-particle oracle.
    def density(radius):
        u = radius / h
        shape = 1 - 6*u*u + 6*u**3 if u < .5 else 2*(1-u)**3
        return 8 / (np.pi*h**3) * shape

    for radius in np.r_[distances, [.1*h, .8*h]]:
        enclosed = quad(lambda s: 4*np.pi*s*s*density(s), 0, min(radius, h),
                        points=[.5*h] if radius > .5*h else None, epsabs=1e-13)[0]
        exterior = (quad(lambda s: 4*np.pi*s*density(s), radius, h,
                         points=[.5*h] if radius < .5*h else None, epsabs=1e-13)[0]
                    if radius < h else 0.)
        expected = (enclosed/radius if radius > 0 else 0.) + exterior
        np.testing.assert_allclose(benchmark.spline_potential_kernel(radius, h), expected, rtol=1e-13)


def test_particle_benchmark_uses_verified_sources_and_shared_resolution(tmp_path):
    # Avoid an exact-origin massive source: Newtonian construction excludes it.
    positions = np.array([[.001, 0., 0.], [-.001, 0., 0.], [1., 0., 0.], [-1., 0., 0.]])
    masses = np.array([5e7, 5e7, 1e5, 1e5])
    source = tmp_path / "source.npz"
    np.savez(source, stars_positions=positions, stars_masses=masses)
    metadata = dict(shared_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    softening_kernel_support_kpc=.8,
                    components=[dict(name="stars", rmin=.02, rmax=20.,
                                     positions_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),
                                     masses_sha256=hashlib.sha256(masses.tobytes()).hexdigest())])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(metadata))
    try:
        result = benchmark.particle_benchmark(source, manifest)
        assert (result["grid"], result["lmax"], result["mmax"], result["symmetry"]) == (30, 4, 4, "n")
        assert result["counts"] == {"stars": 4}
        radius = np.array(result["radii_kpc"])
        assert radius[[0, -1]] == pytest.approx([.01, 10.])
        assert np.all(np.diff(radius) > 0)
        expected = -backend.G * sum(
            mass * benchmark.spline_potential_kernel(abs(radius - position[0]), .8)
            for mass, position in zip(masses, positions))
        np.testing.assert_allclose(result["reference_phi"], expected, rtol=1e-14)
        for values in result["profiles"].values():
            assert len(values) == 321 and np.isfinite(values).all()
        metadata["shared_input_sha256"] = "mismatch"
        manifest.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="provenance"):
            benchmark.particle_benchmark(source, manifest)
    finally:
        backend.setUnits()


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
@pytest.mark.parametrize("grid,lmax,budget", [(30, 4, .02), (30, 6, .01), (80, 8, .005)])
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
