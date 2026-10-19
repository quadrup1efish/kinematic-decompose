#!/usr/bin/env python3
"""Unified analytic/particle potential profiles and raw relative residuals.

Run with AGAMA_PYTHON pointing to an Agama-enabled interpreter. Reference and
project-native libraries always run in separate processes. No production changes.
"""
import argparse
import importlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

# These must precede NumPy/SciPy and either native library.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.special import ellipk, roots_legendre

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "MiyamotoNagai": dict(type="MiyamotoNagai", mass=1e10, scaleRadius=3., scaleHeight=.8),
    "Ferrers": dict(type="Ferrers", mass=1e10, scaleRadius=3., axisRatioY=.65, axisRatioZ=.35),
}


def profile_comparison(phi, reference, reference_minimum):
    """Common reference normalization and raw signed relative residual (%)."""
    phi, reference = np.asarray(phi), np.asarray(reference)
    if (phi.shape != reference.shape or not np.isfinite(phi).all()
            or not np.isfinite(reference).all() or np.any(reference >= 0)
            or not np.isfinite(reference_minimum) or reference_minimum >= 0):
        raise ValueError("Finite matched potentials and a negative reference are required")
    return phi / reference_minimum, 100 * (phi - reference) / np.abs(reference)


def spline_potential_kernel(distance, support):
    """Independent cubic-spline potential kernel; support h, not epsilon."""
    distance = np.asarray(distance, dtype=float)
    if support <= 0 or not np.isfinite(support) or np.any(distance < 0):
        raise ValueError("Require positive support and nonnegative distances")
    u = distance / support
    kernel = np.empty_like(u)
    inner, middle = u < .5, (u >= .5) & (u < 1)
    v = u[inner]
    kernel[inner] = (14/5 - 16/3*v**2 + 48/5*v**4 - 32/5*v**5) / support
    v = u[middle]
    kernel[middle] = (16/5 - 32/3*v**2 + 16*v**3 - 48/5*v**4
                      + 32/15*v**5 - 1/(15*v)) / support
    kernel[u >= 1] = 1 / distance[u >= 1]
    return kernel


def particle_benchmark(input_path, metadata_path, grid=30, lmax=4):
    """Rebuild all components on identical saved physical source arrays."""
    from kinematic_decompose import potential as backend
    backend.setUnits(length=1, mass=1, velocity=1)
    backend.set_num_threads(1)
    metadata = json.loads(Path(metadata_path).read_text())
    digest = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    if digest != metadata["shared_input_sha256"]:
        raise ValueError("Particle input does not match its provenance manifest")
    support = float(metadata["softening_kernel_support_kpc"])
    radii = np.geomspace(.01, 10., 321)
    points = np.column_stack((np.r_[0., radii], np.zeros((len(radii) + 1, 2))))
    reference = np.zeros(len(points))
    profiles = {"softened": np.zeros(len(points)), "unsoftened": np.zeros(len(points))}
    counts = {}
    with np.load(input_path) as source:
        for component in metadata["components"]:
            name = component["name"]
            positions, masses = source[name + "_positions"], source[name + "_masses"]
            for key, array in [("positions", positions), ("masses", masses)]:
                if hashlib.sha256(array.tobytes()).hexdigest() != component[key + "_sha256"]:
                    raise ValueError("Component source hash mismatch: " + name)
            counts[name] = len(masses)
            for index, point in enumerate(points):
                distance = np.linalg.norm(positions - point, axis=1)
                reference[index] -= backend.G * np.dot(masses, spline_potential_kernel(distance, support))
            for method in profiles:
                arguments: dict = dict(type="Multipole", particles=(positions, masses),
                                 symmetry="n", lmax=lmax, mmax=lmax, gridSizeR=grid,
                                 rmin=component["rmin"], rmax=component["rmax"])
                if method == "softened":
                    arguments["softening"] = support
                potential = backend.Potential(**arguments)
                profiles[method] += potential.potential(points)
                del potential
            print("Particle component", name, "N=", len(masses), flush=True)
    if reference[0] > reference[1:].min():
        raise ValueError("Origin is not the reference profile minimum")
    result = dict(radii_kpc=radii.tolist(), reference_phi=reference[1:].tolist(),
                  reference_minimum=float(reference[0]),
                  profiles={key: value[1:].tolist() for key, value in profiles.items()},
                  grid=grid, lmax=lmax, mmax=lmax, symmetry="n", support_kpc=support,
                  counts=counts, input_sha256=digest, source_metadata=metadata,
                  input_path=str(Path(input_path).resolve()), metadata_path=str(Path(metadata_path).resolve()))
    for values in result["profiles"].values():
        profile_comparison(values, result["reference_phi"], result["reference_minimum"])
    return result


def source_nodes(model, order, domain=100.):
    """Return coordinates and volume weights, not renormalized particle masses."""
    if model == "MiyamotoNagai":
        u, w = roots_legendre(2 * order)
        u, w = (u + 1) / 2, w / 2
        scale = 3.8
        angle = np.arctan(domain / scale)
        radius = scale * np.tan(angle * u)
        dr = scale * angle * w / np.cos(angle * u)**2
        v, wz = roots_legendre(4 * order)
        angle_z = np.arctan(domain / .8)
        z = .8 * np.tan(angle_z * v)
        dz = .8 * angle_z * wz / np.cos(angle_z * v)**2
        rr, zz = np.meshgrid(radius, z, indexing="ij")
        # Axisymmetric mmax=0 treats each weighted location as a ring.
        xyz = np.column_stack((rr.ravel(), np.zeros(rr.size), zz.ravel()))
        volume = (2 * np.pi * rr * dr[:, None] * dz[None, :]).ravel()
    elif model == "Ferrers":
        u, w = roots_legendre(order)
        radius, dr = (u + 1) / 2, w / 2
        mu, wm = roots_legendre(order)
        azimuth = np.arange(2 * order) * np.pi / order
        rr, mm, pp = np.meshgrid(radius, mu, azimuth, indexing="ij")
        sine = np.sqrt(1 - mm**2)
        axes = np.array([3., 3. * .65, 3. * .35])
        xyz = np.stack((axes[0] * rr * sine * np.cos(pp),
                        axes[1] * rr * sine * np.sin(pp), axes[2] * rr * mm), axis=-1).reshape(-1, 3)
        volume = (np.prod(axes) * rr**2 * dr[:, None, None]
                  * wm[None, :, None] * (np.pi / order)).ravel()
    else:
        raise ValueError(model)
    return xyz, volume


def probe_points(model):
    radii = np.geomspace(.05, 10., 64)
    axes = np.eye(3)[[0, 2]] if model == "MiyamotoNagai" else np.eye(3)
    profiles = np.concatenate([radii[:, None] * axis for axis in axes])
    # Independent off-axis directions: axes alone do not validate 3D errors.
    index = np.arange(96)
    mu = 1 - 2 * (index + .5) / len(index)
    phi = index * np.pi * (3 - np.sqrt(5))
    directions = np.column_stack((np.sqrt(1 - mu**2) * np.cos(phi),
                                  np.sqrt(1 - mu**2) * np.sin(phi), mu))
    angular = np.concatenate([directions * radius for radius in (.2, 1., 3., 10.)])
    return radii, np.concatenate((profiles, angular))


def reference_worker(args):
    agama = importlib.import_module("agama")
    agama.setUnits(length=1, mass=1, velocity=1)
    reference = agama.Potential(**MODELS[args.model])
    positions, volume = source_nodes(args.model, args.order, args.domain)
    masses = volume * reference.density(positions)
    radii, probes = probe_points(args.model)
    phi_min = float(reference.potential([0., 0., 0.]))
    assert phi_min < 0 and np.isfinite(masses).all() and (masses > 0).all()
    np.savez(args.output, positions=positions, masses=masses, radii=radii, probes=probes,
             truth=reference.potential(probes), phi_min=phi_min, gravity=agama.G)


def field_worker(args):
    if args.worker == "native":
        from kinematic_decompose import potential as backend
    else:
        backend = importlib.import_module("agama")
    backend.setUnits(length=1, mass=1, velocity=1)
    data = np.load(args.input)
    positions, masses, probes = data["positions"], data["masses"], data["probes"]
    symmetry = "a" if args.model == "MiyamotoNagai" else "t"
    kwargs = dict(type="Multipole", particles=(positions, masses), symmetry=symmetry,
                  lmax=args.lmax, mmax=args.lmax,
                  gridSizeR=args.grid, rmin=.001,
                  rmax=float(np.linalg.norm(positions, axis=1).max() * 1.01))
    if args.worker == "native" and args.softening > 0:
        kwargs["softening"] = args.softening
    start = perf_counter()
    potential = backend.Potential(**kwargs)
    elapsed = perf_counter() - start
    np.savez(args.output, phi=potential.potential(probes), build_time_s=elapsed)


def run_worker(python, mode, output, model, **kwargs):
    command = [python, str(Path(__file__).resolve()), "--worker", mode,
               "--output", str(output), "--model", model]
    for key, value in kwargs.items():
        command.extend(["--" + key, str(value)])
    subprocess.run(command, check=True)


def max_error(phi, truth, phi_min):
    return float(np.max(np.abs(np.asarray(phi) - truth)) / abs(phi_min))


def benchmark(orders, scratch):
    agama_python = os.environ.get("AGAMA_PYTHON", sys.executable)
    results = {}
    for model in MODELS:
        rows = []
        source = scratch / "source.npz"
        for order in orders:
            run_worker(agama_python, "reference", source, model, order=order)
            with np.load(source) as data:
                truth, phi_min = data["truth"], float(data["phi_min"])
                row: dict = dict(order=order, n_nodes=len(data["masses"]),
                           mass_msun=float(data["masses"].sum()),
                           radii_kpc=data["radii"].tolist(), phi_min_km2_s2=phi_min,
                           analytic=(truth / phi_min).tolist(), configurations={})
                count = (2 if model == "MiyamotoNagai" else 3) * len(data["radii"])
                select = np.arange(0, count, 8)
                direct = []
                for q in data["probes"][select]:
                    if model == "MiyamotoNagai":
                        # Exact azimuthal ring integral; no point-at-phi=0 shortcut.
                        radius = np.hypot(q[0], q[1])
                        distance2 = ((data["positions"][:, 0] + radius)**2
                                     + (data["positions"][:, 2] - q[2])**2)
                        parameter = 4 * data["positions"][:, 0] * radius / distance2
                        kernel = 2 / np.pi * ellipk(parameter) / np.sqrt(distance2)
                    else:
                        kernel = 1 / np.linalg.norm(data["positions"] - q, axis=1)
                    direct.append(-float(data["gravity"]) * np.sum(data["masses"] * kernel))
                row["direct_axis_max_error"] = max_error(direct, truth[select], phi_min)
                row["direct_axis_probes"] = dict(indices=select.tolist(),
                                                normalized_phi=(np.asarray(direct) / phi_min).tolist())
            for name, mode, grid, lmax, softening in [
                ("native", "native", 30, 6, 0),
                ("agama", "agama", 30, 6, 0),
                ("native_softened", "native", 30, 6, .02),
                ("native_grid160", "native", 160, 6, 0),
                ("native_lmax12", "native", 30, 12, 0),
            ]:
                output = scratch / "field.npz"
                run_worker(sys.executable if mode == "native" else agama_python,
                           mode, output, model, input=source, grid=grid,
                           lmax=lmax, softening=softening)
                with np.load(output) as data:
                    phi = data["phi"]
                    n_axes = 2 if model == "MiyamotoNagai" else 3
                    row["configurations"][name] = dict(
                        normalized_phi=(phi / phi_min).tolist(),
                        max_error=max_error(phi, truth, phi_min),
                        axis_max_errors=[max_error(phi[i*64:(i+1)*64], truth[i*64:(i+1)*64], phi_min)
                                         for i in range(n_axes)],
                        off_axis_max_error=max_error(phi[n_axes*64:], truth[n_axes*64:], phi_min),
                        direct_axis_max_difference=max_error(phi[select], np.asarray(direct), phi_min),
                        build_time_s=float(data["build_time_s"]), grid=grid, lmax=lmax,
                        mmax=lmax, symmetry="a" if model == "MiyamotoNagai" else "t",
                        softening_kpc=softening)
            rows.append(row)
            print(model, order, "N=", row["n_nodes"], "max errors:",
                  {k: round(v["max_error"], 6) for k, v in row["configurations"].items()}, flush=True)
        # Independent domain sweep, with the same quadrature order.
        if model == "MiyamotoNagai":
            run_worker(agama_python, "reference", source, model, order=orders[-1], domain=300.)
            output = scratch / "field.npz"
            run_worker(sys.executable, "native", output, model, input=source, grid=30, lmax=6)
            with np.load(source) as data, np.load(output) as field:
                domain_check = dict(domain_kpc=300., mass_msun=float(data["masses"].sum()),
                                    max_error=max_error(field["phi"], data["truth"], float(data["phi_min"])),
                                    max_change_from_domain100=float(np.max(np.abs(
                                        field["phi"] / float(data["phi_min"])
                                        - rows[-1]["configurations"]["native"]["normalized_phi"]))))
        else:
            domain_check = dict(description="Full compact-support ellipsoid; no omitted density.")
        results[model] = dict(parameters=MODELS[model], rows=rows, domain_check=domain_check)
    return results


def plot_results(results, path, particle_results=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
             "mathtext.fontset": "stix", "font.size": 20, "axes.labelsize": 28,
             "xtick.labelsize": 22, "ytick.labelsize": 22, "legend.fontsize": 20,
             "axes.grid": False, "axes.linewidth": 1.4, "legend.frameon": False,
             "xtick.major.width": 1.4, "ytick.major.width": 1.4,
             "xtick.major.size": 7, "ytick.major.size": 7}
    columns = 2 if particle_results is not None else 1
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, columns, figsize=(7 * columns, 7), squeeze=False,
                                 gridspec_kw={"height_ratios": [4, 1], "hspace": 0},
                                 sharex="col", sharey="row")
        fig.subplots_adjust(left=.085 if columns == 2 else .15, right=.985,
                            bottom=.15, top=.98, wspace=.13, hspace=0)
        _draw_profiles(fig, axes, results, particle_results)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
    return fig


def _draw_profiles(fig, axes, results, particle_results):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    ax, error_ax = axes[:, 0]
    directions = [
        ("MiyamotoNagai", ["MN: x/y", "MN: z"], ["#2166ac", "#4393c3"], ["o", "s"]),
        ("Ferrers", ["Ferrers: x", "Ferrers: y", "Ferrers: z"], ["#b2182b", "#d6604d", "#762a83"], ["^", "D", "v"]),
    ]
    direction_handles = []
    displayed = []
    for model, labels, colors, markers in directions:
        row = results[model]["rows"][-1]
        r = row["radii_kpc"]
        for i, (label, color, marker) in enumerate(zip(labels, colors, markers)):
            sl = slice(i * len(r), (i + 1) * len(r))
            minimum = row["phi_min_km2_s2"]
            reference = np.array(row["analytic"])[sl] * minimum
            phi = np.array(row["configurations"]["native_softened"]["normalized_phi"])[sl] * minimum
            _, residual = profile_comparison(phi, reference, minimum)
            analytic = reference / np.min(reference)
            softened = phi / np.min(phi)
            ax.plot(r, analytic, color=color, lw=4, label=label + ": analytic")
            ax.plot(r, softened, color=color, ls="none", zorder=4,
                    marker=marker, markevery=(i + 2, 7), markersize=10,
                    markerfacecolor="white", markeredgewidth=2.2, label=label + ": softened")
            displayed.extend([analytic, softened])
            error_ax.plot(r, residual, color=color, lw=3, label=label,
                          marker=marker, markevery=(i + 2, 7), markersize=6,
                          markerfacecolor="white", markeredgewidth=1.6)
            direction_handles.append(Line2D([], [], color=color, lw=4,
                                            marker=marker, markersize=10,
                                            markerfacecolor="white", markeredgewidth=2.2, label=label))
    values = np.concatenate(displayed)
    ax.set(xscale="log", xlim=(.1, 10),
           ylim=(min(0., float(values.min()) - .04), max(1.04, float(values.max()) + .04)),
           ylabel=r"$\Phi/\Phi_{\min}$")
    legend = ax.legend(handles=direction_handles, loc="lower left", bbox_to_anchor=(.02, .05),
                       frameon=False)
    ax.add_artist(legend)
    ax.legend(handles=[Line2D([], [], color="0.25", lw=4, label="Analytic"),
                       Line2D([], [], color="0.25", ls="none", marker="o", markersize=10,
                              markerfacecolor="white", markeredgewidth=2.2,
                              label="Softened")],
              loc="upper right", frameon=False)
    analytic_limits = ax.get_ylim()
    if particle_results is not None:
        top, bottom = axes[:, 1]
        row = particle_results
        r, reference, minimum = row["radii_kpc"], row["reference_phi"], row["reference_minimum"]
        reference = np.asarray(reference)
        normalized = reference / np.min(reference)
        top.plot(r, normalized, color="#b2182b", lw=4, label="Direct")
        displayed = [normalized]
        for method, color, marker, linestyle in [("softened", "#2166ac", "o", "-"),
                                                  ("unsoftened", "#222222", "s", "--")]:
            phi = np.asarray(row["profiles"][method])
            _, residual = profile_comparison(phi, reference, minimum)
            normalized = phi / np.min(phi)
            top.plot(r, normalized, color=color, lw=4, label=method.capitalize(),
                     ls="none" if method == "softened" else linestyle,
                     marker=marker if method == "softened" else None,
                     markevery=32, markersize=10, markerfacecolor="white", markeredgewidth=2.2)
            bottom.plot(r, residual, color=color, lw=3, ls=linestyle,
                        marker=marker, markevery=32, markersize=6, markerfacecolor="white",
                        markeredgewidth=1.6)
            displayed.append(normalized)
        values = np.concatenate(displayed)
        top.set(ylim=(min(0., float(values.min()) - .04), max(1.04, float(values.max()) + .04)),
                ylabel="")
        top.legend(loc="lower left")
        # Every displayed curve uses its own sampled minimum and is <= 1.
        limits = (min(analytic_limits[0], top.get_ylim()[0]),
                  max(analytic_limits[1], top.get_ylim()[1]))
        for profile_ax in axes[0]:
            profile_ax.set_ylim(limits)
    for column in range(axes.shape[1]):
        top, bottom = axes[:, column]
        top.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="lower"))
        bottom.yaxis.set_major_locator(MaxNLocator(nbins=3))
        top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        bottom.set(xscale="log", xlabel=r"$r\,(\mathrm{kpc})$", ylabel="Error [%]" if column == 0 else "")
        if column > 0:
            for row_ax in (top, bottom):
                row_ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        bottom.axhline(0., color=".5", lw=1.2, zorder=0)
        bottom.margins(y=.15)
        top.set_xlim(.1, 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["reference", "native", "agama"])
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--order", type=int, default=48)
    parser.add_argument("--domain", type=float, default=100.)
    parser.add_argument("--grid", type=int, default=30)
    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument("--softening", type=float, default=0.)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--particle-input", type=Path, help="Centered physical source NPZ, outside tests")
    parser.add_argument("--particle-metadata", type=Path, help="Matching source provenance JSON")
    parser.add_argument("--results", type=Path, help="Persistent calculation JSON; enables cached redraw")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.worker == "reference":
        reference_worker(args)
    elif args.worker:
        field_worker(args)
    else:
        if args.plot_only:
            if args.results is None:
                parser.error("--plot-only requires --results")
            saved = json.loads(args.results.read_text())
        else:
            if (args.particle_input is None) != (args.particle_metadata is None):
                parser.error("--particle-input and --particle-metadata must be supplied together")
            with tempfile.TemporaryDirectory(prefix="analytic-field-", dir=os.environ.get("TMPDIR")) as scratch:
                results = benchmark([12, 16] if args.smoke else [24, 36, 48], Path(scratch))
            particles = (particle_benchmark(args.particle_input, args.particle_metadata)
                         if args.particle_input is not None else None)
            saved = dict(analytic=results, particles=particles,
                         residual="100*(Phi-Phi_ref)/abs(Phi_ref)",
                         normalization="Upper curves use individual sampled minima; residuals use raw potentials")
            if args.results is not None:
                args.results.parent.mkdir(parents=True, exist_ok=True)
                args.results.write_text(json.dumps(saved, indent=2))
        stem = "potential_analytic_models_smoke" if args.smoke else "potential_analytic_models"
        figure = ROOT / "images" / (stem + ".png")
        plot_results(saved["analytic"], figure, saved["particles"])
        print("Saved", figure)


if __name__ == "__main__":
    main()
