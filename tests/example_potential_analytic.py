#!/usr/bin/env python3
"""Deterministic field validation against Agama MiyamotoNagai and Ferrers.

Run with AGAMA_PYTHON pointing to an Agama-enabled interpreter. Reference and
project-native libraries always run in separate processes. No production changes.
"""
import argparse
import importlib
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
                ("native", "native", 30, 4, 0),
                ("agama", "agama", 30, 4, 0),
                ("native_softened", "native", 30, 4, .02),
                ("native_grid160", "native", 160, 4, 0),
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
            run_worker(sys.executable, "native", output, model, input=source, grid=30, lmax=4)
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


def plot_results(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 16,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12.5,
                         "axes.grid": False})
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
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
            sl = slice(i * 64, (i + 1) * 64)
            analytic = np.array(row["analytic"])[sl]
            softened = np.array(row["configurations"]["native_softened"]["normalized_phi"])[sl]
            ax.plot(r, analytic, color=color, lw=2.8, label=label + ": analytic")
            ax.plot(r, softened, color=color, ls="none", zorder=4,
                    marker=marker, markevery=(i + 2, 7), markersize=7.2,
                    markerfacecolor="white", markeredgewidth=1.5, label=label + ": softened")
            displayed.extend([analytic, softened])
            direction_handles.append(Line2D([], [], color=color, lw=2.8,
                                            marker=marker, markersize=7.2,
                                            markerfacecolor="white", markeredgewidth=1.5, label=label))
    values = np.concatenate(displayed)
    ax.set(xscale="log", xlim=(.1, 10),
           ylim=(min(0., float(values.min()) - .04), max(1.04, float(values.max()) + .04)),
           xlabel=r"$r\,(\mathrm{kpc})$", ylabel=r"$\Phi/\Phi_{\min}$")
    ax.set_box_aspect(1)
    legend = ax.legend(handles=direction_handles, loc="lower left", bbox_to_anchor=(.02, .16),
                       frameon=False, fontsize=12.5)
    ax.add_artist(legend)
    ax.legend(handles=[Line2D([], [], color="0.25", lw=2.8, label="Analytic"),
                       Line2D([], [], color="0.25", ls="none", marker="o", markersize=7.2,
                              markerfacecolor="white", markeredgewidth=1.5,
                              label="Multi-particle softened")],
              loc="upper right", frameon=False, fontsize=12.5)

    fig.savefig(path, dpi=300)
    plt.close(fig)
    return fig


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
    args = parser.parse_args()
    if args.worker == "reference":
        reference_worker(args)
    elif args.worker:
        field_worker(args)
    else:
        with tempfile.TemporaryDirectory(prefix="analytic-field-", dir=os.environ.get("TMPDIR")) as scratch:
            results = benchmark([12, 16] if args.smoke else [24, 36, 48], Path(scratch))
        stem = "potential_analytic_models_smoke" if args.smoke else "potential_analytic_models"
        figure = ROOT / "images" / (stem + ".png")
        plot_results(results, figure)
        print("Saved", figure)


if __name__ == "__main__":
    main()
