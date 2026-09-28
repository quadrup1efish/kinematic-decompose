#!/usr/bin/env python3
"""Grid convergence and analytic-error study on one fixed Plummer sample.

Stock Agama runs in a subprocess so its native library never shares a process
with the project-native extension.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEST_DIRECTORY = Path(__file__).resolve().parent
if str(TEST_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TEST_DIRECTORY))
import example_potential_performance as benchmark
from kinematic_decompose.potential import Potential, setUnits

PARTICLE_COUNT = 1_000_000
GRID_SIZES = (15, 30, 60, 120, 240)
SEED = 2026
RADIAL_ORDER = 64
ANGULAR_ORDER = 96


def kernel_shape(q):
    """Normalized compact cubic-spline kernel used by the native builder."""
    q = np.asarray(q, dtype=float)
    value = np.zeros_like(q)
    inner = q < 0.5
    outer = (q >= 0.5) & (q < 1.0)
    value[inner] = 8 / np.pi * (1 - 6 * q[inner]**2 + 6 * q[inner]**3)
    value[outer] = 16 / np.pi * (1 - q[outer])**3
    return value


def softened_analytic_plummer(radii, support, radial_order=RADIAL_ORDER,
                              angular_order=ANGULAR_ORDER):
    """Convolve the analytic Plummer potential and force with the same kernel.

    Convolution commutes with Poisson's equation. Integrating the analytic
    field against the normalized kernel therefore gives the continuum target
    for the native softened-source model, without using its production solver.
    Angular quadrature is split where the field crosses the truncation radius.
    """
    radii = np.asarray(radii, dtype=float)
    xq, wq = np.polynomial.legendre.leggauss(radial_order)
    q_nodes = []
    q_weights = []
    for left, right in ((0.0, 0.5), (0.5, 1.0)):
        q = left + (right - left) * (xq + 1) / 2
        weights = (right - left) * wq / 2
        q_nodes.append(q)
        q_weights.append(2 * np.pi * q**2 * kernel_shape(q) * weights)
    q_nodes = np.concatenate(q_nodes)
    q_weights = np.concatenate(q_weights)
    mu_nodes, mu_weights = np.polynomial.legendre.leggauss(angular_order)

    phi_h = np.zeros_like(radii)
    force_h = np.zeros_like(radii)
    truncation = benchmark.PLUMMER_TRUNCATION_KPC
    for index, radius in enumerate(radii):
        if radius == 0:
            separations = support * q_nodes
            phi, _ = benchmark.analytic_plummer(separations)
            phi_h[index] = np.sum(q_weights * 2 * phi)
            continue
        for q, radial_weight in zip(q_nodes, q_weights):
            offset = support * q
            mu_cross = (radius**2 + offset**2 - truncation**2) / (2 * radius * offset)
            if -1 < mu_cross < 1:
                intervals = ((-1.0, mu_cross), (mu_cross, 1.0))
            else:
                intervals = ((-1.0, 1.0),)
            for lower, upper in intervals:
                mu = lower + (upper - lower) * (mu_nodes + 1) / 2
                weights = (upper - lower) * mu_weights / 2
                separation = np.sqrt(np.maximum(
                    0.0, radius**2 + offset**2 - 2 * radius * offset * mu
                ))
                phi, force = benchmark.analytic_plummer(separation)
                radial_projection = np.divide(
                    radius - offset * mu, separation,
                    out=np.zeros_like(separation), where=separation > 0,
                )
                phi_h[index] += radial_weight * np.dot(weights, phi)
                force_h[index] += radial_weight * np.dot(
                    weights, force * radial_projection
                )
    return phi_h, force_h


def relative_error_summary(model, truth):
    relative = np.abs(np.asarray(model) - np.asarray(truth)) / np.abs(truth)
    p16, median, p84, p95 = np.percentile(relative, [16, 50, 84, 95])
    return {
        "p16": float(p16),
        "median": float(median),
        "p84": float(p84),
        "p95": float(p95),
        "maximum": float(np.max(relative)),
    }


def radial_cdf_max_deviation(positions):
    radius = np.linalg.norm(positions, axis=1)
    radius.sort()
    scale = benchmark.PLUMMER_SCALE_KPC
    truncation = benchmark.PLUMMER_TRUNCATION_KPC
    enclosed = (truncation / np.sqrt(truncation**2 + scale**2))**3
    cdf = (radius / np.sqrt(radius**2 + scale**2))**3 / enclosed
    empirical = np.arange(1, radius.size + 1) / radius.size
    return float(np.max(np.abs(cdf - empirical)))


def save_convergence_figure(rows, convergence_reference_grid, output_path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
    })
    fig, (conv_ax, error_ax) = plt.subplots(
        1, 2, figsize=(12.4, 5.2), gridspec_kw={"width_ratios": [1, 1]}
    )
    backend_style = {
        "native": ("#303030", "Native softened"),
        "agama": ("#e87500", "Stock Agama"),
    }
    field_style = {
        "potential": ("o", "-", r"$\Phi$"),
        "radial_force": ("^", "--", r"$F_r$"),
    }
    grids = np.asarray([row["grid_size_r"] for row in rows], dtype=float)
    coarse_rows = [row for row in rows
                   if row["grid_size_r"] < convergence_reference_grid]
    coarse_grids = np.asarray([row["grid_size_r"] for row in coarse_rows], dtype=float)

    for backend, (color, backend_label) in backend_style.items():
        finest = next(row[backend] for row in rows
                      if row["grid_size_r"] == convergence_reference_grid)
        for field, (marker, linestyle, field_label) in field_style.items():
            values = np.asarray([row[backend][field] for row in rows], dtype=float)
            reference = np.asarray(finest[field], dtype=float)
            convergence = np.asarray([
                relative_error_summary(row[backend][field], reference)["median"]
                for row in coarse_rows
            ])
            conv_ax.plot(
                coarse_grids, convergence, color=color, marker=marker,
                linestyle=linestyle, linewidth=1.35, markersize=5,
                label=f"{backend_label} {field_label}",
            )

            errors = np.asarray([
                row[backend]["matched_analytic_error"][field]["median"]
                for row in rows
            ])
            error_ax.plot(
                grids, errors, color=color, marker=marker,
                linestyle=linestyle, linewidth=1.35, markersize=5,
                label=f"{backend_label} {field_label}",
            )

    for axis in (conv_ax, error_ax):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(GRID_SIZES)
        axis.set_xticklabels([str(value) for value in GRID_SIZES])
        axis.grid(True, which="major", color="0.72", linestyle="--",
                  alpha=0.55, linewidth=0.8)
        axis.grid(True, which="minor", color="0.82", linestyle="--",
                  alpha=0.35, linewidth=0.65)
        axis.set_xlabel(r"Radial grid size, $N_r$")

    conv_ax.set_ylabel(
        r"Median $|\Delta X|/|X_{N_r=" + str(convergence_reference_grid) + r"}|$"
    )
    conv_ax.set_xlim(GRID_SIZES[0] * 0.85, convergence_reference_grid * 1.15)
    conv_ax.text(
        0.04, 0.04,
        f"Fixed sample: $N={PARTICLE_COUNT:,}$; reference $N_r={convergence_reference_grid}$",
        transform=conv_ax.transAxes, fontsize=8, color="0.3",
    )
    conv_ax.legend(frameon=False, fontsize=8, loc="upper right")

    error_ax.set_ylabel("Median absolute relative error")
    error_ax.set_xlim(GRID_SIZES[0] * 0.85, GRID_SIZES[-1] * 1.15)
    error_ax.text(
        0.04, 0.04,
        "Each method compared with its own continuum target:\n"
        "Native: kernel-convolved Plummer; Agama: unsoftened Plummer.",
        transform=error_ax.transAxes, fontsize=7.7, color="0.3",
    )
    error_ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    setUnits(length=1, mass=1, velocity=1)
    rng = np.random.default_rng(SEED)
    positions = np.ascontiguousarray(
        benchmark.sample_truncated_plummer(PARTICLE_COUNT, rng)
    )
    masses = np.full(PARTICLE_COUNT,
                     benchmark.PLUMMER_TOTAL_MASS_MSUN / PARTICLE_COUNT)
    support = benchmark.SOFTENING_AT_REFERENCE_KPC * (
        benchmark.SOFTENING_REFERENCE_N / PARTICLE_COUNT
    )**(1.0 / 3.0)
    radii = np.geomspace(
        0.05 * benchmark.PLUMMER_SCALE_KPC,
        benchmark.PLUMMER_TRUNCATION_KPC,
        benchmark.ERROR_PROBE_COUNT,
    )
    query = np.column_stack((radii, np.zeros_like(radii), np.zeros_like(radii)))
    truth_phi, truth_force = benchmark.analytic_plummer(radii)
    softened_phi, softened_force = softened_analytic_plummer(radii, support)
    # Independently verify normalization of the radial kernel quadrature.
    qx, qw = np.polynomial.legendre.leggauss(200)
    kernel_mass = 0.0
    for left, right in ((0.0, 0.5), (0.5, 1.0)):
        q = left + (right - left) * (qx + 1) / 2
        kernel_mass += float(4 * np.pi * np.sum(
            (right - left) * qw / 2 * q**2 * kernel_shape(q)
        ))

    worker = Path(__file__).with_name("_agama_potential_performance_worker.py")
    rows = []
    with tempfile.TemporaryDirectory(
        prefix="potential-grid-convergence-", dir=os.environ.get("TMPDIR")
    ) as scratch:
        input_path = Path(scratch) / "agama_grid_input.npz"
        np.savez(input_path, positions=positions, masses=masses,
                 query_radii=radii)
        for grid_size in GRID_SIZES:
            native = Potential(
                type="Multipole", particles=(positions, masses),
                softening=support, symmetry="s", lmax=0, mmax=0,
                rmin=benchmark.RMIN_KPC,
                rmax=benchmark.PLUMMER_TRUNCATION_KPC,
                gridSizeR=grid_size,
            )
            native_phi = np.asarray(native.potential(query), dtype=float).reshape(-1)
            native_force = np.asarray(native.force(query), dtype=float)[:, 0]
            del native

            result = subprocess.run(
                [sys.executable, str(worker), str(input_path), "0",
                 str(benchmark.RMIN_KPC),
                 str(benchmark.PLUMMER_TRUNCATION_KPC), str(grid_size)],
                check=True, capture_output=True, text=True,
            )
            agama_output = json.loads(result.stdout)
            agama_phi = np.asarray(agama_output["potential"], dtype=float)
            agama_force = np.asarray(agama_output["radial_force"], dtype=float)

            row = {
                "grid_size_r": grid_size,
                "native": {
                    "potential": native_phi.tolist(),
                    "radial_force": native_force.tolist(),
                    "matched_analytic_error": {
                        "potential": relative_error_summary(native_phi, softened_phi),
                        "radial_force": relative_error_summary(native_force, softened_force),
                    },
                    "unsoftened_model_error": {
                        "potential": relative_error_summary(native_phi, truth_phi),
                        "radial_force": relative_error_summary(native_force, truth_force),
                    },
                },
                "agama": {
                    "potential": agama_phi.tolist(),
                    "radial_force": agama_force.tolist(),
                    "matched_analytic_error": {
                        "potential": relative_error_summary(agama_phi, truth_phi),
                        "radial_force": relative_error_summary(agama_force, truth_force),
                    },
                },
            }
            rows.append(row)

    reference_grid = GRID_SIZES[-1]
    for row in rows:
        if row["grid_size_r"] == reference_grid:
            continue
        for backend in ("native", "agama"):
            row[backend]["grid_convergence_vs_reference"] = {
                field: relative_error_summary(
                    row[backend][field],
                    next(item[backend][field] for item in rows
                         if item["grid_size_r"] == reference_grid),
                )
                for field in ("potential", "radial_force")
            }

    figure_path = Path(os.environ.get(
        "POTENTIAL_GRID_CONVERGENCE_FIGURE",
        str(Path(__file__).resolve().parents[1]
            / "images" / "potential_grid_convergence.png"),
    ))
    save_convergence_figure(rows, reference_grid, figure_path)
    result = {
        "model": "truncated Plummer sphere",
        "particle_count": PARTICLE_COUNT,
        "seed": SEED,
        "sample_radial_cdf_max_deviation": radial_cdf_max_deviation(positions),
        "softening_support_kpc": float(support),
        "softening_rule": "h=0.2 kpc*(1000/N)^(1/3)",
        "fixed_rmin_kpc": benchmark.RMIN_KPC,
        "fixed_rmax_kpc": benchmark.PLUMMER_TRUNCATION_KPC,
        "symmetry": "spherical, lmax=0",
        "radial_grid_sizes": list(GRID_SIZES),
        "self_convergence_reference_grid": reference_grid,
        "error_probe_count": len(radii),
        "analytic_reference": {
            "native": "unsoftened truncated Plummer convolved with native cubic-spline kernel",
            "agama": "unsoftened analytic truncated Plummer",
            "native_kernel_mass_from_quadrature": kernel_mass,
            "kernel_shape_source": "potential_multipole.cpp softenedKernel; normalized in 3D",
        },
        "figure": str(figure_path),
        "rows": rows,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
