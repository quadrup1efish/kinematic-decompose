#!/usr/bin/env python3
"""Benchmark softened Multipole construction versus stock Agama on a Plummer model.

The stock Agama and project-native extensions are measured in separate processes:
loading both native libraries together can collide at the symbol level.
"""

import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kinematic_decompose.potential import Potential, setUnits

GRID_SIZE_R = 30
REPEATS = 9
MIN_PARTICLES = 100_000
MAX_PARTICLES = 10_000_000
N_LOG_SAMPLES = 13
PLUMMER_SCALE_KPC = 1.0
PLUMMER_TRUNCATION_KPC = 10.0
PLUMMER_TOTAL_MASS_MSUN = 1.0e10
RMIN_KPC = 1.0e-3
SOFTENING_AT_REFERENCE_KPC = 0.2
SOFTENING_REFERENCE_N = 1_000
ERROR_PROBE_COUNT = 64


def summarize(times):
    values = np.asarray(times, dtype=float)
    p16, median, p84 = np.percentile(values, [16, 50, 84])
    return {
        "build_times_s": values.tolist(),
        "build_p16_s": float(p16),
        "build_median_s": float(median),
        "build_p84_s": float(p84),
    }


def time_native(positions, masses, support, rmax, query):
    times = []
    for _ in range(REPEATS):
        start = perf_counter()
        potential = Potential(
            type="Multipole",
            particles=(positions, masses),
            softening=support,
            symmetry="s",
            lmax=0,
            mmax=0,
            rmin=RMIN_KPC,
            rmax=rmax,
            gridSizeR=GRID_SIZE_R,
        )
        times.append(perf_counter() - start)
        del potential
        gc.collect()
    potential = Potential(
        type="Multipole", particles=(positions, masses), softening=support,
        symmetry="s", lmax=0, mmax=0, rmin=RMIN_KPC, rmax=rmax,
        gridSizeR=GRID_SIZE_R,
    )
    phi = np.asarray(potential.potential(query), dtype=float).reshape(-1)
    force = np.asarray(potential.force(query), dtype=float)
    result = summarize(times)
    result["potential_probe"] = phi.tolist()
    result["radial_force_probe"] = force[:, 0].tolist()
    return result


def time_stock_agama(positions, masses, rmax, query, scratch_dir):
    worker = Path(__file__).with_name("_agama_potential_performance_worker.py")
    input_path = Path(scratch_dir) / "agama_benchmark_input.npz"
    np.savez(input_path, positions=positions, masses=masses,
             query_radii=query[:, 0])
    result = subprocess.run(
        [sys.executable, str(worker), str(input_path), str(REPEATS),
         str(RMIN_KPC), str(rmax)],
        check=True, capture_output=True, text=True,
    )
    output = json.loads(result.stdout)
    summary = summarize(output["build_times_s"])
    summary["potential_probe"] = output["potential"]
    summary["radial_force_probe"] = output["radial_force"]
    return summary


def analytic_plummer(radii):
    """Potential and inward radial-force magnitude of the truncated Plummer model."""
    radii = np.asarray(radii, dtype=float)
    scale = PLUMMER_SCALE_KPC
    truncation = PLUMMER_TRUNCATION_KPC
    enclosed_fraction = (truncation / np.sqrt(truncation**2 + scale**2))**3
    source_mass = PLUMMER_TOTAL_MASS_MSUN / enclosed_fraction
    g = 4.30091727067736e-6  # kpc (km/s)^2 / Msun
    inside = radii <= truncation
    phi = np.empty_like(radii)
    force = np.empty_like(radii)
    boundary_term = scale**2 / (truncation**2 + scale**2)**1.5
    phi[inside] = -g * source_mass * (
        1 / np.sqrt(radii[inside]**2 + scale**2) - boundary_term
    )
    force[inside] = -(g * source_mass * radii[inside]
                      / (radii[inside]**2 + scale**2)**1.5)
    phi[~inside] = -g * PLUMMER_TOTAL_MASS_MSUN / radii[~inside]
    force[~inside] = -g * PLUMMER_TOTAL_MASS_MSUN / radii[~inside]**2
    return phi, force


def error_summary(model, truth):
    absolute_relative = np.abs(np.asarray(model) - truth) / np.abs(truth)
    p16, median, p84 = np.percentile(absolute_relative, [16, 50, 84])
    return {"p16": float(p16), "median": float(median), "p84": float(p84)}


def save_scaling_figure(rows):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })
    fig, (ax, err_ax) = plt.subplots(1, 2, figsize=(12.4, 5.2),
                                     gridspec_kw={"width_ratios": [1, 1]})
    styles = [
        ("native", "#303030", "o", "Native softened"),
        ("agama", "#e87500", "s", "Stock Agama, unsoftened"),
    ]
    exponents = {}
    ordered = sorted(rows, key=lambda row: row["n_particles"])
    n = np.asarray([row["n_particles"] for row in ordered], dtype=float)
    for key, color, marker, label in styles:
        med = np.asarray([row[key]["build_median_s"] for row in ordered])
        lo = np.asarray([row[key]["build_p16_s"] for row in ordered])
        hi = np.asarray([row[key]["build_p84_s"] for row in ordered])
        errors = np.vstack((med - lo, hi - med))
        ax.errorbar(n, med, yerr=errors, color=color, marker=marker,
                    linestyle="none", linewidth=1.1, markersize=5.5,
                    capsize=2.5, elinewidth=1.0, zorder=3)
        ax.plot(n, med, color=color, linewidth=1.05, alpha=0.7, zorder=2)
        high_n = n > MIN_PARTICLES
        exponent, intercept = np.polyfit(np.log(n[high_n]), np.log(med[high_n]), 1)
        exponents[key] = float(exponent)
        fit_n = np.geomspace(n[high_n][0], n[high_n][-1], 60)
        ax.loglog(fit_n, np.exp(intercept) * fit_n**exponent,
                  linestyle="--", color=color, linewidth=1.7,
                  alpha=0.8, zorder=1)
        fit_end = float(np.exp(intercept) * MAX_PARTICLES**exponent)
        ax.annotate(rf"$\alpha={exponent:.2f}$",
                    xy=(MAX_PARTICLES, fit_end), xytext=(-5, 4 if key == "native" else -9),
                    textcoords="offset points", ha="right", va="center",
                    color=color, fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(MIN_PARTICLES, MAX_PARTICLES)
    ax.set_xlabel(r"Number of particles, $N_{\mathrm{part}}$")
    ax.set_ylabel("Potential build time [s]")
    for i, (key, color, _, label) in enumerate(styles):
        ax.text(0.035, 0.95 - 0.065 * i, label, transform=ax.transAxes,
                ha="left", va="top", color=color, fontsize=11.5)
    ax.text(0.035, 0.79,
            "Fixed grid and $l_{\\max}=0$; expected cost $O(N)$",
            transform=ax.transAxes, ha="left", va="top", color="0.3", fontsize=9)
    ax.grid(True, which="major", color="0.72", linestyle="--", alpha=0.55, linewidth=0.8)
    ax.grid(True, which="minor", color="0.82", linestyle="--", alpha=0.35, linewidth=0.65)
    error_styles = [
        ("native", "potential_error", "#303030", "o", "-", r"Native $|\Delta\Phi/\Phi|$"),
        ("agama", "potential_error", "#e87500", "s", "-", r"Agama $|\Delta\Phi/\Phi|$"),
        ("native", "force_error", "#303030", "^", "--", r"Native $|\Delta F_r/F_r|$"),
        ("agama", "force_error", "#e87500", "v", "--", r"Agama $|\Delta F_r/F_r|$"),
    ]
    for backend, metric, color, marker, linestyle, label in error_styles:
        med = np.asarray([row[backend][metric]["median"] for row in ordered])
        err_ax.plot(n, med, color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.35, markersize=4.2, label=label)
    err_ax.set_xscale("log")
    err_ax.set_yscale("log")
    err_ax.set_xlim(MIN_PARTICLES, MAX_PARTICLES)
    err_ax.set_xlabel(r"Number of particles, $N_{\mathrm{part}}$")
    error_medians = [row[backend][metric]["median"]
                     for row in ordered for backend, metric, *_ in error_styles]
    err_ax.set_ylim(max(min(error_medians) * 0.5, 1e-8), max(error_medians) * 2)
    err_ax.set_ylabel("Median absolute relative error")
    err_ax.text(0.04, 0.025,
                "Median over $r=0.05$–$10$ kpc\n"
                "$h(N)=0.2\\,\\mathrm{kpc}(1000/N)^{1/3}$; Native includes softening bias.",
                transform=err_ax.transAxes, fontsize=7.7, color="0.3",
                va="bottom")
    err_ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
    err_ax.grid(True, which="major", color="0.72", linestyle="--", alpha=0.55, linewidth=0.8)
    err_ax.grid(True, which="minor", color="0.82", linestyle="--", alpha=0.35, linewidth=0.65)
    fig.tight_layout()
    default_path = Path(__file__).resolve().parents[1] / "images" / "potential_performance_scaling.png"
    path = os.environ.get("POTENTIAL_PERFORMANCE_FIGURE", str(default_path))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path, exponents


def sample_truncated_plummer(count, rng):
    """Draw from a Plummer sphere truncated at the fixed outer radius."""
    scale = PLUMMER_SCALE_KPC
    truncation = PLUMMER_TRUNCATION_KPC
    enclosed_fraction = (truncation / np.sqrt(truncation**2 + scale**2))**3
    q = rng.random(count) * enclosed_fraction
    q23 = q**(2.0 / 3.0)
    radius = scale * np.sqrt(q23 / (1.0 - q23))
    cosine = rng.uniform(-1.0, 1.0, count)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    sine = np.sqrt(1.0 - cosine**2)
    return np.column_stack((radius * sine * np.cos(azimuth),
                            radius * sine * np.sin(azimuth),
                            radius * cosine))


def main():
    setUnits(length=1, mass=1, velocity=1)
    rng = np.random.default_rng(2026)
    all_positions = sample_truncated_plummer(MAX_PARTICLES, rng)
    rmax = PLUMMER_TRUNCATION_KPC

    counts = np.unique(np.rint(np.geomspace(
        MIN_PARTICLES, MAX_PARTICLES, N_LOG_SAMPLES
    )).astype(int))
    rows = []
    with tempfile.TemporaryDirectory(prefix="potential-perf-", dir=os.environ.get("TMPDIR")) as scratch:
        for n in counts:
            positions = np.ascontiguousarray(all_positions[:n])
            masses = np.full(n, PLUMMER_TOTAL_MASS_MSUN / n)
            support = SOFTENING_AT_REFERENCE_KPC * (
                SOFTENING_REFERENCE_N / n
            )**(1.0 / 3.0)
            radii = np.geomspace(0.05 * PLUMMER_SCALE_KPC,
                                 PLUMMER_TRUNCATION_KPC, ERROR_PROBE_COUNT)
            query = np.column_stack((radii, np.zeros_like(radii), np.zeros_like(radii)))
            truth_phi, truth_force = analytic_plummer(radii)
            native = time_native(positions, masses, support, rmax, query)
            agama = time_stock_agama(positions, masses, rmax, query, scratch)
            for result in (native, agama):
                result["potential_error"] = error_summary(result["potential_probe"], truth_phi)
                result["force_error"] = error_summary(result["radial_force_probe"], truth_force)
            rows.append({"n_particles": int(n),
                         "softening_support_kpc": float(support),
                         "native": native, "agama": agama})

    figure, exponents = save_scaling_figure(rows)
    radii = np.linalg.norm(all_positions, axis=1)
    sorted_radii = np.sort(radii)
    q = (sorted_radii / np.sqrt(sorted_radii**2 + PLUMMER_SCALE_KPC**2))**3
    enclosed_fraction = (PLUMMER_TRUNCATION_KPC /
                         np.sqrt(PLUMMER_TRUNCATION_KPC**2 + PLUMMER_SCALE_KPC**2))**3
    empirical_cdf = np.arange(1, MAX_PARTICLES + 1) / MAX_PARTICLES
    max_cdf_deviation = float(np.max(np.abs(q / enclosed_fraction - empirical_cdf)))
    print(json.dumps({
        "analytic_model": {"density": "truncated Plummer sphere",
                   "potential_inside": "-G*M0*(1/sqrt(r^2+a^2)-a^2/(Rt^2+a^2)^(3/2)); M0=Mtotal/[Rt^3/(Rt^2+a^2)^(3/2)]",
                   "potential_outside": "-G*Mtotal/r",
                   "scale_radius_kpc": PLUMMER_SCALE_KPC,
                   "truncation_radius_kpc": PLUMMER_TRUNCATION_KPC,
                   "total_mass_msun": PLUMMER_TOTAL_MASS_MSUN,
                   "softening_rule": "h=0.2 kpc*(1000/N)^(1/3)",
                   "sample_size": MAX_PARTICLES,
                   "radial_cdf_max_deviation": max_cdf_deviation,
                   "seed": 2026},
        "sample": {"n_source_particles": MAX_PARTICLES,
                   "gridSizeR": GRID_SIZE_R, "rmin_kpc": RMIN_KPC,
                   "rmax_kpc": rmax,
                   "symmetry": "spherical", "lmax": 0,
                   "native_kernel": "compact cubic spline, support h",
                   "error_reference": "unsoftened analytic truncated Plummer; native errors include the deliberate softening bias",
                   "error_probe_radii_kpc": np.geomspace(
                       0.05 * PLUMMER_SCALE_KPC, PLUMMER_TRUNCATION_KPC,
                       ERROR_PROBE_COUNT).tolist(),
                   "agama_kernel": "stock unsoftened particle source",
                   "repeats_per_N_per_backend": REPEATS,
                   "errorbar_percentiles": [16, 84]},
        "n_sampling": "rounded geomspace from 1e5 to 1e7; nested prefixes of one independent analytic-model realization; masses renormalized to fixed total mass at each N",
        "asymptotic_complexity_at_fixed_grid_lmax": "O(N) for both particle-source builders",
        "measured_loglog_slope_N_gt_1e5": exponents,
        "figure": figure,
        "by_particle_count": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
