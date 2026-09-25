#!/usr/bin/env python3
"""Benchmark native softened Multipole build/evaluation against real TNG particles."""

import gc
import json
import os
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.gravity.kinematic_solver import construct_galaxy_potential_model
from kinematic_decompose.potential import Potential

RUN = "TNG100-3"
SNAP = 99
SUBHALO = 5
GRID_SIZE_R = 30
REPEATS = 7
LMAX_VALUES = (0, 2, 4, 6, 8, 10, 12)
PARTICLE_COUNTS = (256, 512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536)
SCALING_FIT_MIN_N = 1024
EXTRAPOLATE_TO_N = 1_000_000
PREDICTION_COUNTS = (1_000, 10_000, 100_000, 1_000_000)


def make_potential(positions, masses, support, symmetry, lmax, rmax):
    return Potential(
        type="Multipole",
        particles=(positions, masses),
        softening=support,
        symmetry=symmetry,
        lmax=lmax,
        mmax=lmax,
        rmin=0,
        rmax=rmax,
        gridSizeR=GRID_SIZE_R,
    )


def time_configuration(positions, masses, support, symmetry, lmax, rmax, query):
    build_times = []
    evaluation_times = []
    for _ in range(REPEATS):
        start = perf_counter()
        pot = make_potential(positions, masses, support, symmetry, lmax, rmax)
        build_times.append(perf_counter() - start)
        start = perf_counter()
        values = pot.potential(query)
        evaluation_times.append(perf_counter() - start)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Non-finite potential encountered during timing")
        del pot
        gc.collect()
    return {
        "build_median_s": float(np.median(build_times)),
        "build_min_s": float(np.min(build_times)),
        "evaluation_median_s": float(np.median(evaluation_times)),
        "evaluation_min_s": float(np.min(evaluation_times)),
    }


def save_scaling_figure(count_results, lmax_results):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    })
    fig, (ax_n, ax_l) = plt.subplots(1, 2, figsize=(10.2, 4.6))
    colors = {"spherical_monopole": "#4c72b0", "axisymmetric_l4": "#c44e52",
              "nonaxisymmetric_l4": "#55a868"}
    labels = {"spherical_monopole": "Spherical, $l_{\\max}=0$",
              "axisymmetric_l4": "Axisymmetric, $l_{\\max}=4$",
              "nonaxisymmetric_l4": "No symmetry, $l_{\\max}=4$"}

    fitted_n_slopes = {}
    for model, color in colors.items():
        rows = sorted((row for row in count_results if row["model"] == model),
                      key=lambda row: row["n_particles"])
        n = np.asarray([row["n_particles"] for row in rows], dtype=float)
        build = np.asarray([row["build_median_s"] for row in rows])
        fit_rows = [row for row in rows if row["n_particles"] >= SCALING_FIT_MIN_N]
        n_fit = np.asarray([row["n_particles"] for row in fit_rows], dtype=float)
        build_fit = np.asarray([row["build_median_s"] for row in fit_rows])
        alpha = float(np.polyfit(np.log(n_fit), np.log(build_fit), 1)[0])
        fitted_n_slopes[model] = alpha
        ax_n.loglog(n, build, "o-", color=color,
                    label=f"{labels[model]}: $\\alpha={alpha:.2f}$")
        coeff = np.polyfit(np.log(n_fit), np.log(build_fit), 1)
        n_projection = np.geomspace(n_fit[0], EXTRAPOLATE_TO_N, 120)
        projected_time = np.exp(coeff[1]) * n_projection**coeff[0]
        measured_end = n_fit[-1]
        measured_mask = n_projection <= measured_end
        ax_n.loglog(n_projection[measured_mask], projected_time[measured_mask],
                    "-", color=color, alpha=0.7, lw=1.2)
        ax_n.loglog(n_projection[~measured_mask], projected_time[~measured_mask],
                    ":", color=color, alpha=0.9, lw=1.6)
    sphere_rows = sorted((row for row in count_results
                         if row["model"] == "spherical_monopole"),
                         key=lambda row: row["n_particles"])
    sphere_fit = [row for row in sphere_rows if row["n_particles"] >= SCALING_FIT_MIN_N]
    nref = np.geomspace(SCALING_FIT_MIN_N, EXTRAPOLATE_TO_N, 120)
    tref = sphere_fit[0]["build_median_s"] * nref / sphere_fit[0]["n_particles"]
    ax_n.loglog(nref, tref, "k--", lw=1, label="$O(N)$ reference")
    ax_n.set(xlabel="Number of particles, $N$", ylabel="Potential build time [s]")
    ax_n.set_xlim(1e3, EXTRAPOLATE_TO_N)
    ax_n.axvline(sphere_rows[-1]["n_particles"], color="0.45", ls="--", lw=1,
                 label="measured range limit")
    ax_n.add_artist(ax_n.legend(handles=[Line2D([], [], color="0.2", ls=":",
                        label="power-law extrapolation")], frameon=False,
                        loc="lower right", fontsize=9))
    ax_n.text(0.03, 0.96, "(a)", transform=ax_n.transAxes, va="top", fontweight="bold")
    ax_n.legend(frameon=False, fontsize=9)

    orders = np.asarray([row["lmax"] + 1 for row in lmax_results], dtype=float)
    build_l = np.asarray([row["build_median_s"] for row in lmax_results])
    eval_l = np.asarray([row["evaluation_median_s"] for row in lmax_results])
    fit = orders >= 3  # lmax >= 2; omit lmax=0 from the nontrivial-order fit
    beta = float(np.polyfit(np.log(orders[fit]), np.log(build_l[fit]), 1)[0])
    ax_l.loglog(orders, build_l, "o-", color="#4c72b0",
                label=f"Build: $\\beta={beta:.2f}$")
    ax_l.loglog(orders, eval_l, "s--", color="#c44e52", label="Evaluation")
    xfit = np.geomspace(orders[fit][0], orders[fit][-1], 60)
    fit_coeff = np.polyfit(np.log(orders[fit]), np.log(build_l[fit]), 1)
    ax_l.loglog(xfit, np.exp(fit_coeff[1]) * xfit**fit_coeff[0],
                ":", color="#4c72b0", lw=1, label="Build power-law fit")
    ax_l.set(xlabel="Expansion order index, lmax + 1",
             ylabel="Time [s]")
    ax_l.text(0.03, 0.96, "(b)", transform=ax_l.transAxes, va="top", fontweight="bold")
    ax_l.legend(frameon=False)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.96, wspace=0.30)
    path = os.environ.get("POTENTIAL_PERFORMANCE_FIGURE", "/tmp/potential_performance_scaling.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path, fitted_n_slopes, beta


def main():
    snapshot = Snapshot(f"/Users/yuwa/sims.TNG/{RUN}/output", SNAP)
    snapshot.load_particle(ID=SUBHALO, load_particle_fields="potential")
    snapshot.physical_units()
    snapshot.load_group_catalog(ID=SUBHALO)
    snapshot.GC_physical_units()
    snapshot.center(cen=snapshot.group_catalog["SubhaloPos"])
    snapshot.faceon(
        align_with="star",
        range=[3*snapshot.properties["eps"], 5*snapshot.s.r50],
        as_context=False,
    )
    galaxy = snapshot.container
    support = 2.8 * float(galaxy.properties["eps"])
    rmax = float(galaxy.R_vir)
    query = np.ascontiguousarray(np.asarray(galaxy.s["pos"], dtype=float))

    dm_pos = np.ascontiguousarray(np.asarray(galaxy.dm["pos"], dtype=float))
    dm_mass = np.full(len(dm_pos), float(galaxy.properties["mDM"]))
    rng = np.random.default_rng(2026)
    dm_order = rng.permutation(len(dm_pos))
    n_values = sorted(set(PARTICLE_COUNTS + (len(dm_pos),)))
    n_results = []
    for n in n_values:
        choice = dm_order[np.arange(n) % len(dm_order)]
        positions = np.ascontiguousarray(dm_pos[choice])
        masses = np.ascontiguousarray(dm_mass[choice])
        for name, symmetry, lmax in (("spherical_monopole", "s", 0),
                                     ("axisymmetric_l4", "a", 4),
                                     ("nonaxisymmetric_l4", "n", 4)):
            row = {"n_particles": n, "model": name}
            row.update(time_configuration(positions, masses, support, symmetry, lmax, rmax, query))
            n_results.append(row)

    star_pos = np.ascontiguousarray(np.asarray(galaxy.s["pos"], dtype=float))
    star_mass = np.ascontiguousarray(np.asarray(galaxy.s["mass"], dtype=float))
    lmax_results = []
    for lmax in LMAX_VALUES:
        row = {"n_particles": len(star_pos), "symmetry": "axisymmetric", "lmax": lmax}
        row.update(time_configuration(star_pos, star_mass, support, "a", lmax, rmax, query))
        lmax_results.append(row)

    pipeline_build_times = []
    pipeline_eval_times = []
    for _ in range(REPEATS):
        start = perf_counter()
        pot = construct_galaxy_potential_model(galaxy)
        pipeline_build_times.append(perf_counter() - start)
        start = perf_counter()
        values = pot.potential(np.ascontiguousarray(np.asarray(galaxy["pos"], dtype=float)))
        pipeline_eval_times.append(perf_counter() - start)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Non-finite pipeline potential encountered")
        del pot
        gc.collect()

    slopes = {}
    extrapolated_build_times = {}
    for model in ("spherical_monopole", "axisymmetric_l4", "nonaxisymmetric_l4"):
        rows = [row for row in n_results if row["model"] == model
                and row["n_particles"] >= SCALING_FIT_MIN_N]
        fit_coeff = np.polyfit(
            np.log([row["n_particles"] for row in rows]),
            np.log([row["build_median_s"] for row in rows]), 1)
        slopes[model] = float(fit_coeff[0])
        extrapolated_build_times[model] = {
            str(n): float(np.exp(fit_coeff[1]) * n**fit_coeff[0])
            for n in PREDICTION_COUNTS
        }

    figure_path, fitted_n_slopes, lmax_build_slope = save_scaling_figure(n_results, lmax_results)

    print(json.dumps({
        "sample": {"run": RUN, "snap": SNAP, "subhalo": SUBHALO,
                   "n_dm": len(dm_pos), "n_star": len(star_pos),
                   "n_gas": len(galaxy.g), "kernel_support_kpc": support,
                   "gridSizeR": GRID_SIZE_R, "rmax_kpc": rmax,
                   "repeats": REPEATS},
        "pipeline_build_median_s": float(np.median(pipeline_build_times)),
        "pipeline_eval_median_s": float(np.median(pipeline_eval_times)),
        "particle_count_scaling_exponent": slopes,
        "particle_count_scaling_exponent_plotted": fitted_n_slopes,
        "fitted_build_time_estimates_s": extrapolated_build_times,
        "lmax_build_scaling_exponent_plotted": lmax_build_slope,
        "particle_count_fit_range": [SCALING_FIT_MIN_N,
                                     max(row["n_particles"] for row in n_results)],
        "N_above_real_sample_uses_deterministic_repeats": True,
        "particle_count_extrapolation_limit": EXTRAPOLATE_TO_N,
        "figure": figure_path,
        "by_particle_count": n_results,
        "by_lmax": lmax_results,
    }, indent=2))


if __name__ == "__main__":
    main()
