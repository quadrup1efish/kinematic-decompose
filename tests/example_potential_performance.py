#!/usr/bin/env python3
"""Performance-only Multipole benchmark: construction versus particle count.

No analytic reference, field profile, accuracy metric or speed assertion belongs
here. Native and stock Agama run serially in separate processes. Timed regions
exclude imports, input generation, disk I/O, warmup and destruction.
"""
import os

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SYMMETRIES = {
    "s": {"label": "Spherical", "lmax": 0, "mmax": 0, "color": "#333333"},
    "a": {"label": "Axisymmetric", "lmax": 4, "mmax": 0, "color": "#0072B2"},
    "t": {"label": "Triaxial", "lmax": 4, "mmax": 4, "color": "#D55E00"},
    "n": {"label": "No symmetry", "lmax": 4, "mmax": 4, "color": "#009E73"},
}
BACKENDS = {
    "native_softened": ("this work", "-", "o"),
    "agama": ("Agama", "--", "s"),
}
OPERATIONS = ("build",)


def logarithmic_counts(low, high, samples):
    if low < 1 or high < low or samples < 1:
        raise ValueError("Require 1 <= minimum <= maximum and positive sample count")
    return np.unique(np.rint(np.geomspace(low, high, samples)).astype(int)).tolist()


def source_sample(count, seed):
    """Common reflection-symmetric triaxial Gaussian in expectation, not truth."""
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.normal(size=(count, 3)) * [1.0, 0.7, 0.4])


def run_worker(input_path, config):
    python = os.environ.get("AGAMA_PYTHON", sys.executable) if config["backend"] == "agama" else sys.executable
    worker = Path(__file__).with_name("_potential_performance_worker.py")
    env = os.environ.copy()
    env.update({name: "1" for name in THREAD_VARIABLES})
    result = subprocess.run([python, str(worker), str(input_path), json.dumps(config)],
                            env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Benchmark worker failed:\n" + result.stderr)
    output = json.loads(result.stdout)
    if output["config"] != config:
        raise ValueError("Worker configuration does not match the requested case")
    if any(value != "1" for value in output["runtime"]["threads"].values()):
        raise ValueError("Worker thread limits are not controlled")
    return output


def validate_results(report):
    config = report["configuration"]
    rows = report["rows"]
    actual = [(row["operation"], row["symmetry"], row["backend"], row["size"]) for row in rows]
    expected = {(op, sym, backend, count)
                for op in OPERATIONS for sym in SYMMETRIES for backend in config["backends"]
                for count in config["particle_counts"]}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Missing, duplicated or unexpected benchmark cases")
    for row in rows:
        times = np.asarray(row["times_s"], dtype=float)
        if len(times) != config["repeats"] or not np.isfinite(times).all() or np.any(times <= 0):
            raise ValueError("Invalid raw timing samples")
        np.testing.assert_allclose([row["p16_s"], row["median_s"], row["p84_s"]],
                                   np.percentile(times, [16, 50, 84]), rtol=1e-14)
    if len(report["workers"]) != len(SYMMETRIES) * len(config["backends"]) * len(config["particle_counts"]):
        raise ValueError("Incomplete worker provenance")


def benchmark(args):
    particle_counts = logarithmic_counts(args.min_particles, args.max_particles, args.particle_samples)
    backends = [name for name in BACKENDS if name != "agama" or not args.native_only]
    config = {"particle_counts": particle_counts, "backends": backends,
              "repeats": args.repeats, "grid_size": args.grid_size, "support_kpc": args.support,
              "rmin_kpc": 0.001, "rmax_kpc": 30.0, "seed": args.seed,
              "symmetries": SYMMETRIES, "threads": 1,

              "source_distribution": "Gaussian axes (1,0.7,0.4) kpc; fixed total mass 1e10 Msun",
              "timing_scope": "public constructor only; one warmup per case; destruction excluded",
              "comparison_scope": "matched inputs/configurations, not equal-accuracy certification"}
    report = {"configuration": config, "rows": [], "workers": []}
    sources = source_sample(max(particle_counts), args.seed)
    scratch_root = Path(os.environ.get("TMPDIR", Path.home() / ".hermes/cache/scratch"))
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="potential-performance-", dir=scratch_root) as scratch:
        input_path = Path(scratch) / "input.npz"
        for index, count in enumerate(particle_counts):
            np.savez(input_path, positions=sources[:count], masses=np.full(count, 1e10 / count))
            identity = None
            for symmetry, spec in SYMMETRIES.items():
                order = backends if index % 2 == 0 else backends[::-1]
                for backend in order:
                    worker_config = {"stage": "build", "backend": backend, "symmetry": symmetry,
                                     "lmax": spec["lmax"], "mmax": spec["mmax"],
                                     "grid_size": args.grid_size, "rmin": 0.001, "rmax": 30.0,
                                     "support": args.support, "repeats": args.repeats}
                    result = run_worker(input_path, worker_config)
                    hashes = tuple(result[key] for key in ("source_sha256", "mass_sha256"))
                    if identity is not None and hashes != identity:
                        raise ValueError("Input digests differ across symmetry/backend cases")
                    identity = hashes
                    report["workers"].append({key: value for key, value in result.items() if key != "results"})
                    for row in result["results"]:
                        report["rows"].append({"symmetry": symmetry, "backend": backend, **row})
            args.results.parent.mkdir(parents=True, exist_ok=True)
            args.results.write_text(json.dumps(report, indent=2))
            print("Completed build N=" + str(count), flush=True)
    validate_results(report)
    return report


def load_build_measurements(path):
    """Extract original construction measurements from a legacy mixed cache."""
    report = json.loads(Path(path).read_text())
    if any(row["operation"] != "build" for row in report["rows"]):
        report["rows"] = [row for row in report["rows"] if row["operation"] == "build"]
        report["workers"] = [worker for worker in report["workers"]
                             if worker["config"]["stage"] == "build"]
        for key in ("query_counts", "query_source_particles", "query_distribution"):
            report["configuration"].pop(key, None)
        report["configuration"]["timing_scope"] = "public constructor only; one warmup per case; destruction excluded"
        report["source_measurements"] = str(Path(path).resolve())
    validate_results(report)
    return report


def save_scaling_figure(report, path):
    validate_results(report)
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 14,
                         "axes.labelsize": 19, "xtick.labelsize": 16, "ytick.labelsize": 16,
                         "legend.fontsize": 13, "legend.frameon": False, "axes.grid": False})
    fig, ax = plt.subplots(figsize=(7, 7))
    for symmetry, spec in SYMMETRIES.items():
        for backend in report["configuration"]["backends"]:
            _, style, marker = BACKENDS[backend]
            rows = sorted((row for row in report["rows"]
                           if row["symmetry"] == symmetry and row["backend"] == backend),
                          key=lambda row: row["size"])
            x = [row["size"] for row in rows]
            median = np.array([row["median_s"] for row in rows])
            low = np.array([row["p16_s"] for row in rows])
            high = np.array([row["p84_s"] for row in rows])
            ax.errorbar(x, median, yerr=np.vstack((median-low, high-median)),
                        color=spec["color"], linestyle=style, marker=marker,
                        markerfacecolor="none", markersize=5, linewidth=1.8,
                        capsize=2, elinewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("Construction time (s)")
    ax.set_box_aspect(1)
    symmetry_handles = [Line2D([], [], color=spec["color"], linewidth=2.5,
                               label=spec["label"] + " (" + sym + ")") for sym, spec in SYMMETRIES.items()]
    backend_handles = [Line2D([], [], color="black", linestyle=BACKENDS[key][1],
                              marker=BACKENDS[key][2], markerfacecolor="none", label=BACKENDS[key][0])
                       for key in report["configuration"]["backends"]]
    symmetry_legend = ax.legend(handles=symmetry_handles, loc="upper left")
    ax.add_artist(symmetry_legend)
    ax.legend(handles=backend_handles, loc="lower right")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-particles", type=int, default=100_000)
    parser.add_argument("--max-particles", type=int, default=10_000_000)
    parser.add_argument("--particle-samples", type=int, default=7)

    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--support", type=float, default=0.02, help="Fixed compact-spline support h in kpc")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Read saved measurements without rerunning timings")
    parser.add_argument("--results", type=Path, default=Path(os.environ.get("TMPDIR", Path.home() / ".hermes/cache/scratch")) / "potential_performance_results.json")
    parser.add_argument("--figure", type=Path, default=Path(os.environ.get("POTENTIAL_PERFORMANCE_FIGURE", Path(__file__).resolve().parents[1] / "images/potential_performance_scaling.png")))
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.grid_size < 4 or not np.isfinite(args.support) or args.support <= 0:
        parser.error("Invalid count, grid size or support radius")
    return args


def main():
    args = parse_args()
    report = load_build_measurements(args.results) if args.plot_only else benchmark(args)
    figure = save_scaling_figure(report, args.figure)
    plt.close(figure)
    print(json.dumps({"figure": str(args.figure), "measurements": str(args.results),
                      "timing_cases": len(report["rows"])}))


if __name__ == "__main__":
    main()
