"""Performance schema/plot tests; fixtures are synthetic, not benchmark claims."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("example_potential_performance.py")
spec = importlib.util.spec_from_file_location("performance_benchmark", SCRIPT)
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def synthetic_report():
    config = {"particle_counts": [100, 1000],
              "backends": list(benchmark.BACKENDS), "repeats": 3,
              "grid_size": 30}
    rows = []
    for operation in benchmark.OPERATIONS:
        for symmetry in benchmark.SYMMETRIES:
            for backend in config["backends"]:
                counts = config["particle_counts"]
                for count in counts:
                    times = np.array([1., 2., 3.]) * count * 1e-6
                    low, median, high = np.percentile(times, [16, 50, 84])
                    rows.append({"operation": operation, "symmetry": symmetry, "backend": backend,
                                 "size": count, "times_s": times.tolist(), "p16_s": low,
                                 "median_s": median, "p84_s": high})
    worker_count = len(benchmark.SYMMETRIES) * len(config["backends"]) * len(config["particle_counts"])
    return {"configuration": config, "rows": rows, "workers": [{} for _ in range(worker_count)]}


def test_performance_plot_contains_only_timing_and_all_symmetries(tmp_path):
    fig = benchmark.save_scaling_figure(synthetic_report(), tmp_path / "timings.png")
    try:
        assert len(fig.axes) == 1
        assert not fig.texts
        assert not fig.legends
        assert benchmark.BACKENDS.keys() == {"native_softened", "agama"}
        for ax in fig.axes:
            assert ax.get_xscale() == ax.get_yscale() == "log"
            assert ax.get_title() == ""
            assert "time (s)" in ax.get_ylabel()
            assert len(ax.containers) == 8
            assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())
        ax = fig.axes[0]
        assert ax.get_xlabel() == r"$N$"
        assert ax.get_box_aspect() == 1
        legends = [ax.get_legend(), *ax.artists]
        assert all(legend is not None for legend in legends)
        labels = [text.get_text() for legend in legends for text in legend.get_texts()]
        assert all(spec["label"] + " (" + key + ")" in labels
                   for key, spec in benchmark.SYMMETRIES.items())
        assert all(item[0] in labels for item in benchmark.BACKENDS.values())
        assert "this work" in labels and "Agama" in labels
        assert not any("unsoftened" in label.lower() for label in labels)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_box = ax.get_window_extent(renderer)
        for legend in legends:
            box = legend.get_window_extent(renderer)
            assert axes_box.contains(box.x0, box.y0)
            assert axes_box.contains(box.x1, box.y1)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "repeats", "summary", "workers"])
def test_incomplete_or_inconsistent_measurements_are_rejected(damage):
    report = copy.deepcopy(synthetic_report())
    if damage == "missing":
        report["rows"].pop()
    elif damage == "duplicate":
        report["rows"].append(report["rows"][0])
    elif damage == "repeats":
        report["rows"][0]["times_s"].pop()
    elif damage == "summary":
        report["rows"][0]["median_s"] *= 2
    else:
        report["workers"].pop()
    with pytest.raises((ValueError, AssertionError)):
        benchmark.validate_results(report)


@pytest.mark.parametrize("symmetry", list(benchmark.SYMMETRIES))
def test_real_native_worker_times_only_construction(tmp_path, symmetry):
    input_path = tmp_path / "input.npz"
    np.savez(input_path, positions=benchmark.source_sample(32, 31), masses=np.ones(32))
    angular = benchmark.SYMMETRIES[symmetry]
    config = {"stage": "build", "backend": "native_softened", "symmetry": symmetry,
              "lmax": angular["lmax"], "mmax": angular["mmax"], "rmin": .001, "rmax": 30.,
              "grid_size": 30, "support": .02, "repeats": 2}
    built = benchmark.run_worker(input_path, config)
    assert len(built["results"]) == 1
    assert built["results"][0]["operation"] == "build"
    assert "query_sha256" not in built
    for row in built["results"]:
        assert len(row["times_s"]) == 2
        assert row["median_s"] > 0
    assert all(value == "1" for value in built["runtime"]["threads"].values())


def test_plot_only_does_not_start_workers(tmp_path):
    data_path = tmp_path / "timings.json"
    data_path.write_text(json.dumps(synthetic_report()))
    figure_path = tmp_path / "timings.png"
    result = subprocess.run([sys.executable, str(SCRIPT), "--plot-only", "--results", str(data_path),
                             "--figure", str(figure_path)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["timing_cases"] == 16
    assert figure_path.is_file()


def test_default_particle_scan_matches_requested_range():
    args = benchmark.parse_args([])
    assert args.min_particles == 100_000
    assert args.max_particles == 10_000_000
    assert not any("query" in name for name in vars(args))


def test_legacy_cache_import_preserves_original_build_samples(tmp_path):
    report = synthetic_report()
    original_rows = copy.deepcopy(report["rows"])
    report["workers"] = [{"config": {"stage": "build"}} for _ in report["workers"]]
    report["rows"].append({"operation": "potential"})
    report["workers"].append({"config": {"stage": "query"}})
    report["configuration"]["query_counts"] = [1]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(report))
    loaded = benchmark.load_build_measurements(path)
    assert loaded["rows"] == original_rows
    assert "query_counts" not in loaded["configuration"]
    assert loaded["source_measurements"] == str(path.resolve())
