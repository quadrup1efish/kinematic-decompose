"""Serial, isolated timing worker; no analytic fields or accuracy metrics."""
import os

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"

import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


def digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def summarize(times):
    values = np.asarray(times, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Timing samples must be finite and positive")
    low, median, high = np.percentile(values, [16, 50, 84])
    return {"times_s": values.tolist(), "p16_s": float(low),
            "median_s": float(median), "p84_s": float(high)}


def main():
    config = json.loads(sys.argv[2])
    backend = config["backend"]
    if backend not in ("native_softened", "agama"):
        raise ValueError("Unsupported performance backend: " + backend)
    if backend == "agama":
        module = importlib.import_module("agama")
        extension = module
    else:
        module = importlib.import_module("kinematic_decompose.potential")
        module.set_num_threads(1)
        extension = importlib.import_module("kinematic_decompose.potential._potential")
    module.setUnits(length=1, mass=1, velocity=1)
    with np.load(sys.argv[1]) as data:
        positions = data["positions"].copy()
        masses = data["masses"].copy()

    kwargs = dict(type="Multipole", particles=(positions, masses),
                  symmetry=config["symmetry"], lmax=config["lmax"], mmax=config["mmax"],
                  rmin=config["rmin"], rmax=config["rmax"], gridSizeR=config["grid_size"])
    if backend == "native_softened":
        kwargs["softening"] = config["support"]
    repeats = config["repeats"]
    if config["stage"] != "build":
        raise ValueError("Only construction timing is supported")
    warmup = module.Potential(**kwargs)
    del warmup
    gc.collect()
    times = []
    for _ in range(repeats):
        start = perf_counter()
        potential = module.Potential(**kwargs)
        times.append(perf_counter() - start)
        del potential
        gc.collect()
    results = [{"operation": "build", "size": len(positions), **summarize(times)}]
    module_file = extension.__file__
    assert module_file is not None
    binary = Path(module_file).resolve()
    print(json.dumps({"config": config, "source_sha256": digest(positions),
                      "mass_sha256": digest(masses),
                      "runtime": {"python": sys.version, "numpy": np.__version__,
                                  "extension": str(binary),
                                  "extension_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                                  "threads": {key: os.environ[key] for key in THREAD_VARIABLES}},
                      "results": results}))


if __name__ == "__main__":
    main()
