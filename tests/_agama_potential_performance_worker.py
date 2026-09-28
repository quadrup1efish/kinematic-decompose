"""Isolated stock-Agama build timer used by example_potential_performance.py."""
import json
import sys
from time import perf_counter

import agama
import numpy as np

input_path, repeats = sys.argv[1], int(sys.argv[2])
rmin, rmax = float(sys.argv[3]), float(sys.argv[4])
grid_size_r = int(sys.argv[5]) if len(sys.argv) > 5 else 30
data = np.load(input_path)
positions = np.asarray(data["positions"], dtype=float)
masses = np.asarray(data["masses"], dtype=float)
query_radii = np.asarray(data["query_radii"], dtype=float)
query = np.column_stack((query_radii, np.zeros_like(query_radii), np.zeros_like(query_radii)))
agama.setUnits(length=1, mass=1, velocity=1)
timings = []
for _ in range(repeats):
    start = perf_counter()
    potential = agama.Potential(
        type="Multipole",
        particles=(positions, masses),
        symmetry="s",
        lmax=0,
        mmax=0,
        rmin=rmin,
        rmax=rmax,
        gridSizeR=grid_size_r,
    )
    timings.append(perf_counter() - start)
    del potential
potential = agama.Potential(
    type="Multipole",
    particles=(positions, masses),
    symmetry="s",
    lmax=0,
    mmax=0,
    rmin=rmin,
    rmax=rmax,
    gridSizeR=grid_size_r,
)
phi = np.asarray(potential.potential(query), dtype=float).reshape(-1)
force = np.asarray(potential.force(query), dtype=float)
print(json.dumps({"build_times_s": timings, "potential": phi.tolist(),
                  "radial_force": force[:, 0].tolist()}))
