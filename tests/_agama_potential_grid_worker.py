"""Isolated stock-Agama field evaluator for the grid accuracy study."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import json
import sys

import agama
import numpy as np

input_path = sys.argv[1]
rmin, rmax = float(sys.argv[2]), float(sys.argv[3])
grid_size_r = int(sys.argv[4])
with np.load(input_path) as data:
    positions = data["positions"].copy()
    masses = data["masses"].copy()
    radii = data["query_radii"].copy()
query = np.column_stack((radii, np.zeros_like(radii), np.zeros_like(radii)))
agama.setUnits(length=1, mass=1, velocity=1)
potential = agama.Potential(type="Multipole", particles=(positions, masses),
                            symmetry="s", lmax=0, mmax=0, rmin=rmin, rmax=rmax,
                            gridSizeR=grid_size_r)
phi = np.asarray(potential.potential(query), dtype=float).reshape(-1)
force = np.asarray(potential.force(query), dtype=float)
print(json.dumps({"potential": phi.tolist(), "radial_force": force[:, 0].tolist()}))
