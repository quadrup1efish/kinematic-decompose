"""Cross-binary regression for cached natural cubic spline grid operands.

Set MULTIPOLE_BASELINE_BINARY to a pre-change _potential extension to run.
Each binary is imported in its own process (pybind types must not be mixed).
"""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


WORKER = r'''
import importlib.util
import sys
from pathlib import Path
import numpy as np
binary, output = sys.argv[1:]
if binary != "current":
    name = "kinematic_decompose.potential._potential"
    spec = importlib.util.spec_from_file_location(name, binary)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
from kinematic_decompose.potential import Potential
rng = np.random.default_rng(719)
# Exceed 32 * 30 radial nodes * 25 SH slots even for symmetry='n'.
pos = rng.normal(size=(32000, 3)) * [1.7, 0.8, 0.5]
mass = rng.uniform(0.5, 1.5, len(pos))
# All three directions, origin, axis and many decades inside/outside the grid.
radii = np.geomspace(1e-8, 1e5, 100)
directions = np.array([[1., 0., 0.], [0., 0., 1.], [0.3, 0.4, np.sqrt(0.75)]])
query = np.vstack([np.zeros((1, 3)), (radii[:, None, None]*directions).reshape(-1, 3)])
result = {}
cases = [("spherical", "s", 0, .8, False),
         ("axisymmetric", "a", 4, .8, False),
         ("nonsymmetric", "n", 4, .8, False),
         ("central", "n", 4, .8, True),
         ("narrow", "a", 4, .03, False),
         ("wide", "a", 4, 4., False),
         ("unsoftened", "a", 4, None, False),
         ("variable", "a", 4, np.full(len(mass), .8), False)]
for label, sym, order, softening, central in cases:
    p = np.vstack([pos, np.zeros((1, 3))]) if central else pos
    m = np.append(mass, 31.) if central else mass
    pot = Potential(type="Multipole", particles=(p, m), softening=softening,
                    symmetry=sym, lmax=order, mmax=order,
                    gridSizeR=30, rmin=.01, rmax=20.)
    result[label+"_phi"] = pot.potential(query)
    result[label+"_force"] = pot.force(query)
    export = Path(output).with_name(label+".ini")
    pot.export(export)
    rows = []
    for line in export.read_text().splitlines():
        try:
            row = [float(x) for x in line.split()]
        except ValueError:
            continue
        if len(row) > 1:
            rows.append(row)
    result[label+"_coefs"] = np.array(rows)
np.savez(output, **result)
'''


@pytest.fixture(scope="module")
def binary_results(tmp_path_factory):
    baseline = os.environ.get("MULTIPOLE_BASELINE_BINARY")
    if not baseline:
        pytest.skip("set MULTIPOLE_BASELINE_BINARY for isolated old/new regression")
    assert Path(baseline).is_file()
    root = tmp_path_factory.mktemp("multipole_coefficients")
    outputs = []
    for label, binary in [("old", baseline), ("new", "current")]:
        directory = root / label
        directory.mkdir()
        output = directory / "values.npz"
        subprocess.run([sys.executable, "-c", WORKER, binary, str(output)], check=True)
        outputs.append(dict(np.load(output)))
    return outputs


@pytest.mark.parametrize("case", ["spherical", "axisymmetric", "nonsymmetric", "central",
                                  "narrow", "wide", "unsoftened", "variable"])
def test_cached_basis_matches_original_arithmetic(binary_results, case):
    old, new = binary_results
    for quantity in ["coefs", "phi", "force"]:
        a, b = old[case+"_"+quantity], new[case+"_"+quantity]
        assert a.size > 0 and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        # Grid-only caching must not change sample arithmetic or solver results.
        np.testing.assert_array_equal(b, a)
