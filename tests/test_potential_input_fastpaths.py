"""Bitwise cross-binary regression for particle-construction fast paths."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

WORKER = r'''
import importlib.util, json, sys
from pathlib import Path
import numpy as np
binary, output = sys.argv[1:]
if binary != 'current':
    name = 'kinematic_decompose.potential._potential'
    spec = importlib.util.spec_from_file_location(name, binary)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
from kinematic_decompose.potential import Potential
rng = np.random.default_rng(192)
probe = np.vstack([np.zeros((1, 3)), np.geomspace(1e-8, 1e5, 32)[:, None] * [0.3, 0.4, 0.8660254037844386]])
result, errors = {}, {}
for n in (1, 8, 128, 1024, 32000):
    for sym in ('s', 'a', 'n'):
        for mode in ('scalar', 'variable', 'none', 'central_positive', 'central_zero', 'central_negative'):
            pos = rng.normal(size=(n, 6))[:, ::2]  # noncontiguous
            mass = rng.uniform(0.5, 1.5, n)
            if n <= 1024:
                radii = np.resize([.01, np.nextafter(.01, 0), 20., np.nextafter(20., 21)], n)
                pos *= (radii / np.linalg.norm(pos, axis=1))[:, None]
                pos = pos.astype(np.float32)
            if mode.startswith('central'):
                pos[0] = 0
                mass[0] = {'central_positive': 1., 'central_zero': 0., 'central_negative': -1.}[mode]
            softening = None if mode == 'none' else np.full(n, .8) if mode == 'variable' else .8
            key = f'{n}_{sym}_{mode}'
            try:
                pot = Potential(type='Multipole', particles=(pos, mass), softening=softening,
                    symmetry=sym, lmax=4, mmax=0 if sym != 'n' else 4,
                    gridSizeR=30, rmin=.01, rmax=20.)
                result[key + '_phi'] = pot.potential(probe)
                result[key + '_force'] = pot.force(probe)
            except Exception as ex:
                errors[key] = [type(ex).__name__, str(ex)]
for sym in ('s', 'a', 'n', '', 'unsupported'):
    for eps in (0., -.1, np.nan, np.inf, np.array([]), np.array([.8, .9])):
        key = f'invalid_{sym}_{repr(eps)}'
        try:
            Potential(type='Multipole', particles=(np.ones((3, 3)), np.ones(3)),
                softening=eps, symmetry=sym, lmax=4, mmax=0, gridSizeR=30)
        except Exception as ex:
            errors[key] = [type(ex).__name__, str(ex)]
result['errors_json'] = np.array(json.dumps(errors, sort_keys=True))
np.savez(output, **result)
'''


def test_particle_construction_fastpaths_bitwise(tmp_path):
    baseline = os.environ.get('MULTIPOLE_BASELINE_BINARY')
    if not baseline:
        pytest.skip('set MULTIPOLE_BASELINE_BINARY for isolated binary regression')
    assert Path(baseline).is_file()
    outputs = []
    for label, binary in [('old', baseline), ('new', 'current')]:
        output = tmp_path / (label + '.npz')
        subprocess.run([sys.executable, '-c', WORKER, binary, str(output)], check=True)
        with np.load(output) as data:
            outputs.append({key: data[key].copy() for key in data.files})
    old, new = outputs
    assert old.keys() == new.keys()
    for key in old:
        a, b = old[key], new[key]
        assert a.shape == b.shape and a.dtype == b.dtype, key
        assert a.tobytes() == b.tobytes(), key
