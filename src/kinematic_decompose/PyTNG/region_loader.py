"""Stream a spatial selection from snapshot files into a pynbody container."""
from pathlib import Path

import h5py
import numpy as np
import pynbody
from pynbody.array import SimArray

from .cutout_loader import header_to_properties, infer_run_from_dir
from .illustris_python.snapshot import snapPath
from .illustris_python.util import partTypeNum
from .tng_config import (PARTICLE_DTYPE, UnitComvingLength, enforce_dtype,
                         get_particle_field_name, get_particle_field_unit)


def load_region(basePath, snapNum, center, half_size, fields, shape='box',
                periodic=True, block_size=1_000_000):
    """Scan coordinates in row blocks; retain only the selected particles.

    Non-coordinate fields are read only in blocks containing selected rows.
    No group catalog, offsets table, or particle membership is required.
    """
    basePath = str(basePath)
    with h5py.File(snapPath(basePath, snapNum), 'r') as f:
        props = header_to_properties(
            f['Header'], infer_run_from_dir(basePath, basePath=basePath), basePath, snapNum)
        num_files = int(f['Header'].attrs['NumFilesPerSnapshot'])
    paths = [snapPath(basePath, snapNum, i) for i in range(num_files)]
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing snapshot chunk: {path}')

    context = dict(a=props['a'], h=props['h'])
    def native_length(value):
        if isinstance(value, SimArray):
            value = value.in_units(UnitComvingLength, **context)
        return np.asarray(value, dtype=float)

    center, half_size = native_length(center), native_length(half_size)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('center must contain three finite coordinates')
    if shape not in ('box', 'sphere'):
        raise ValueError("shape must be 'box' or 'sphere'")
    allowed = ((), (3,)) if shape == 'box' else ((),)
    if (half_size.shape not in allowed or not np.isfinite(half_size).all()
            or np.any(half_size <= 0)):
        raise ValueError('half_size must be positive: scalar radius or scalar/3-vector box half-width')
    if not isinstance(block_size, (int, np.integer)) or block_size <= 0:
        raise ValueError('block_size must be a positive integer')
    if not fields:
        raise ValueError('Select at least one particle family')

    # Coordinates are always loaded; other fields follow the existing templates.
    requested = {
        family: list(dict.fromkeys(['Coordinates', *names]))
        for family, names in fields.items()
    }
    data = {family: {name: [] for name in names}
            for family, names in requested.items()}
    boxsize = float(props['boxsize'])
    for path in paths:
        with h5py.File(path, 'r') as f:
            for family, names in requested.items():
                group_name = f'PartType{partTypeNum(family)}'
                if group_name not in f:
                    continue
                group = f[group_name]
                for start in range(0, len(group['Coordinates']), block_size):
                    stop = start + block_size
                    pos = group['Coordinates'][start:stop]
                    delta = pos - center
                    if periodic:
                        delta -= boxsize * np.rint(delta / boxsize)
                    mask = (np.all(np.abs(delta) <= half_size, axis=1)
                            if shape == 'box' else
                            np.sum(delta**2, axis=1) <= half_size**2)
                    if not mask.any():
                        continue
                    for name in names:
                        if family == 'dm' and name == 'Masses':
                            values = np.full(mask.sum(), float(props['mDM']),
                                             dtype=PARTICLE_DTYPE)
                        else:
                            values = (pos if name == 'Coordinates' else
                                      group[name][start:stop])[mask]
                            if np.issubdtype(values.dtype, np.floating):
                                values = values.astype(PARTICLE_DTYPE, copy=False)
                        data[family][name].append(values)

    counts = {family: sum(len(v) for v in arrays['Coordinates'])
              for family, arrays in data.items()}
    container = pynbody.new(**counts, order=','.join(counts))
    # Cast only the pre-created float arrays, never integer particle IDs.
    enforce_dtype(container)
    props['region_center'] = SimArray(center.copy(), UnitComvingLength)
    props['region_half_size'] = SimArray(half_size.copy(), UnitComvingLength)
    props['region_shape'] = shape
    props['region_periodic'] = periodic
    container.properties = props
    for value in props.values():
        if isinstance(value, SimArray):
            value.sim = container

    for family, arrays in data.items():
        if not counts[family]:
            continue
        fam = getattr(container, family)
        for name, chunks in arrays.items():
            values = np.concatenate(chunks)
            fam[get_particle_field_name(name)] = SimArray(
                values, get_particle_field_unit(name))
        if family == 'dm' and 'Masses' not in arrays:
            fam['mass'] = SimArray(
                np.full(counts[family], float(props['mDM']), dtype=PARTICLE_DTYPE),
                get_particle_field_unit('Masses'))
    return container
