"""Existing HDF5 schema, index updates and completion markers."""

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np


TREE_DTYPE = np.dtype([
    ('z0_subID',         'i8'),
    ('snapNum',          'i4'),
    ('progenitor_subID', 'i8'),
])

CATALOG_DTYPE = np.dtype([
    ('snapNum',            'i4'),
    ('subfindID',          'i8'),
    ('method',             'S8'),
    ('skipped',            '?'),
    ('traceback_complete', '?'),
    ('mass_star',          'f4'),
    ('krot',               'f4'),
    ('r50',                'f4'),
    ('n_particles',        'i4'),
    ('mass_bulge',         'f4'),
    ('mass_disk',          'f4'),
    ('mass_colddisk',      'f4'),
    ('mass_warmdisk',      'f4'),
    ('mass_spheroid',      'f4'),
    ('mass_halo',          'f4'),
    ('n_dm_particles','i4'),       # ← DM 粒子数，用于筛选
    ('dm_mass', 'f4'),
    ('mass_counter_rotating_disk', 'f4'),
    ('host_id', 'i8'), ('is_central', 'i1'),
    ('s_b', 'f8'), ('vmax', 'f8'), ('v200c', 'f8'),
    ('tcool_Myr', 'f8'), ('tff_Myr', 'f8'), ('tcool_tff', 'f8'),
])


def normalize_array(v):
    """强制固定 dtype：浮点 → float32，整数 → int64"""
    v = np.asarray(v)
    if np.issubdtype(v.dtype, np.floating):
        return v.astype(np.float32)
    if np.issubdtype(v.dtype, np.integer):
        return v.astype(np.int64)
    return v


def _write_dataset(group, name, value):
    value = normalize_array(value)
    if name in group:
        dataset = group[name]
        if dataset.shape == value.shape and dataset.dtype == value.dtype:
            dataset[...] = value
            return
        del group[name]
    group.create_dataset(name, data=value)


def write_snapshot_data(snap_grp, structure_dict):
    for comp, sub in structure_dict.items():
        cg = snap_grp.require_group(comp)
        for k, v in sub.items():
            _write_dataset(cg, k, v)


def write_potential(grp, pot):
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
    tmp.close()
    pot.export(tmp.name)
    with open(tmp.name, 'r') as f:
        content = f.read()
    os.unlink(tmp.name)
    if 'potential_ini' in grp.attrs:
        grp.attrs.modify('potential_ini', content)
    else:
        grp.attrs['potential_ini'] = content


def write_dict_to_group(grp, d):
    for k, v in d.items():
        if isinstance(v, dict):
            write_dict_to_group(grp.require_group(k), v)
        else:
            _write_dataset(grp, k, v)


def write_evolution_data(evo_grp, evo_dict, structures):
    for s in structures:
        sg = evo_grp.require_group(s)
        for k, v in evo_dict[s].items():
            if v is not None:
                _write_dataset(sg, k, v)


def _ensure_catalog_dataset(h5):
    if 'catalog' not in h5:
        return h5.create_dataset('catalog', shape=(0,), dtype=CATALOG_DTYPE,
                                 maxshape=(None,), chunks=True, compression='gzip')
    return h5['catalog']


def _catalog_key_index(h5, ds):
    cached = getattr(h5, '_catalog_row_index', None)
    if cached is None or cached[0] != len(ds):
        keys = list(zip(ds['snapNum'][:].tolist(), ds['subfindID'][:].tolist()))
        indices = {key: i for i, key in enumerate(keys)}
        if len(indices) != len(keys):
            raise ValueError('/catalog: duplicate keys require explicit reconciliation')
        cached = h5._catalog_row_index = [len(ds), indices]
    return cached


def _store_catalog_row(h5, ds, row):
    cached = _catalog_key_index(h5, ds)
    key = (int(row['snapNum']), int(row['subfindID']))
    idx = cached[1].get(key)
    appended = idx is None
    if idx is None:
        idx = len(ds)
        ds.resize(idx + 1, axis=0)
    try:
        ds[idx] = row
    except BaseException:
        if appended:
            ds.resize(idx, axis=0)
        raise
    if appended:
        cached[1][key] = idx
        cached[0] = len(ds)


def _catalog_structure(data_grp):
    result = {}
    names = ('star',) if data_grp.attrs.get('_method') == 'trace' else (
        'star', 'bulge', 'disk', 'colddisk', 'warmdisk', 'spheroid', 'halo', 'counter_rotating_disk')
    for name in names:
        if f'{name}/mass' in data_grp:
            result[name] = {'mass': data_grp[f'{name}/mass'][()]}
    if 'star' not in result:
        result['star'] = {'mass': data_grp.attrs.get('_stellar_mass', 0.0)}
    return result


def _cached_payload_valid(data_grp):
    if not data_grp.attrs.get('_complete', False):
        return False
    if data_grp.attrs.get('_skipped', False):
        return data_grp.attrs.get('_n_particles', -1) == 0
    if data_grp.attrs.get('_method') == 'trace':
        return 'star/mass' in data_grp and 'sfh' in data_grp
    return (data_grp.attrs.get('_method') == 'full'
            and all(f'{c}/mass' in data_grp for c in
                    ('star', 'bulge', 'disk', 'colddisk', 'warmdisk', 'spheroid', 'halo'))
            and 'gmm' in data_grp and 'potential_ini' in data_grp.attrs)


def _repair_cached_catalogs(h5, snapNum, subfindID, data_grp):
    _upsert_catalog(h5, snapNum, subfindID, data_grp, _catalog_structure(data_grp))
    h5.flush()


def _publish_progress_status(h5, snapNum, subfindID, data_grp):
    """Publish writer-owned progress; monitoring need not open a locked HDF5."""
    filename = os.environ.get('KD_CATALOG_RUN_STATUS')
    if not filename:
        return
    import json
    from datetime import datetime, timezone
    info = {'phase': 'object_committed', 'pid': os.getpid(),
            'run_id': str(h5.attrs.get('run_id', '')),
            'started_utc': str(h5.attrs.get('started_utc', '')),
            'updated_utc': datetime.now(timezone.utc).isoformat(),
            'output': str(Path(h5.filename).resolve()),
            'rebuild_potentials': True,
            'catalog_rows': len(h5['catalog']),
            'last_commit': {'snapshot': int(snapNum), 'subfindID': int(subfindID),
                            'method': str(data_grp.attrs.get('_method', '')),
                            'potential_source': str(data_grp.attrs.get('_potential_source', 'not_applicable'))}}
    path = Path(filename)
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(info, indent=2))
        os.replace(temporary, path)
    except OSError as exc:
        print(f'[STATUS_WARNING] {exc}', flush=True)


def _commit_snapshot(h5, snapNum, subfindID, data_grp, structure_dict):
    """Payload is already written; publish completion only after the summary row."""
    data_grp.attrs['_complete'] = False
    try:
        _upsert_catalog(h5, snapNum, subfindID, data_grp, structure_dict)
        h5.flush()
        data_grp.attrs['_complete'] = True
        h5.flush()
    except BaseException:
        data_grp.attrs['_complete'] = False
        h5.flush()
        raise
    _publish_progress_status(h5, snapNum, subfindID, data_grp)


def _invalidate_snapshot(data_grp):
    old_chain_complete = data_grp.attrs.get('_traceback_complete', False)
    data_grp.attrs['_complete'] = False
    data_grp.attrs['_traceback_complete'] = False
    if old_chain_complete:
        snapNum, subfindID = map(int, data_grp.name.strip('/').split('/'))
        _set_traceback_complete(data_grp.file, snapNum, subfindID, False)
    data_grp.file.flush()


def _clear_snapshot_payload(data_grp):
    for name in ('star', 'disk', 'colddisk', 'warmdisk', 'spheroid', 'bulge',
                 'halo', 'counter_rotating_disk', 'dm', 'gmm', 'sfh', 'evolution', 's_b',
                 'vmax', 'v200c', 'ecut_distribution', 'eoemin_distribution', 'inner_cgm'):
        if name in data_grp:
            del data_grp[name]
    for name in ('potential_ini', '_ecut_method', '_eoemin_cut', '_stellar_mass',
                 '_krot', '_r50', '_n_particles', '_n_dm_particles', '_dm_mass',
                 '_traceback_snap_count', '_potential_source'):
        if name in data_grp.attrs:
            del data_grp.attrs[name]


def _set_traceback_complete(h5, snapNum, subfindID, value):
    data_grp = h5.require_group(f'{snapNum}/{subfindID}')
    data_grp.attrs['_traceback_complete'] = False
    ds, index = None, None
    if 'catalog' in h5:
        ds = h5['catalog']
        index = _catalog_key_index(h5, ds)[1].get((snapNum, subfindID))
    if value and index is None:
        raise ValueError('Cannot complete a traceback without its catalog row')
    try:
        if index is not None:
            row = ds[index]
            row['traceback_complete'] = value
            ds[index] = row
        h5.flush()
        data_grp.attrs['_traceback_complete'] = value
        h5.flush()
    except BaseException:
        data_grp.attrs['_traceback_complete'] = False
        if index is not None:
            row = ds[index]
            row['traceback_complete'] = False
            ds[index] = row
        h5.flush()
        raise


CGM_COLUMNS = ('tcool_Myr', 'tff_Myr', 'tcool_tff')


def _check_catalog_schema(h5):
    """Only write files with the current catalog layout."""
    roots = {'catalog', 'tree', 'time_14', 'time_140'}
    ds = h5.get('catalog')
    if (any(key not in roots and not key.isdecimal() for key in h5)
            or (ds is not None and (not isinstance(ds, h5py.Dataset)
                                    or ds.dtype != CATALOG_DTYPE or ds.maxshape != (None,)))):
        raise ValueError('Unsupported catalog layout: use a new output path; no in-place migration')


def _upsert_catalog(h5, snapNum, subfindID, data_grp, structure_dict):
    """Update the canonical summary without duplicating keys or CGM scalars."""
    ds = _ensure_catalog_dataset(h5)
    idx = _catalog_key_index(h5, ds)[1].get((snapNum, subfindID))
    row = np.zeros(1, dtype=CATALOG_DTYPE)[0]
    for name in CATALOG_DTYPE.names:
        if CATALOG_DTYPE[name].kind == 'f':
            row[name] = np.nan
    row['snapNum'], row['subfindID'] = snapNum, subfindID
    attrs = data_grp.attrs
    for name, default in dict(method='full', skipped=False, traceback_complete=False,
                              krot=np.nan, r50=np.nan, n_particles=0, n_dm_particles=0,
                              dm_mass=np.nan, host_id=-1, is_central=-1).items():
        row[name] = attrs.get('_' + name, default)
    for component in ('star', 'bulge', 'disk', 'colddisk', 'warmdisk', 'spheroid',
                      'halo', 'counter_rotating_disk'):
        if component == 'star' or row['method'] != b'trace':
            row['mass_' + component] = structure_dict.get(component, {}).get('mass', np.nan)
    if row['method'] == b'trace':
        row['krot'] = np.nan
    for name in ('s_b', 'vmax', 'v200c'):
        if name in data_grp:
            row[name] = data_grp[name][()]
    if idx is not None and 'inner_cgm' in data_grp:
        old = ds[idx]
        for name in CGM_COLUMNS:
            row[name] = old[name]
    _store_catalog_row(h5, ds, row)
