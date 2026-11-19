import pickle

import h5py
import numpy as np
import pytest

from kinematic_decompose.analysis import catalog as public_catalog
from kinematic_decompose.analysis.catalog import _snapshot, _storage, catalog


def test_unsupported_summary_dtype_is_not_modified(tmp_path):
    path = tmp_path / 'legacy.h5'
    with h5py.File(path, 'w') as h5:
        h5['catalog'] = [1]
    with pytest.raises(ValueError, match='no in-place migration'):
        catalog.batch_compute('mock', [(99, 7)], path)
    with h5py.File(path, 'r') as h5:
        assert set(h5) == {'catalog'}
        np.testing.assert_array_equal(h5['catalog'][:], [1])


def test_public_import():
    assert public_catalog.build_catalog is catalog.build_catalog


@pytest.mark.parametrize('radius', [200., 0.])
def test_store_catalog_velocities(monkeypatch, tmp_path, radius):
    def load_single(base, snap, *, subhaloID=None, haloID=None):
        assert (base, snap) == ('mock', 21)
        if subhaloID is not None:
            assert subhaloID == 7
            return {'SubhaloVmax': 180., 'SubhaloGrNr': 42}
        assert haloID == 42
        return {'Group_M_Crit200': 80., 'Group_R_Crit200': radius}

    monkeypatch.setattr(_snapshot.groupcat, 'loadSingle', load_single)
    monkeypatch.setattr(_snapshot.groupcat, 'loadHeader', lambda *args: {'Time': .25})
    path = tmp_path / 'velocities.h5'
    with h5py.File(path, 'w') as h5:
        _snapshot._write_catalog_velocities(h5.require_group('21/7'), 'mock', 21, 7)
    with h5py.File(path, 'r+') as h5:
        grp = h5['21/7']
        assert grp['vmax'][()] == 180.
        if radius:
            mass_msun = 80e10 / .6774
            radius_pkpc = .25 * radius / .6774
            assert grp['v200c'][()] == pytest.approx(np.sqrt(4.30094e-6 * mass_msun / radius_pkpc))
        else:
            assert np.isnan(grp['v200c'][()])
        _storage._clear_snapshot_payload(grp)
        assert 'vmax' not in grp and 'v200c' not in grp


@pytest.mark.parametrize('use_legacy_cache', [False, True])
def test_traceback_writes_existing_schema(monkeypatch, tmp_path, use_legacy_cache):
    path = tmp_path / 'catalog.h5'
    buckets = ([[] for _ in range(100)], [[] for _ in range(100)])
    buckets[0][21].append(np.uint64(2**63 + 1))
    buckets[1][99].append(np.uint64(2**63 + 2))
    options = {}
    if use_legacy_cache:
        (tmp_path / '7_stellar_assembly.pkl').write_bytes(
            pickle.dumps(dict(iord_in=buckets[0], iord_ex=buckets[1])))
        options['stellar_assembly_dir'] = tmp_path

    def assembly(base, sid):
        assert not use_legacy_cache
        return buckets

    def store(h5, snap, sid, base, **options):
        assert options['iord_in'] == buckets[0] and options['iord_ex'] == buckets[1]
        assert np.asarray(options['iord_in'][21]).dtype == np.dtype('uint64')
        grp = h5.require_group(f'{snap}/{sid}')
        grp.attrs.update(_method='trace', _stellar_mass=5.0, _n_particles=2)
        structure = {'star': {'mass': 5.0}}
        _storage.write_snapshot_data(grp, structure)
        grp.create_group('sfh')
        _storage._commit_snapshot(h5, snap, sid, grp, structure)
        return grp, True, 0

    monkeypatch.setattr(catalog, 'compute_stellar_assembly', assembly)
    monkeypatch.setattr(catalog, 'load_sorted_tree', lambda *a: [(99, 7)])
    monkeypatch.setattr(catalog, 'compute_and_store', store)
    header_calls = []

    def header(base, snap):
        assert base == 'mock'
        header_calls.append(snap)
        return {'Time': snap / 99, 'Omega0': 1., 'OmegaLambda': 0., 'HubbleParam': .7}

    monkeypatch.setattr(catalog.il.groupcat, 'loadHeader', header)
    assert catalog.build_catalog('mock', path, z0_targets=[7, 7],
                                 **options) == (0, 0, 0)
    with h5py.File(path) as h5:
        assert h5['catalog'].dtype == _storage.CATALOG_DTYPE
        assert set(h5) == {'99', 'catalog', 'time_14'}
        assert h5['catalog'][0]['traceback_complete']
        assert h5['99/7'].attrs['_traceback_complete']
        assert h5['99/7/star/mass'][()] == 5
        expected = (2 / 3) * 9.77792 / .7 * (np.array(catalog.SNAPSHOTS) / 99)**1.5
        np.testing.assert_allclose(h5['time_14'][:], expected, rtol=2e-5)
        assert header_calls == list(catalog.SNAPSHOTS)
        assert h5['99/7/evolution/star/AM'].shape == (14, 3)


def test_main_branch_includes_snapshots_13_and_17(monkeypatch):
    monkeypatch.setattr(catalog.il.sublink, 'loadTree', lambda *a, **kw: {
        'SnapNum': [99, 21, 17, 13, 12], 'SubfindID': [7, 8, 9, 10, 11],
    })
    assert catalog.load_sorted_tree('mock', 7) == [(13, 10), (17, 9), (21, 8), (99, 7)]


def test_twelve_snapshot_catalog_is_not_modified(tmp_path):
    path = tmp_path / 'old_grid.h5'
    with h5py.File(path, 'w') as h5:
        h5.create_dataset('catalog', shape=(0,), maxshape=(None,), dtype=_storage.CATALOG_DTYPE)
        h5['time_12'] = np.arange(12)
    with pytest.raises(ValueError, match='use a new output path'):
        catalog.batch_compute('mock', [(13, 7)], path)
    with h5py.File(path, 'r') as h5:
        assert set(h5) == {'catalog', 'time_12'}
        assert len(h5['catalog']) == 0
        np.testing.assert_array_equal(h5['time_12'][:], np.arange(12))


def test_backfills_formed_mass_fractions_without_changing_snapshot_mass(tmp_path):
    from types import SimpleNamespace

    from kinematic_decompose.analysis.catalog import _evolution as evolution

    def stars(ids, ages, masses):
        return dict(iord=np.array(ids), tform=np.array(ages), GFM_InitialMass=np.array(masses))

    galaxy = SimpleNamespace(
        star=stars([1, 2, 3, 4, 5], [.4, .8, 1.2, .3, .6], [2., 6., 5., 20., 2.]),
        colddisk=stars([1, 3], [.4, 1.2], [2., 5.]),
        warmdisk=stars([5], [.6], [2.]),
        bulge=stars([2, 4], [.8, .3], [6., 20.]),
    )
    structures = ['star', 'bulge', 'halo', 'colddisk', 'warmdisk']
    galaxy.halo = stars([], [], [])
    evo = evolution.init_evolution_arrays(structures, 5)
    evo['colddisk']['mass'][4] = 3.
    accum = evolution.init_accumulators()
    accum['pending_formed_mass_backfill'].extend([13, 17])
    ages = {13: .6, 17: 1.}
    slots = {13: 1, 17: 2}
    iord_in = [[] for _ in range(26)]
    iord_in[13] = [1, 5]
    iord_in[17] = [2]
    iord_in[25] = [3]
    evolution._backfill_formed_mass_fractions(evo, galaxy, accum, structures, 25,
                                              iord_in, slots, ages.__getitem__)
    assert not accum['pending_formed_mass_backfill']
    # A subsequent full frame must not replace the first available labels.
    evolution._backfill_formed_mass_fractions(evo, galaxy, accum, structures, 33,
                                              iord_in, slots, ages.__getitem__)
    with h5py.File(tmp_path / 'history.h5', 'w') as h5:
        _storage.write_evolution_data(h5, evo, structures)
        np.testing.assert_allclose(h5['colddisk/formed_mass_fraction_backfill'][:],
                                   [np.nan, .5, .2, np.nan, np.nan])
        np.testing.assert_allclose(h5['warmdisk/formed_mass_fraction_backfill'][1:3], [.5, .2])
        np.testing.assert_allclose(h5['bulge/formed_mass_fraction_backfill'][1:3], [0., .6])
        np.testing.assert_array_equal(h5['star/formed_mass_fraction_source_snap'][:],
                                      [-1, 25, 25, -1, -1])
        np.testing.assert_allclose(h5['colddisk/mass'][:], [np.nan] * 4 + [3.])


def test_assembly_failure_does_not_create_catalog(monkeypatch, tmp_path):
    def fail(*args):
        raise OSError('assembly unavailable')

    monkeypatch.setattr(catalog, 'compute_stellar_assembly', fail)
    path = tmp_path / 'catalog.h5'
    with pytest.raises(OSError, match='assembly unavailable'):
        catalog.run_traceback('mock', 7, path)
    assert not path.exists()


@pytest.mark.parametrize('failure_stage', ['assembly', 'tree', 'snapshot'])
def test_batch_trace_failure_is_recorded_and_next_target_runs(monkeypatch, tmp_path,
                                                               failure_stage):
    path = tmp_path / 'catalog.h5'
    calls = []
    failing = True

    def assembly(base, sid):
        if failing and sid == 7 and failure_stage == 'assembly':
            raise FileNotFoundError('missing cache')
        return [], []

    def tree(base, sid):
        if failing and sid == 7 and failure_stage == 'tree':
            return []
        return [(21, sid + 100), (25, 207), (99, sid)] if sid == 7 else [(21, sid + 100), (99, sid)]

    def store(h5, snap, sid, base, **kw):
        calls.append((snap, sid))
        if failing and sid == 207 and failure_stage == 'snapshot':
            raise ValueError('bad particle data')
        grp = h5.require_group(f'{snap}/{sid}')
        grp.attrs.update(_method='trace', _stellar_mass=5., _n_particles=2)
        structure = {'star': {'mass': 5.}}
        _storage.write_snapshot_data(grp, structure)
        grp.require_group('sfh')
        _storage._commit_snapshot(h5, snap, sid, grp, structure)
        return grp, True, 0

    monkeypatch.setattr(catalog, '_snapshot_age_lookup', lambda _: float)
    monkeypatch.setattr(catalog, 'compute_stellar_assembly', assembly)
    monkeypatch.setattr(catalog, 'load_sorted_tree', tree)
    monkeypatch.setattr(catalog, 'compute_and_store', store)
    assert catalog.build_catalog('mock', path, z0_targets=[7, 8],
                                 snapshot_targets=[(33, 9)]) == (1, 0, 0)
    assert (99, 7) not in calls and calls[-3:] == [(21, 108), (99, 8), (33, 9)]
    with h5py.File(path, 'r') as h5:
        bad = h5['99/7'].attrs
        assert not bad['_traceback_complete']
        assert bad['_traceback_failed_snap'] == (25 if failure_stage == 'snapshot' else -1)
        assert bad['_traceback_failed_subfind_id'] == (207 if failure_stage == 'snapshot' else -1)
        assert str(bad['_traceback_error']).startswith(failure_stage + ':')
        if failure_stage == 'snapshot':
            assert h5['21/107'].attrs['_complete']
        assert h5['99/8'].attrs['_traceback_complete']
        assert h5['33/9'].attrs['_complete']

    failing = False
    assert catalog.build_catalog('mock', path, z0_targets=[7]) == (0, 0, 0)
    with h5py.File(path, 'r') as h5:
        assert h5['99/7'].attrs['_traceback_complete']
        assert '_traceback_error' not in h5['99/7'].attrs


def test_batch_trace_does_not_swallow_output_error(monkeypatch, tmp_path):
    path = tmp_path / 'catalog.h5'
    monkeypatch.setattr(catalog, 'compute_stellar_assembly', lambda *args: ([], []))
    monkeypatch.setattr(catalog, 'load_sorted_tree', lambda base, sid: [(99, sid)])

    def fail(*args, **kwargs):
        raise OSError('write failed')

    monkeypatch.setattr(catalog, 'compute_and_store', fail)
    with pytest.raises(OSError, match='write failed'):
        catalog.build_catalog('mock', path, z0_targets=[7, 8])
    with h5py.File(path, 'r') as h5:
        assert not h5['99/7'].attrs['_traceback_complete']
        assert '99/8' not in h5


def test_batch_routes_full_and_trace(monkeypatch, tmp_path):
    from types import SimpleNamespace

    calls = []

    class Snapshot:
        def __init__(self, base, snap):
            self.s = SimpleNamespace()

        def load_particle(self, ID, **kwargs):
            self.s = {'pos': range(1000 if ID == 7 else 999)}

    def branch(name):
        def process(*args, **kwargs):
            calls.append(name)
        return process

    monkeypatch.setattr(_snapshot, 'Snapshot', Snapshot)
    monkeypatch.setattr(_snapshot, '_full_decompose', branch('full'))
    monkeypatch.setattr(_snapshot, '_trace_sfh', branch('trace'))
    assert catalog.batch_compute('mock', [(13, 7), (17, 8), (12, 7)],
                                 tmp_path / 'batch.h5') == (2, 0, 0)
    assert calls == ['full', 'trace']
