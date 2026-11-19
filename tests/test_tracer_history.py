import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pynbody
import pytest
from runpy import run_path

from kinematic_decompose import config

from kinematic_decompose.PyTNG.tng_config import enforce_dtype
from kinematic_decompose.PyTNG.tracer import seed_decomposed_galaxy, trace_snapshot
from kinematic_decompose.visualization.tracer import relative_positions, visualize_tracer_history


LARGE_ID = 2**63 + 173


def _snapshot(root, snap, shards):
    folder = root / f'snapdir_{snap:03d}'
    folder.mkdir()
    totals = sum(np.array([len(shard.get(t, ())) for t in range(6)]) for shard in shards)
    for i, shard in enumerate(shards):
        with h5py.File(folder / f'snap_{snap:03d}.{i}.hdf5', 'w') as f:
            attrs = f.create_group('Header').attrs
            attrs.update({'NumFilesPerSnapshot': len(shards), 'NumPart_Total': totals,
                          'NumPart_ThisFile': [len(shard.get(t, ())) for t in range(6)],
                          'Redshift': 1 if snap == 50 else 2, 'Time': .5 if snap == 50 else 1/3,
                          'HubbleParam': .5, 'BoxSize': 100.})
            for p, records in shard.items():
                g = f.create_group(f'PartType{p}')
                if p == 3:
                    g['TracerID'] = np.array([r[0] for r in records], 'u8')
                    g['ParentID'] = np.array([r[1] for r in records], 'u8')
                else:
                    g['ParticleIDs'] = np.array([r[0] for r in records], 'u8')
                    g['Coordinates'] = np.asarray([r[1] for r in records])
                    if p == 4:
                        g['GFM_StellarFormationTime'] = [r[2] for r in records]


@pytest.fixture
def history(tmp_path):
    _snapshot(tmp_path, 50, [
        {3: [(42, LARGE_ID), (0, 5), (7, 8), (8, 8), (9, 5)],
         4: [(LARGE_ID, [1, 0, 0], .4), (5, [98, 5, 0], .3)]},
        {3: [(4, LARGE_ID), (6, LARGE_ID)],
         4: [(8, [2, 0, 0], -.1), (101, [0, 0, 0], .4)]},
    ])
    _snapshot(tmp_path, 37, [
        {3: [(6, 31), (0, 0), (42, 77), (4, 0), (9, 30)],
         0: [(0, [99, 3, 0])], 4: [(30, [1, 4, 0], -.3)]},
        {3: [(8, 99)], 5: [(31, [1, 1, 0])], 4: [(77, [2, 4, 0], .2)]},
    ])
    galaxy = pynbody.new(star=4)
    galaxy.s['iord'] = np.array([8, 101, LARGE_ID, 5], 'u8')
    galaxy.s['aform'] = np.array([-.1, .4, .4, .3])
    galaxy.s['label'] = np.array([3, 3, 2, 1])
    enforce_dtype(galaxy)
    assert galaxy.s['iord'].dtype == np.dtype('u8')
    assert galaxy.s['iord'][2] == LARGE_ID
    return tmp_path, galaxy


def test_frozen_cohort_tracks_parent_phases(history):
    base, galaxy = history
    cohort = seed_decomposed_galaxy(base, 50, 17, galaxy, block_size=2)
    np.testing.assert_array_equal(cohort.tracer_ids, [0, 4, 6, 9, 42])
    np.testing.assert_array_equal(cohort.labels, [1, 2, 2, 1, 2])
    assert cohort.coverage('bulge') == (1, 1, 3)
    assert cohort.coverage('disk') == (1, 1, 2)
    assert cohort.coverage('halo') == (1, 0, 0)
    earlier = trace_snapshot(base, cohort, 37, block_size=2)
    np.testing.assert_array_equal(earlier.parent_ids, [0, 0, 31, 30, 77])
    assert earlier.parent_found.all()
    np.testing.assert_array_equal(earlier.phase, ['gas', 'gas', 'black hole', 'wind', 'star'])
    np.testing.assert_array_equal(earlier.coordinates[:2], [[99, 3, 0]] * 2)
    np.testing.assert_allclose(relative_positions(earlier, [1, 0, 0])[0], [-4/3, 2, 0])
    seed = trace_snapshot(base, cohort, 50)
    assert seed.parent_found.all() and np.all(seed.phase == 'star')
    fig = visualize_tracer_history(cohort, [earlier, seed], {37: [1, 0, 0], 50: [1, 0, 0]},
                                   half_width=10)
    assert [ax.get_title() for ax in fig.axes[:2]] == ['$z=2.00$', '$z=1.00$']
    assert len(fig.axes) == 6
    fig.savefig(base / 'history.png')
    assert (base / 'history.png').stat().st_size > 0
    plt.close(fig)


def test_missing_tracer_and_missing_parent_are_distinct(history):
    base, galaxy = history
    _snapshot(base, 23, [{3: [(0, 0), (4, 100)], 0: [(0, [9, 1, 0])]}])
    cohort = seed_decomposed_galaxy(base, 50, 17, galaxy)
    earlier = trace_snapshot(base, cohort, 23)
    np.testing.assert_array_equal(earlier.tracer_found, [True, True, False, False, False])
    np.testing.assert_array_equal(earlier.parent_found, [True, False, False, False, False])
    assert np.isnan(earlier.coordinates[1:]).all()
    assert earlier.parent_ids[0] == 0 and earlier.phase[0] == 'gas'
    with pytest.raises(ValueError, match='later than the seed'):
        trace_snapshot(base, cohort, 51)


def test_incomplete_snapshot_fails_instead_of_showing_partial_history(history):
    base, galaxy = history
    cohort = seed_decomposed_galaxy(base, 50, 17, galaxy)
    (base / 'snapdir_037/snap_037.1.hdf5').unlink()
    with pytest.raises(FileNotFoundError, match='1/2 chunks missing'):
        trace_snapshot(base, cohort, 37)


def test_duplicate_matched_tracer_is_rejected(history):
    base, galaxy = history
    _snapshot(base, 23, [{3: [(0, 5), (0, 6)]}])
    cohort = seed_decomposed_galaxy(base, 50, 17, galaxy)
    with pytest.raises(ValueError, match='Duplicate matched TracerID'):
        trace_snapshot(base, cohort, 23, block_size=1)


def test_example_caches_and_redraws_without_snapshot_access(history, monkeypatch):
    base, _ = history
    output = base / 'TestRun' / 'output'
    output.mkdir(parents=True)
    for folder in base.glob('snapdir_*'):
        folder.rename(output / folder.name)
    monkeypatch.setattr(config, 'BASEPATH', str(base))
    labels = base / 'seed-labels.npz'
    np.savez(labels, run='TestRun', seed_snap=50, subhalo_id=17,
             star_ids=np.array([101, LARGE_ID, 5], 'u8'), star_labels=[3, 2, 1])
    run = run_path('tests/example_tracer_history.py')['run']
    args = argparse.Namespace(plot_only=False, run='TestRun', subhalo=17, snapshots=[37, 50],
                              seed=50, centers={37: [1, 0, 0], 50: [1, 0, 0]},
                              seed_particles=labels, structures=['bulge', 'disk'], half_width=10,
                              cache=base / 'cache.npz', figure=base / 'history-example.png',
                              comoving=False)
    run(args)
    output.rename(base / 'offline')
    args.plot_only = True
    args.figure = base / 'redrawn.png'
    run(args)
    assert args.figure.stat().st_size > 0
    plt.close('all')
