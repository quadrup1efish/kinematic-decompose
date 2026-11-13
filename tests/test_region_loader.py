"""Spatial snapshot selection and compatibility with pynbody transforms."""
from pathlib import Path

import h5py
import numpy as np
import pytest
from pynbody.array import SimArray

from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.PyTNG.tng_config import get_particle_field_unit


@pytest.fixture
def snapshot_files(tmp_path):
    output = tmp_path / 'TNG100-3' / 'output'
    directory = output / 'snapdir_099'
    directory.mkdir(parents=True)
    rng = np.random.default_rng(42)
    positions = np.vstack(([[0., 5., 5.], [10., 5., 5.], [9.5, 5., 5.],
                            [0.5, 5., 5.], [1., 6., 6.]],
                           rng.uniform(3, 7, size=(30, 3)), [[9.75, 5., 5.]]))
    source = {}
    for family, pt in [('gas', 0), ('dm', 1), ('star', 4), ('bh', 5)]:
        source[family] = {'Coordinates': positions.copy(),
                          'Velocities': rng.normal(size=positions.shape),
                          'ParticleIDs': np.arange(len(positions), dtype=np.uint64) +
                          np.uint64(2**55 + pt * 1000)}
        if family != 'dm':
            source[family]['Masses'] = rng.uniform(.01, .1, len(positions))
    for i, indices in enumerate(np.array_split(np.arange(len(positions)), 3)):
        with h5py.File(directory / f'snap_099.{i}.hdf5', 'w') as f:
            header = f.create_group('Header')
            header.attrs.update(dict(Time=.25, Redshift=3., HubbleParam=.6774,
                                     Omega0=.3089, OmegaLambda=.6911, BoxSize=10.,
                                     NumFilesPerSnapshot=3))
            for family, pt in [('gas', 0), ('dm', 1), ('star', 4), ('bh', 5)]:
                group = f.create_group(f'PartType{pt}')
                for field, values in source[family].items():
                    group[field] = values[indices]
    return str(output), source


def _fields(families=('gas', 'dm', 'star', 'bh')):
    return {family: ['Coordinates', 'Velocities', 'Masses', 'ParticleIDs']
            for family in families}


@pytest.mark.parametrize('shape,size', [('box', .75), ('box', [1., 1., 1.]),
                                        ('sphere', 1.)])
@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('block_size', [1, 7, 100])
def test_selection_matches_full_array(snapshot_files, shape, size, periodic, block_size):
    output, source = snapshot_files
    snapshot = Snapshot(output, 99, header_source='snapshot')
    container = snapshot.load_region([0., 5., 5.], size, _fields(),
                                     shape=shape, periodic=periodic,
                                     block_size=block_size)
    for family, raw in source.items():
        delta = raw['Coordinates'] - [0., 5., 5.]
        if periodic:
            delta -= 10 * np.rint(delta / 10)
        mask = (np.all(np.abs(delta) <= size, axis=1) if shape == 'box' else
                np.sum(delta**2, axis=1) <= size**2)
        fam = getattr(container, family)
        np.testing.assert_array_equal(fam['iord'], raw['ParticleIDs'][mask])
        assert fam['iord'].dtype == np.uint64
        for field, alias in [('Coordinates', 'pos'), ('Velocities', 'vel')]:
            np.testing.assert_array_equal(fam[alias], raw[field][mask].astype(np.float32))
            assert fam[alias].units == get_particle_field_unit(field)
            assert fam[alias].dtype == np.float32
        expected_mass = (float(snapshot.properties['mDM']) if family == 'dm'
                         else raw['Masses'][mask])
        np.testing.assert_allclose(fam['mass'], expected_mass, rtol=1e-7)
        assert fam['mass'].dtype == np.float32
    assert snapshot.ID == [] and not snapshot.group_catalog
    np.testing.assert_array_equal(container['pos'],
                                  np.concatenate([getattr(container, f)['pos']
                                                  for f in _fields()]))


def test_physical_units_center_and_unit_bearing_selection(snapshot_files):
    output, source = snapshot_files
    snapshot = Snapshot(output, 99, header_source='snapshot')
    factor = .25 / .6774
    container = snapshot.load_region(
        SimArray([0., 5., 5.], 'kpc') * factor,
        SimArray(.75 * factor, 'kpc'), _fields())
    raw_pos = np.asarray(container['pos']).copy()
    raw_vel = np.asarray(container['vel']).copy()
    raw_mass = np.asarray(container['mass']).copy()
    particle_ids = np.asarray(container['iord']).copy()
    snapshot.physical_units()
    np.testing.assert_allclose(container['pos'], raw_pos * factor, rtol=2e-7)
    np.testing.assert_allclose(container['vel'], raw_vel * np.sqrt(.25), rtol=2e-7)
    np.testing.assert_allclose(container['mass'], raw_mass * 1e10 / .6774, rtol=2e-7)
    np.testing.assert_allclose(snapshot.properties['region_center'], [0, 5 * factor, 5 * factor])
    np.testing.assert_allclose(snapshot.properties['region_half_size'], .75 * factor)
    np.testing.assert_array_equal(container['iord'], particle_ids)
    velocity_center = SimArray([.2, .1, -.3], 'km s^-1')
    snapshot.center(cen=snapshot.properties['region_center'], vel_cen=velocity_center)
    expected = raw_pos - [0, 5, 5]
    expected -= 10 * np.rint(expected / 10)
    np.testing.assert_allclose(container['pos'], expected * factor, atol=1e-6)
    np.testing.assert_allclose(container['vel'], raw_vel * .5 - velocity_center,
                               atol=2e-7)
    assert np.isfinite(container.star['j']).all()
    assert np.isfinite(float(container.properties['t']))


@pytest.mark.parametrize('mode', ['explicit', 'com', 'ssc'])
def test_center_without_catalog(snapshot_files, mode):
    output, _ = snapshot_files
    snapshot = Snapshot(output, 99, header_source='snapshot')
    snapshot.load_region([5, 5, 5], 2., _fields(('star',)))
    snapshot.physical_units()
    relative_pos = np.asarray(snapshot.star['pos'] - snapshot.star['pos'][0]).copy()
    if mode != 'explicit':
        snapshot.center(mode=mode)
        if mode == 'com':
            np.testing.assert_allclose(snapshot.star.mean_by_mass('pos'), 0., atol=2e-7)
    else:
        snapshot.center(cen=snapshot.properties['region_center'])
    np.testing.assert_allclose(snapshot.star['pos'] - snapshot.star['pos'][0],
                               relative_pos, atol=2e-7)
    np.testing.assert_allclose(snapshot.star.mean_by_mass('vel'), 0., atol=1e-7)


def test_region_closes_hdf5_handles_on_missing_field(snapshot_files, monkeypatch):
    output, _ = snapshot_files
    opened = []
    real_file = h5py.File

    def tracked_file(*args, **kwargs):
        handle = real_file(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(h5py, 'File', tracked_file)
    snapshot = Snapshot(output, 99, header_source='snapshot')
    with pytest.raises(KeyError):
        snapshot.load_region([5, 5, 5], 2., {'star': ['MissingField']})
    assert opened and all(not f.id.valid for f in opened)


def test_empty_region_and_family_absent_in_first_chunk(snapshot_files):
    output, _ = snapshot_files
    path = Path(output) / 'snapdir_099/snap_099.0.hdf5'
    with h5py.File(path, 'r+') as f:
        del f['PartType5']
    snapshot = Snapshot(output, 99, header_source='snapshot')
    container = snapshot.load_region([5, 5, 5], 2., {'bh': ['Coordinates']})
    assert len(container.bh) > 0
    container = snapshot.load_region([2, 2, 2], .01)
    assert len(container) == 0
    container.physical_units()


def test_missing_chunk_is_not_silently_ignored(snapshot_files):
    output, _ = snapshot_files
    (Path(output) / 'snapdir_099/snap_099.1.hdf5').unlink()
    snapshot = Snapshot(output, 99, header_source='snapshot')
    with pytest.raises(FileNotFoundError, match='Missing snapshot chunk'):
        snapshot.load_region([0, 5, 5], 1.)


@pytest.mark.parametrize('kwargs', [dict(center=[1, 2]), dict(half_size=-1),
                                   dict(half_size=[1, 2]), dict(shape='cylinder'),
                                   dict(shape='sphere', half_size=[1, 2, 3]),
                                   dict(block_size=0)])
def test_invalid_region_arguments(snapshot_files, kwargs):
    output, _ = snapshot_files
    snapshot = Snapshot(output, 99, header_source='snapshot')
    options = dict(center=[0, 5, 5], half_size=1.)
    options.update(kwargs)
    with pytest.raises(ValueError):
        snapshot.load_region(**options)


def test_dm_mass_without_explicit_field_and_reload(snapshot_files):
    output, _ = snapshot_files
    snapshot = Snapshot(output, 99, header_source='snapshot')
    snapshot.load_region([0, 5, 5], 1., {'dm': []})
    snapshot.physical_units()
    container = snapshot.load_region([0, 5, 5], 1., {'dm': []})
    assert container.dm['pos'].units == get_particle_field_unit('Coordinates')
    np.testing.assert_allclose(container.dm['mass'], float(snapshot.properties['mDM']))


def test_pynbody_surface_density_visualization(snapshot_files, tmp_path):
    from example_load_region import plot_surface_density, surface_density_maps
    import matplotlib.pyplot as plt

    output, _ = snapshot_files
    snapshot = Snapshot(output, 99, header_source='snapshot')
    snapshot.load_region([5, 5, 5], 5., _fields(('gas', 'dm', 'star')))
    snapshot.physical_units()
    snapshot.center(cen=snapshot.properties['region_center'], with_velocity=False)
    maps, statistics = surface_density_maps(snapshot.container, width_kpc=3., resolution=32)
    for family in ('dm', 'gas', 'star'):
        assert maps[family].shape == (32, 32)
        assert np.isfinite(maps[family]).all() and np.any(maps[family] > 0)
        assert statistics[family]['image_mass_Msol'] > 0
    np.testing.assert_allclose(statistics['star']['smooth_median_kpc'], 3. / 32)
    fig, limits = plot_surface_density(maps, width_mpc=.003)
    fig.canvas.draw()
    assert len(fig.axes) == 6  # Three maps and three horizontal colorbars.
    for ax in fig.axes[::2]:
        box = ax.get_window_extent()
        np.testing.assert_allclose(box.width, box.height, rtol=1e-6)
    assert all(lo > 0 and hi > lo for lo, hi in limits.values())
    fig.savefig(tmp_path / 'region_surface_density.png')
    plt.close(fig)


@pytest.mark.slow
def test_real_tng100_3_region_and_transforms():
    """Compare every selected ID/field against independent periodic intervals."""
    output = Path('/Users/yuwa/sims.TNG/TNG100-3/output')
    paths = [output / f'snapdir_099/snap_099.{i}.hdf5' for i in range(7)]
    if not all(path.is_file() for path in paths):
        pytest.skip('Complete local TNG100-3 snapshot is not available')
    with h5py.File(paths[0], 'r') as f:
        center = f['PartType4/Coordinates'][0]
        box = float(f['Header'].attrs['BoxSize'])
    size = np.array([1500., 1000., 1000.])
    snapshot = Snapshot(str(output), 99, header_source='snapshot')
    container = snapshot.load_region(center, size, _fields())
    expected = {family: {field: [] for field in names}
                for family, names in _fields().items()}
    for path in paths:
        with h5py.File(path, 'r') as f:
            for family, pt in [('gas', 0), ('dm', 1), ('star', 4), ('bh', 5)]:
                group = f[f'PartType{pt}']
                for start in range(0, len(group['Coordinates']), 2_000_000):
                    pos = group['Coordinates'][start:start + 2_000_000]
                    lo, hi = (center - size) % box, (center + size) % box
                    inside = np.where(lo <= hi, (pos >= lo) & (pos <= hi),
                                      (pos >= lo) | (pos <= hi))
                    mask = inside.all(axis=1)
                    if not mask.any():
                        continue
                    for field in expected[family]:
                        if family == 'dm' and field == 'Masses':
                            continue
                        values = (pos if field == 'Coordinates' else
                                  group[field][start:start + 2_000_000])[mask]
                        expected[family][field].append(values)
    for family, arrays in expected.items():
        fam = getattr(container, family)
        assert len(fam) > 0
        for field, alias in [('Coordinates', 'pos'), ('Velocities', 'vel'),
                             ('Masses', 'mass'), ('ParticleIDs', 'iord')]:
            if family == 'dm' and field == 'Masses':
                continue
            truth = np.concatenate(arrays[field])
            if field != 'ParticleIDs':
                truth = truth.astype(np.float32)
            np.testing.assert_array_equal(fam[alias], truth)
    print('Real region counts:', {f: len(getattr(container, f)) for f in expected})
    raw_pos, raw_vel = np.asarray(container['pos']).copy(), np.asarray(container['vel']).copy()
    raw_mass = np.asarray(container['mass']).copy()
    factor = snapshot.properties['a'] / snapshot.properties['h']
    snapshot.physical_units()
    np.testing.assert_allclose(container['pos'], raw_pos * factor, rtol=2e-7)
    np.testing.assert_allclose(container['mass'], raw_mass * 1e10 / snapshot.properties['h'], rtol=2e-7)
    vel_center = SimArray(np.average(container.star['vel'], axis=0,
                                    weights=container.star['mass']), 'km s^-1')
    snapshot.center(cen=snapshot.properties['region_center'], vel_cen=vel_center)
    truth = raw_pos.astype(float) - center
    truth -= box * np.rint(truth / box)
    np.testing.assert_allclose(container['pos'], truth * factor, atol=.03)
    np.testing.assert_allclose(container['vel'], raw_vel - vel_center, atol=1e-4)
    before_pos = np.asarray(container['pos']).copy()
    radius = np.linalg.norm(before_pos, axis=1)
    speed = np.linalg.norm(np.asarray(container['vel']), axis=1)
    with snapshot.faceon(range=(0, 3000), as_context=True):
        angular_momentum = np.asarray(container.star['j']).T @ np.asarray(container.star['mass'])
        assert np.linalg.norm(angular_momentum[:2]) < abs(angular_momentum[2]) * 1e-5
        np.testing.assert_allclose(np.linalg.norm(np.asarray(container['pos']), axis=1),
                                   radius, rtol=3e-6)
        np.testing.assert_allclose(np.linalg.norm(np.asarray(container['vel']), axis=1),
                                   speed, rtol=3e-6)
    np.testing.assert_allclose(container['pos'], before_pos, atol=.002)
    snapshot.sideon(range=(0, 3000))
    np.testing.assert_allclose(np.linalg.norm(np.asarray(container['pos']), axis=1),
                               radius, rtol=3e-6)
