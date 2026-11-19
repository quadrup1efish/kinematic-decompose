import numpy as np
import matplotlib.pyplot as plt
import pytest

from kinematic_decompose.analysis import stellar_birth
from kinematic_decompose.analysis.stellar_birth import (
    BirthPotential, compute_birth_circularity_history, compute_birth_distributions,
    compute_birth_spatial_history, measure_stellar_birth,
)
from kinematic_decompose.visualization.stellar_birth import (
    plot_birth_history, plot_birth_spatial_history, visualize_stellar_birth,
)


class Kepler:
    def potential(self, pos):
        return -100 / np.linalg.norm(pos, axis=1)

    def force(self, pos):
        return -100 * pos / np.linalg.norm(pos, axis=1)[:, None]**3


def test_birth_and_current_share_seed_frame_and_exact_ids(monkeypatch):
    ids = np.array([2**63 + 1, 2**63 + 2], dtype=np.uint64)
    pos = np.array([[1., 0., 0.], [1., 0., 0.]])
    vel = np.array([[0., 10., 0.], [0., -10., 0.]])
    frame = stellar_birth.TNGCircularityFrame(
        BirthPotential(Kepler(), np.geomspace(.1, 10, 100), np.eye(3)),
        ids[::-1], pos[::-1], vel[::-1])
    calls = []

    def load(snap, sub):
        calls.append((snap, sub))
        return frame

    def birth(*args, seed_axis, load_potential, main_branch_birth, **kwargs):
        np.testing.assert_array_equal(seed_axis, [0, 0, 1])
        assert load_potential(21, 78482) is frame.reference
        return measure_stellar_birth(
            dict(ParticleIDs=ids, BirthPos=pos, BirthVel=vel,
                 GFM_StellarFormationTime=[1., 1.], GFM_InitialMass=[1., 1.]),
            ids, [2, 3], branch_snaps=[20, 21], branch_scale=[.9, 1.],
            branch_position=np.zeros((2, 3)), branch_velocity=np.zeros((2, 3)),
            age_gyr=lambda a: a, hubble=1., box_size=100., seed_axis=seed_axis,
            load_potential=lambda s: load_potential(s, 78482),
            main_branch_birth=main_branch_birth, max_reference_dt_gyr=.05)

    monkeypatch.setattr(stellar_birth, 'analyze_tng_stellar_birth', birth)
    result = stellar_birth.analyze_tng_birth_and_current(
        'output', 21, 78482, ids, [2, 3], age_gyr=lambda a: a,
        main_branch_birth=[True, False], max_reference_dt_gyr=.05, load_frame=load)
    assert calls == [(21, 78482)]
    np.testing.assert_allclose(result.current.circularity, [1, -1])
    assert result.birth.circularity[0] == pytest.approx(1.)
    assert result.birth.status[1] == 'host_unconfirmed'
    with pytest.raises(ValueError, match='missing'):
        frame.measure([23])


def test_tng_frame_rotates_sources_but_returns_box_vectors(monkeypatch):
    import pynbody
    from kinematic_decompose.PyTNG import snapshot_loader
    from kinematic_decompose.gravity import kinematic_solver

    galaxy = pynbody.new(star=3)
    galaxy['pos'][:] = [[0., 1., 0.], [0., -1., 0.], [0., 1., 0.]]
    galaxy['vel'] = np.array([[0., 0., 10.], [0., 0., -10.], [0., 0., -100.]])
    galaxy['mass'] = np.ones(3)
    galaxy.s['aform'] = np.array([.3, .3, -.3])
    galaxy.s['iord'] = np.array([2**63 + 1, 2**63 + 2, 2**63 + 3], dtype=np.uint64)
    positions = np.array(galaxy['pos'])
    velocities = np.array(galaxy['vel'])

    class Snapshot:
        container = galaxy
        group_catalog = {'SubhaloPos': np.ones(3), 'SubhaloVel': np.full(3, 7.)}

        def __init__(self, *args):
            pass

        def load_particle(self, **kwargs):
            pass

        def physical_units(self):
            pass

        def load_group_catalog(self, **kwargs):
            pass

        def GC_physical_units(self):
            pass

        def center(self, *, cen, vel_cen):
            np.testing.assert_array_equal(cen, [1, 1, 1])
            np.testing.assert_array_equal(vel_cen, [7, 7, 7])

    def build(rotated):
        np.testing.assert_allclose(np.asarray(rotated['pos'])[:, 2], 0.)
        np.testing.assert_allclose(np.cross(rotated['pos'][:2], rotated['vel'][:2])[:, 2], 10.)
        return Kepler()

    monkeypatch.setattr(snapshot_loader, 'Snapshot', Snapshot)
    monkeypatch.setattr(kinematic_solver, 'construct_galaxy_potential_model', build)
    assert stellar_birth.build_tng_circularity_frame('output', 21, 78482) is None
    frame = stellar_birth.build_tng_circularity_frame('output', 21, 78482, min_spin_stars=2)
    assert frame is not None
    np.testing.assert_allclose(frame.reference.rotation[2], [1, 0, 0])
    np.testing.assert_array_equal(frame.position, positions[:2])
    np.testing.assert_array_equal(frame.velocity, velocities[:2])
    np.testing.assert_array_equal(galaxy['pos'], positions)
    np.testing.assert_allclose(frame.measure(frame.particle_ids).circularity, 1.)


def example(**overrides):
    ids = np.arange(2**63, 2**63 + 6, dtype=np.uint64)
    a = np.array([.5, .5, .5, .5, .3, .8])
    speed = np.array([10, -10, 0, 25, 10, 10])
    fields = dict(ParticleIDs=ids[::-1], GFM_StellarFormationTime=a[::-1],
                  BirthPos=np.column_stack(([1, 1, 1, 1, 99, 3], np.zeros((6, 2))))[::-1],
                  BirthVel=(np.column_stack((np.zeros(6), speed, np.zeros(6)))
                            / np.sqrt(a[:, None]))[::-1], GFM_InitialMass=np.ones(6))
    calls = []

    def load(snap):
        calls.append(snap)
        return BirthPotential(Kepler(), np.geomspace(.1, 20, 100), np.eye(3))

    options = dict(branch_snaps=[4, 21], branch_scale=[.4, .6],
                   branch_position=[[99, 0, 0], [1, 0, 0]], branch_velocity=np.zeros((2, 3)),
                   age_gyr=lambda scale: 10 * scale, hubble=.5, box_size=100., seed_axis=[0, 0, 1],
                   main_branch_birth=np.array([True, True, False, True, True, True]),
                   load_potential=load, max_reference_dt_gyr=1.)
    options.update(overrides)
    return measure_stellar_birth(fields, ids, [2, 3, 3, 2, 2, 2], **options), calls


def test_same_energy_circularity_and_periodic_center():
    birth, calls = example()
    np.testing.assert_allclose(birth.circularity[:2], [1, -1], atol=1e-10)
    np.testing.assert_allclose(birth.position[:4], np.repeat([[1., 0., 0.]], 4, axis=0))
    np.testing.assert_allclose(birth.energy[:2], [-50., -50.])
    assert birth.status.tolist() == ['ok', 'ok', 'host_unconfirmed', 'unbound',
                                     'outside_branch', 'outside_branch']
    assert np.isnan(birth.circularity[2:]).all()
    assert calls == [4]
    assert birth.reference_dt_gyr[0] == -1
    assert birth.particle_ids[0] == 2**63


def test_no_reference_or_distant_reference():
    birth, calls = example(max_reference_dt_gyr=.5)
    assert birth.status[0] == 'reference_too_far' and not calls
    birth, _ = example(load_potential=lambda snap: None)
    assert birth.status[0] == 'no_reference_frame'
    with pytest.raises(ValueError, match='confirmed main-branch'):
        example(main_branch_birth=None)


def test_birth_render(tmp_path):
    birth, _ = example()
    result = compute_birth_distributions(
        birth, time_edges=[4., 5., 6.], radius_edges=[0., 1., 2.],
        circularity_edges=[-1.2, 0., 1.2], seed_circularity=np.zeros(6))
    fig, coverage = visualize_stellar_birth(
        birth, time_edges=[4., 5., 6.], radius_edges=[0., 1., 2.],
        circularity_edges=[-1.2, 0., 1.2], seed_circularity=np.zeros(6))
    assert coverage['bulge']['radius_mass'] == 4e10
    assert coverage['halo']['circularity_mass'] == 2e10
    assert 'disk' not in coverage
    assert result.coverage == coverage
    assert result.panels[1][2][0].sum() == 2e10
    path = tmp_path / 'stellar_birth.png'
    fig.savefig(path)
    assert path.stat().st_size > 1000
    plt.close(fig)


def test_circularity_replaces_only_the_upper_map():
    pos = np.ones((5, 3))
    masses = [1., 2., 4., 8., 16.]
    time = [3., 2., 2., 1., .5]
    spatial = compute_birth_spatial_history(
        pos, pos, time, masses, seed_time=3., lookback_edges=[0., 1., 2.],
        rotation=np.eye(3), time_edges=[0., 1., 2.], radius_edges=[0., 2.],
        projection_edges=[-2., 0., 2.])
    circular = compute_birth_circularity_history(
        time, masses, [-1., 1., 0., 1.5, 0.], ['ok', 'ok', 'host_unconfirmed', 'ok', 'ok'],
        seed_time=3., time_edges=spatial.time_edges_gyr, circularity_edges=[-1., 0., 1.])
    assert (circular.window_mass, circular.valid_mass, circular.plotted_mass) == (15., 11., 3.)
    np.testing.assert_allclose(circular.sfr_per_circularity * 1e9, [[1., 0.], [0., 2.]])
    fig = plot_birth_history(spatial, top_panel='jz/jc', circularity_history=circular,
                             redshift_at_time=lambda t: 4/t-1,
                             merger_times=[2.5])
    radial = plot_birth_history(spatial, top_panel='R', circularity_history=circular)
    assert 'R_' in radial.axes[0].get_ylabel()
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'Redshift' and 'j_z/j_c' in ax.get_ylabel()
    assert any(line.get_label() == 'SubLink merger' for line in ax.lines)
    np.testing.assert_allclose(ax.collections[0].get_array().filled(0).T, circular.sfr_per_circularity)
    images = [im.get_array().filled(0) for panel in fig.axes for im in panel.images]
    old_images = [im.get_array().filled(0) for panel in radial.axes for im in panel.images]
    assert len(images) == len(old_images) == 12
    np.testing.assert_array_equal(images, old_images)
    fig.canvas.draw()
    plt.close(fig)
    plt.close(radial)
    with pytest.raises(ValueError, match='dimensionless circularity axis'):
        plot_birth_history(spatial, top_panel='jz/jc', circularity_history=circular,
                           virial_history=([1., 3.], [1., 2.]))
    with pytest.raises(ValueError, match='requires a circularity history'):
        plot_birth_history(spatial, top_panel='jz/jc')

    current = compute_birth_circularity_history(
        time, masses, [0., .2, .4, .6, .8], ['ok'] * 5, seed_time=3.,
        time_edges=spatial.time_edges_gyr, circularity_edges=[-1., 0., 1.])
    both = plot_birth_history(spatial, top_panel='jz/jc', circularity_history=circular,
                             current_circularity_history=current)
    top = [axis for axis in both.axes if axis.collections and axis.get_ylabel().startswith('$(')]
    assert [axis.get_ylabel() for axis in top] == [r'$(j_z/j_c)_{\rm birth}$',
                                                   r'$(j_z/j_c)_{\rm now}$']
    assert top[0].collections[0].norm is top[1].collections[0].norm
    np.testing.assert_allclose(top[1].collections[0].get_array().filled(0).T,
                               current.sfr_per_circularity)
    both.canvas.draw()
    plt.close(both)


def test_tng_adapter_passes_the_matching_progenitor(monkeypatch):
    a = .4
    star_id = np.array([2**63 + 1], dtype=np.uint64)
    fields = dict(ParticleIDs=star_id, GFM_StellarFormationTime=[a],
                  BirthPos=[[1.25, 0., 0.]], BirthVel=[[0., 10 / np.sqrt(a), 0.]],
                  GFM_InitialMass=[1.])
    il = stellar_birth.il
    monkeypatch.setattr(il.snapshot, 'loadSubhalo', lambda *args, **kw: fields)
    monkeypatch.setattr(il.sublink, 'loadTree', lambda *args, **kw: {
        'SnapNum': np.array([21, 4]), 'SubfindID': np.array([78482, 39850])})
    monkeypatch.setattr(il.groupcat, 'loadHeader', lambda base, snap: {
        'Time': {4: a, 21: .6}[snap], 'HubbleParam': .5, 'BoxSize': 100.})
    monkeypatch.setattr(il.groupcat, 'loadSingle', lambda *args, **kw: {
        'SubhaloPos': np.zeros(3), 'SubhaloVel': np.zeros(3)})
    seen = []

    def load(snap, subhalo_id):
        seen.append((snap, subhalo_id))
        return BirthPotential(Kepler(), np.geomspace(.1, 20, 100), np.eye(3))

    birth = stellar_birth.analyze_tng_stellar_birth(
        'output', 21, 78482, star_id, [3], seed_axis=[0, 0, 1],
        age_gyr=lambda a: 10 * a, main_branch_birth=[True],
        load_potential=load, max_reference_dt_gyr=.1)
    assert seen == [(4, 39850)]
    np.testing.assert_allclose(birth.circularity, [1.])


def test_position_only_adapter_does_not_require_velocities(monkeypatch):
    def load_subhalo(*args, fields):
        assert 'BirthVel' not in fields
        return dict(ParticleIDs=[23], GFM_StellarFormationTime=[.5],
                    BirthPos=[[1., 0., 0.]], GFM_InitialMass=[1.])

    il = stellar_birth.il
    monkeypatch.setattr(il.snapshot, 'loadSubhalo', load_subhalo)
    monkeypatch.setattr(il.sublink, 'loadTree', lambda *args, **kw: {
        'SnapNum': np.array([21, 4]), 'SubfindID': np.array([78482, 39850])})
    monkeypatch.setattr(il.groupcat, 'loadHeader', lambda base, snap: {
        'Time': {4: .4, 21: .6}[snap], 'HubbleParam': .5, 'BoxSize': 100.})
    monkeypatch.setattr(il.groupcat, 'loadSingle', lambda *args, **kw: {
        'SubhaloPos': np.zeros(3)})
    birth = stellar_birth.analyze_tng_stellar_birth(
        'output', 21, 78482, [23], [2], seed_axis=[0, 0, 1], age_gyr=lambda a: 10 * a)
    np.testing.assert_allclose(birth.position, [[1., 0., 0.]])
    assert birth.status.tolist() == ['position_only']
    assert np.isnan(birth.circularity).all()


def test_spatial_history_window_excludes_mass_but_all_ages_keeps_it(monkeypatch):
    xyz = np.array([[.25, .25, .25], [1., 1., 1.]])
    result = compute_birth_spatial_history(
        xyz, xyz, [.5, 1.5], [1., 2.], seed_time=2., lookback_edges=[0., 1.],
        rotation=np.eye(3), time_edges=[0., .3, 1.], radius_edges=[0., 1., 2.],
        projection_edges=[-2., 0., .5, 2.])
    assert result.total_mass == 3.
    assert result.window_mass == result.sfh_mass == 2.
    np.testing.assert_allclose(result.projected_mass, np.tile([3., 2.], (4, 1)))
    assert result.stellar_surface_density[0, 0, 1, 1] == 4.
    assert result.stellar_surface_density[0, 1, 2, 2] == pytest.approx(2. / 2.25)
    expected = np.array([[0., 0.], [0., 2.]])
    np.testing.assert_allclose(result.sfr_surface_density * .7 * 1e9 * np.pi * [1., 3.], expected)

    def no_rebin(*args, **kwargs):
        pytest.fail('Drawing a binned result must not rebin it')

    monkeypatch.setattr(np, 'histogram2d', no_rebin)
    fig = plot_birth_history(result)
    fig.canvas.draw()
    plt.close(fig)


@pytest.mark.parametrize('show_box_axes', [False, True])
def test_spatial_history_counts_shared_time_edges_once(show_box_axes):
    pos = np.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    fig, stats = plot_birth_spatial_history(
        pos, pos, [3., 2., 1.], [1., 2., 3.], seed_time=3.,
        lookback_edges=[0., 1., 2.], rotation=np.eye(3), half_width=3.,
        virial_history=([1., 2., 3.], [.5, 1., 2.], [True, False, True]),
        merger_times=[3., 2.25, 2.25, .5],
        redshift_at_time=lambda t: 4 / t - 1,
        map_bins=12, radial_bins=12, show_box_axes=show_box_axes)
    assert stats['time_bin_mass'] == [1., 5.]
    assert stats['sfh_mass'] == 6.
    te, re = stats['sfh_time_edges'], stats['sfh_radius_edges']
    np.testing.assert_allclose(te, np.linspace(0, 2, 421))
    np.testing.assert_allclose(re, np.linspace(0, 3, 13))
    nrows = 6 if show_box_axes else 4
    np.testing.assert_allclose(stats['projected_mass'], np.tile([6., 1., 5.], (nrows, 1)))
    ax = fig.axes[0]
    displayed = ax.collections[0].get_array().filled(0)
    expected, _, _ = np.histogram2d([0., 1., 2.], np.hypot(pos[:, 0], pos[:, 1]),
                                    bins=(te, re), weights=[1., 2., 3.])
    np.testing.assert_allclose(displayed.T * (np.diff(te)[:, None] * 1e9
                               * np.pi * np.diff(re**2)[None, :]), expected)
    assert ax.get_xlabel() == 'Redshift'
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ['0.33', '1.00', '3.00']
    assert any(text.get_text() == 'Current' for text in fig.texts)
    np.testing.assert_allclose(np.asarray(ax.lines[1].get_ydata()), [.5, 1., 2.])
    np.testing.assert_allclose([line.get_xdata()[0] for line in ax.lines[2:]], [0., .75])
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        r'FoF $R_{200c}$', 'SubLink merger']
    fig.canvas.draw()
    bounds = np.array([p.get_position().extents for p in fig.axes if p.images]).reshape(nrows, 3, 4)
    np.testing.assert_allclose(bounds[:, :-1, 2], bounds[:, 1:, 0], atol=1e-12)
    np.testing.assert_allclose(bounds[:-1, :, 1], bounds[1:, :, 3], atol=1e-12)
    plt.close(fig)
