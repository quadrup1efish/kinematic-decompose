from types import SimpleNamespace

import numpy as np
import pytest

from kinematic_decompose.analysis import velocity_offset as vo


def _particles(vel_offset=(0.0, 0.0, 0.0)):
    shift = np.asarray(vel_offset, dtype=float)
    return {
        'star': {'pos': np.array([[1.0, 0, 0], [9.0, 0, 0]]),
                 'vel': np.array([[100.0, 0, 0], [-100.0, 0, 0]]) + shift,
                 'mass': np.array([9.8e10, 2e9])},
        'gas': {'pos': np.array([[0, 5.0, 0]]),
                'vel': np.array([[0.0, 0.0, 0.0]]) + shift,
                'mass': np.array([5e9])},
        'dm': {'pos': np.array([[0, 0, 8.0]]),
               'vel': np.array([[0.0, 50.0, 0.0]]) + shift,
               'mass': np.array([9.5e11])},
    }


def test_velocity_offset_by_hand():
    result = vo.baryonic_velocity_offset(_particles())
    baryon_mass = 9.8e10 + 2e9 + 5e9
    total_mass = baryon_mass + 9.5e11
    momentum_x = 9.8e10 * 100 - 2e9 * 100
    expected_baryon = np.array([momentum_x / baryon_mass, 0, 0])
    expected_total = np.array([momentum_x / total_mass, 9.5e11 * 50 / total_mass, 0])
    expected_v_circ = np.sqrt(vo.G_KPC_KMS2_MSOL * 9.8e10 / 1.0)
    assert result['r90'] == 1.0
    assert result['mass_baryon'] == pytest.approx(baryon_mass)
    assert result['mass_total'] == pytest.approx(total_mass)
    assert result['mass_enclosed_r90'] == pytest.approx(9.8e10)
    np.testing.assert_allclose(result['v_baryon'], expected_baryon, rtol=1e-12)
    np.testing.assert_allclose(result['v_total'], expected_total, rtol=1e-12)
    assert result['v_circ'] == pytest.approx(expected_v_circ, rel=1e-12)
    expected_s_b = np.linalg.norm(expected_baryon - expected_total) / expected_v_circ
    assert result['s_b'] == pytest.approx(expected_s_b, rel=1e-12)
    assert result['valid'] is True
    assert result['n_particles'] == {'star': 2, 'gas': 1, 'dm': 1}


def test_aperture_mass_reaches_all_particles():
    result = vo.baryonic_velocity_offset(_particles(), circular_mass='aperture')
    total_mass = 9.8e10 + 2e9 + 5e9 + 9.5e11
    assert result['mass_enclosed_r90'] == pytest.approx(total_mass)
    assert result['v_circ'] == pytest.approx(np.sqrt(vo.G_KPC_KMS2_MSOL * total_mass), rel=1e-12)


def test_common_velocity_is_irrelevant():
    base = vo.baryonic_velocity_offset(_particles())
    shifted = vo.baryonic_velocity_offset(_particles((37.0, -12.0, 4.0)))
    assert shifted['s_b'] == pytest.approx(base['s_b'], rel=1e-12)
    np.testing.assert_allclose(shifted['velocity_offset'], base['velocity_offset'], rtol=1e-12)


def test_equal_velocities_give_zero_offset():
    velocities = {'star': np.full((2, 3), 10.0), 'gas': np.full((1, 3), 10.0),
                  'dm': np.full((1, 3), 10.0)}
    particles = {name: {**block, 'vel': velocities[name]}
                 for name, block in _particles().items()}
    result = vo.baryonic_velocity_offset(particles)
    assert result['s_b'] == pytest.approx(0.0, abs=1e-12)


def test_particles_outside_aperture_are_excluded():
    result = vo.baryonic_velocity_offset(_particles(), aperture=4.0)
    assert result['n_particles'] == {'star': 1, 'gas': 0, 'dm': 0}
    assert result['mass_total'] == pytest.approx(9.8e10)


def test_missing_stars_are_reported_not_raised():
    particles = _particles()
    del particles['star']
    result = vo.baryonic_velocity_offset(particles)
    assert result['valid'] is False
    assert np.isnan(result['s_b']) and np.isnan(result['r90'])


@pytest.mark.parametrize('kwargs,match', [
    ({'aperture': 0.0}, 'aperture'),
    ({'circular_mass': 'total'}, 'circular_mass'),
])
def test_invalid_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        vo.baryonic_velocity_offset(_particles(), **kwargs)


def test_invalid_blocks():
    with pytest.raises(ValueError, match='must not be empty'):
        vo.baryonic_velocity_offset({})
    with pytest.raises(ValueError, match='star or gas'):
        vo.baryonic_velocity_offset({'dm': _particles()['dm']})
    bad = {'star': {'pos': np.zeros((2, 2)), 'vel': np.zeros((2, 3)),
                    'mass': np.ones(2)}}
    with pytest.raises(ValueError, match='shape'):
        vo.baryonic_velocity_offset(bad)
    negative = {'star': {'pos': np.zeros((1, 3)), 'vel': np.zeros((1, 3)),
                         'mass': np.array([-1.0])}}
    with pytest.raises(ValueError, match='non-negative'):
        vo.baryonic_velocity_offset(negative)
    with pytest.raises(ValueError, match='missing'):
        vo.baryonic_velocity_offset({'star': {'pos': np.zeros((1, 3)),
                                              'vel': np.zeros((1, 3))}})


def test_gravity_constant_matches_pynbody():
    from pynbody import units
    assert vo.G_KPC_KMS2_MSOL == pytest.approx(
        units.G.ratio('kpc km^2 s^-2 Msol^-1'), rel=1e-12)


def test_measure_loads_and_centres(monkeypatch):
    from kinematic_decompose.PyTNG import snapshot_loader

    calls = []

    def family(block):
        return {key: np.asarray(value, dtype=float) for key, value in block.items()}

    galaxy = SimpleNamespace(
        s=family(_particles()['star']),
        g=family(_particles()['gas']),
        dm=family(_particles()['dm']),
        bh=family({'pos': np.zeros((1, 3)), 'vel': np.zeros((1, 3)),
                   'mass': np.array([1e9])}),
    )

    class Snapshot:
        def __init__(self, base, snap):
            calls.append(('init', base, snap))
            self.container = galaxy
            self.group_catalog = {'SubhaloPos': np.zeros(3)}

        def load_particle(self, ID, load_particle_fields):
            calls.append(('load', ID, tuple(sorted(load_particle_fields))))

        def physical_units(self):
            calls.append('units')

        def load_group_catalog(self, ID):
            calls.append(('catalog', ID))

        def GC_physical_units(self):
            calls.append('catalog_units')

        def center(self, cen):
            calls.append('center')

    monkeypatch.setattr(snapshot_loader, 'Snapshot', Snapshot)
    result = vo.measure_baryon_velocity_offset('/sim', 99, 433289)
    assert calls == [('init', '/sim', 99), ('load', 433289, ('dm', 'gas', 'star')),
                     'units', ('catalog', 433289), 'catalog_units', 'center']
    assert result['s_b'] == pytest.approx(vo.baryonic_velocity_offset(_particles())['s_b'])
    assert (result['snap_num'], result['subfind_id']) == (99, 433289)
    with pytest.raises(ValueError, match='unknown species'):
        vo.measure_baryon_velocity_offset('/sim', 99, 1, species=('star', 'dust'))


def test_particles_from_container_skips_empty_families():
    empty = {'pos': np.zeros((0, 3)), 'vel': np.zeros((0, 3)), 'mass': np.zeros(0)}
    galaxy = SimpleNamespace(
        s={key: np.asarray(value, dtype=float) for key, value in _particles()['star'].items()},
        g=empty, dm=empty)
    particles = vo.particles_from_container(galaxy)
    assert set(particles) == {'star'}
    with pytest.raises(ValueError, match='unknown species'):
        vo.particles_from_container(galaxy, species=('star', 'dust'))
