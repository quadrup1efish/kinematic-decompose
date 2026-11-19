from types import SimpleNamespace

import numpy as np
import pytest

from kinematic_decompose.analysis import sfh_trace


@pytest.mark.parametrize('n_stars', [0, 2])
def test_trace_paths(monkeypatch, n_stars):
    calls = []

    class Stars(dict):
        r50 = 2.0
        vel_disp = 30.0

    stars = Stars(pos=np.zeros((n_stars, 3)), mass=np.array([2e9, 3e9]),
                  GFM_InitialMass=np.array([3e9, 4e9]), tform=np.array([.25, .35]))
    dm = {'pos': np.zeros((3, 3))}
    galaxy = SimpleNamespace(s=stars, dm=dm, properties={'mDM': 10.0})
    center = np.zeros(3)

    def record(name):
        def call(**kwargs):
            calls.append(name)
        return call

    def snapshot(base, snap):
        assert (base, snap) == ('mock/output', 40)
        return loaded

    loaded = SimpleNamespace(
        s=stars, dm=dm, container=galaxy, properties={'eps': .1},
        group_catalog={'SubhaloPos': center},
        load_particle=record('particles'), physical_units=record('units'),
        load_group_catalog=record('catalog'), GC_physical_units=record('catalog_units'),
        center=record('center'), faceon=record('faceon'),
    )
    monkeypatch.setattr(sfh_trace, 'Snapshot', snapshot)
    result = sfh_trace.compute_sfh_trace('mock/output', 40, 7)
    if n_stars == 0:
        assert calls == ['particles', 'units']
        assert result[0] is sfh_trace.EMPTY_STELLAR_TRACE
        assert result[1] == {'n_dm_particles': 3, 'dm_mass': 30.0}
    else:
        assert len(result) == 3
        assert calls == ['particles', 'units', 'catalog', 'catalog_units', 'center', 'faceon']
        structure, sfh, returned_snapshot = result
        assert returned_snapshot is loaded
        assert structure['star'] == {'mass': 5e9, 'r50': 2.0, 'vel_disp': 30.0,
                                     'n_particles': 2}
        assert structure['dm'] == {'mass': 30.0, 'n_particles': 3}
        assert sfh['total_mass'] == 5e9
        assert sfh['n_particles'] == 2
        assert sfh['cumsum_sf'][-1] == 7e9
        assert sfh['SFR'].shape == (138,)
