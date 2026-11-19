from types import SimpleNamespace

import h5py
import numpy as np

from kinematic_decompose.analysis.catalog import _evolution as ev, _snapshot as ss


def test_assembly_windows_cover_full_trace_and_final_frame(monkeypatch, tmp_path):
    names = ('star', 'bulge', 'halo', 'colddisk', 'warmdisk')
    evo = ev.init_evolution_arrays(names, ev.N_SNAPS_TOTAL)
    accum = ev.init_accumulators()
    ins, ex = ([[] for _ in range(100)] for _ in range(2))
    for k in (0, 21, 23, 25, 98, 99):
        ins[k] = [k + 100]
        ex[k] = [k + 200]
    all_i, all_e = [100, 121, 123, 125, 198, 199], [200, 221, 223, 225, 298, 299]

    class Stars(dict):
        r50 = vel_disp = 1.
        AM = np.zeros(3)

    def stars(ids):
        return Stars(iord=np.array(ids, dtype=int), mass=np.ones(len(ids)),
                     GFM_InitialMass=np.ones(len(ids)),
                     tform=np.where(np.asarray(ids) == 199, np.float32(13.802719), 1.),
                     jzojc=np.ones(len(ids)))

    def mock_snapshot(snap):
        sp = stars([i for i in all_e if i - 200 <= snap])
        disk = stars([i for i in all_i if i - 100 <= snap])
        total = stars(np.r_[disk['iord'], sp['iord']])
        galaxy = SimpleNamespace(star=total, s=total, disk=disk, spheroid=sp,
                                 colddisk=disk, warmdisk=stars([]), bulge=sp, halo=stars([]))
        return SimpleNamespace(container=galaxy, s=total, properties={'eps': 0.1},
                               faceon=lambda **kw: None)

    for key in ('_invalidate_snapshot', '_clear_snapshot_payload', '_write_catalog_velocities',
                'write_snapshot_data', '_commit_snapshot', 'write_dict_to_group'):
        monkeypatch.setattr(ss, key, lambda *args: None)
    monkeypatch.setattr(ss, '_measure_s_b', lambda _: 0.)
    monkeypatch.setattr(ss, 'compute_sfh_trace', lambda *a, **kw:
                        ({'star': {'mass': 8., 'r50': 1.}}, {}, mock_snapshot(25)))
    monkeypatch.setattr(ev, '_write_snap_params_to_evo', lambda *args: None)
    pairs = [(21, 1), (25, 2), (99, 3)]
    slots = {s: ev.SNAPSHOTS.index(s) for s, _ in pairs}
    with h5py.File(tmp_path / 'trace.h5', 'w') as h5:
        last_full = 0
        for j, (snap, sub) in enumerate(pairs):
            group = h5.create_group(f'{snap}/{sub}')
            if snap == 25:
                ss._trace_sfh(h5, group, group.name, snap, sub, '', 8, False, True,
                              evo, accum, names, j, pairs, ins, ex, slots, last_full)
            else:
                obj = mock_snapshot(snap)
                ev._run_evolution_step(h5, group, obj, obj.container, {}, evo, accum,
                                       names, j, pairs, ins, ex, slots, snap, last_full,
                                       snapnum_to_age=lambda s: 13.802718326812728 if s == 99 else 0.)
                last_full = snap
        edges = h5['time_140'][:]
        np.testing.assert_array_equal(edges[:-1], np.linspace(0, 13.8, 138)[:-1])
        assert edges[-1] >= np.float32(13.802719)
        assert h5['99/3/evolution/star/SFR'][-1] > 0
        for (snap, sub), expected in zip(pairs, (2, 4, 6)):
            g = h5[f'{snap}/{sub}/evolution']
            np.testing.assert_allclose(np.dot(g['star/SFR'][1:], np.diff(edges)) * 1e9, expected, rtol=1e-6)
            assert g['star/cumsum_sf'][slots[snap]] == expected
            assert g['star/cumsum_ex_situ'][slots[snap]] == expected
        g = h5['99/3/evolution']
        np.testing.assert_allclose(np.dot(g['colddisk/SFR'][1:], np.diff(edges)) * 1e9, 6, rtol=1e-6)
        assert g['colddisk/cumsum_sf'][slots[25]] == 4
        assert g['bulge/cumsum_ex_situ'][slots[99]] == 6
    assert sorted(accum['sf_all']) == all_i
    assert sorted(accum['ex_all']) == all_e
