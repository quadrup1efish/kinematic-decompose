"""Route a single target through full decomposition or lightweight SFH."""

import gc

import numpy as np

from ...PyTNG.illustris_python import groupcat
from ...PyTNG.snapshot_loader import Snapshot
from ...mixture import util
from ...pipeline import decompose_prepared_galaxy
from ..inner_cgm import measure_inner_cgm
from ..sfh import get_sfr
from ..sfh_trace import EMPTY_STELLAR_TRACE, compute_sfh_trace
from ..velocity_offset import G_KPC_KMS2_MSOL, baryonic_velocity_offset, particles_from_container
from ._decomposition import save_structure_properties, train_auto_gaussian_mixture_model
from ._evolution import _find_prev_snap_in_pairs, _run_evolution_step, flatten_iord
from ._storage import (
    CGM_COLUMNS, _catalog_key_index,
    _cached_payload_valid, _clear_snapshot_payload, _commit_snapshot,
    _invalidate_snapshot, _repair_cached_catalogs, write_dict_to_group,
    write_evolution_data, write_potential, write_snapshot_data,
)

N_DECOMPOSE = 1000


def _store_inner_cgm(data_grp, base_path, snap, subfind_id, cooling_table_dir, *, overwrite=False):
    """Store the two timescales and their ratio independently of decomposition."""
    if cooling_table_dir is None:
        return
    ds = data_grp.file['catalog']
    idx = _catalog_key_index(data_grp.file, ds)[1][snap, subfind_id]
    row = ds[idx]
    if ('inner_cgm' in data_grp and not overwrite
            and all(np.isfinite(row[name]) for name in CGM_COLUMNS)):
        return
    result = measure_inner_cgm(base_path, snap, subfind_id, cooling_table_dir=cooling_table_dir)
    if 'inner_cgm' in data_grp:
        del data_grp['inner_cgm']
    grp = data_grp.create_group('inner_cgm')
    grp.attrs.update(selection='subhalo_members', geometry='host_GroupPos_R200c',
                     radius_fraction=0.1, density_capped=result.density_capped,
                     enclosed_mass_fraction=result.enclosed_mass_fraction,
                     is_central=result.is_central)
    for name, value in zip(CGM_COLUMNS, (result.tcool_Myr, result.tff_Myr, result.ratio)):
        row[name] = value
    ds[idx] = row
    data_grp.file.flush()


def _write_catalog_velocities(data_grp, base_path, snap, subfind_id):
    """Store subhalo Vmax and host FoF V200c in km/s."""
    subhalo = groupcat.loadSingle(base_path, snap, subhaloID=subfind_id)
    vmax = float(subhalo['SubhaloVmax'])
    v200c = np.nan
    host_id = int(subhalo['SubhaloGrNr'])
    data_grp.attrs['_host_id'] = host_id
    if host_id >= 0:
        halo = groupcat.loadSingle(base_path, snap, haloID=host_id)
        data_grp.attrs['_is_central'] = int(halo['GroupFirstSub'] == subfind_id) if 'GroupFirstSub' in halo else -1
        mass, radius = map(float, (halo['Group_M_Crit200'], halo['Group_R_Crit200']))
        a = float(groupcat.loadHeader(base_path, snap)['Time'])
        if np.all(np.isfinite([mass, radius, a])) and min(mass, radius, a) > 0:
            # The h factors in mass and physical radius cancel.
            v200c = np.sqrt(G_KPC_KMS2_MSOL * 1e10 * mass / (a * radius))
    data_grp['vmax'] = vmax if np.isfinite(vmax) and vmax >= 0 else np.nan
    data_grp['v200c'] = v200c


def _measure_s_b(galaxy):
    particles = particles_from_container(galaxy)
    if 'star' not in particles:
        return np.nan
    result = baryonic_velocity_offset(particles)
    return result['s_b'] if result['valid'] and result['r90'] > 0 else np.nan


def compute_and_store(h5, snapNum, subfindID, basePath, *, overwrite=False,
                      snapnum_to_age=None,
                      track_evolution=False,
                      evo=None, accum=None, evo_structures=None,
                      i_pair=None, pairs=None,
                      iord_in=None, iord_ex=None, snap_to_idx=None,
                      last_full_decomp_snap=0):
    """
    两级处理核心函数。

    n >= N_DECOMPOSE  → 完整运动学分解 (_full_decompose)
    n <  N_DECOMPOSE  → trace SFH          (_trace_sfh)

    返回 (data_grp, is_new, last_full_decomp_snap)
    """
    data_key = f"{snapNum}/{subfindID}"
    data_grp = h5.require_group(data_key)

    # 已完成检查
    if not track_evolution and not overwrite and _cached_payload_valid(data_grp):
        _repair_cached_catalogs(h5, snapNum, subfindID, data_grp)
        method = data_grp.attrs.get('_method', 'full')
        print(f"  ✓ {data_key} (cached, method={method})")
        return data_grp, False, last_full_decomp_snap

    _invalidate_snapshot(data_grp)

    # 快速粒子数检查
    snap_check = Snapshot(basePath, snapNum)
    snap_check.load_particle(ID=subfindID, load_particle_fields='potential')
    n_parts = len(snap_check.s['pos'])
    del snap_check; gc.collect()
    print(f"  {data_key}: n_particles={n_parts}")

    if n_parts >= N_DECOMPOSE:
        print(f"  [FULL] {data_key}")
        return _full_decompose(
            h5, data_grp, data_key, snapNum, subfindID,
            basePath, overwrite,
            track_evolution, evo, accum, evo_structures,
            i_pair, pairs, iord_in, iord_ex, snap_to_idx,
            last_full_decomp_snap=last_full_decomp_snap,
            snapnum_to_age=snapnum_to_age)
    else:
        print(f"  [TRACE] {data_key} (n={n_parts} < {N_DECOMPOSE})")
        return _trace_sfh(
            h5, data_grp, data_key, snapNum, subfindID,
            basePath, n_parts, overwrite,
            track_evolution, evo, accum, evo_structures,
            i_pair, pairs, iord_in, iord_ex, snap_to_idx,
            last_full_decomp_snap=last_full_decomp_snap)


def _full_decompose(h5, data_grp, data_key, snapNum, subfindID,
                    basePath, overwrite,
                    track_evolution, evo, accum, evo_structures,
                    i_pair, pairs, iord_in, iord_ex, snap_to_idx,
                    last_full_decomp_snap=0, *, snapnum_to_age=None):
    """完整运动学分解 + evolution 写入。
    返回 (data_grp, is_new, new_last_full_decomp_snap)
    """

    _invalidate_snapshot(data_grp)

    # ── 加载粒子 ──
    snap = Snapshot(basePath, snapNum)
    snap.load_particle(ID=subfindID, load_particle_fields={
        "star": ['Coordinates', 'Velocities', 'Masses',
                 'ParticleIDs', 'GFM_StellarFormationTime',
                 'GFM_InitialMass'],
        "gas": ['Coordinates', 'Velocities', 'Masses'],
        "dm": ['Coordinates', 'Velocities']})
    n_parts = len(snap.s['pos'])
    snap.physical_units()
    snap.load_group_catalog(ID=subfindID)
    snap.GC_physical_units()
    snap.center(cen=snap.group_catalog['SubhaloPos'])
    # 二次检查：加载完整粒子后可能发现不足
    if n_parts < N_DECOMPOSE:
        #del snap, pot; gc.collect()
        print(f"  → Fallback to trace (n={n_parts})")
        return _trace_sfh(h5, data_grp, data_key, snapNum, subfindID,
                          basePath, n_parts, overwrite,
                          track_evolution, evo, accum, evo_structures,
                          i_pair, pairs, iord_in, iord_ex, snap_to_idx,
                          last_full_decomp_snap=last_full_decomp_snap)

    # ── 分解 ──
    with snap.faceon(align_with='star',
                     range=[3 * snap.properties['eps'], 5 * snap.s.r50],
                     as_context=True):
        galaxy = snap.container
        print(f"  [POT] Building NEW stars+gas+DM potential: {data_key}", flush=True)
        s_b = _measure_s_b(galaxy)
        eoemin_distribution = {}
        X, model, galaxy, eoemin_cut, jzojc_cut, pot = decompose_prepared_galaxy(
            galaxy, trainer=lambda prepared, potential: train_auto_gaussian_mixture_model(
                prepared, potential, eoemin_distribution=eoemin_distribution))
        structure_dict = save_structure_properties(galaxy)
        gmm_dict = util.decompose_mixture_model(model, eoemin_cut,
                                                jzojc_cut, -jzojc_cut)
    print('  [FULL] Finish training')

    # ── 写入 H5 ──
    _clear_snapshot_payload(data_grp)
    data_grp.attrs['_potential_source'] = 'rebuilt_star_gas_dm'
    data_grp['s_b'] = s_b
    _write_catalog_velocities(data_grp, basePath, snapNum, subfindID)
    data_grp.attrs['_ecut_method'] = eoemin_distribution.get('ecut_method', 'two_component_skew_t')
    data_grp.attrs['_eoemin_cut'] = float(eoemin_cut)
    write_snapshot_data(data_grp, structure_dict)
    write_potential(data_grp, pot)
    if 'fit_density' in eoemin_distribution:
        dist = data_grp.create_group('eoemin_distribution')
        dist['fit_density'] = eoemin_distribution['fit_density']
        dist.attrs.update(first_center=eoemin_distribution['first_center'],
                          bin_width=eoemin_distribution['bin_width'])
    if gmm_dict:
        write_dict_to_group(data_grp.require_group('gmm'), gmm_dict)

    data_grp.attrs['_complete'] = False
    data_grp.attrs['_skipped'] = False
    data_grp.attrs['_method'] = 'full'
    data_grp.attrs['_stellar_mass'] = float(structure_dict['star']['mass'])
    data_grp.attrs['_krot'] = float(structure_dict['star']['krot'])
    data_grp.attrs['_r50'] = float(structure_dict['star']['r50'])
    data_grp.attrs['_n_particles'] = n_parts
    data_grp.attrs['_n_dm_particles'] = len(snap.dm['pos'])
    data_grp.attrs['_dm_mass'] = float(galaxy.properties['mDM'] * len(snap.dm['pos']))


    # ── Evolution ──
    if track_evolution:
        _run_evolution_step(
            h5, data_grp, snap, galaxy, structure_dict,
            evo, accum, evo_structures, i_pair, pairs,
            iord_in, iord_ex, snap_to_idx, snapNum,
            last_full_decomp_snap=last_full_decomp_snap, snapnum_to_age=snapnum_to_age)
    _commit_snapshot(h5, snapNum, subfindID, data_grp, structure_dict)
    del snap, galaxy, X, model, eoemin_cut, jzojc_cut, structure_dict, gmm_dict, pot
    gc.collect()
    print(f"    {data_key} done")

    # ★ 做了完整分解 → 更新 last_full_decomp_snap
    return data_grp, True, snapNum


def _trace_sfh(h5, data_grp, data_key, snapNum, subfindID,
               basePath, n_parts, overwrite,
               track_evolution, evo, accum, evo_structures,
               i_pair, pairs, iord_in, iord_ex, snap_to_idx,
               last_full_decomp_snap=0):
    """Trace SFH + evolution（仅贡献 star 级别，不分解组件）。
    返回 (data_grp, is_new, last_full_decomp_snap) — 不改变 last_full_decomp_snap。
    """
    _invalidate_snapshot(data_grp)

    result = compute_sfh_trace(basePath, snapNum, subfindID, include_gas=True)

    # 空星系
    if result[0] is EMPTY_STELLAR_TRACE:
        _clear_snapshot_payload(data_grp)
        dm_info = result[1]
        data_grp['s_b'] = np.nan
        _write_catalog_velocities(data_grp, basePath, snapNum, subfindID)
        if not isinstance(dm_info, dict):
            raise ValueError('Empty stellar result must include measured DM information')
        data_grp.attrs['_skipped'] = True
        data_grp.attrs['_n_particles'] = 0
        data_grp.attrs['_method'] = 'trace'
        data_grp.attrs['_stellar_mass'] = 0.0
        data_grp.attrs['_r50'] = np.nan
        data_grp.attrs['_krot'] = np.nan
        data_grp.attrs['_n_dm_particles'] = dm_info['n_dm_particles']
        data_grp.attrs['_dm_mass'] = dm_info['dm_mass']
        _commit_snapshot(h5, snapNum, subfindID, data_grp, {'star': {'mass': 0}})
        return data_grp, True, last_full_decomp_snap

    structure_dict_light, sfh_dict, snap = result
    s_b = _measure_s_b(snap.container)

    # 写入
    _clear_snapshot_payload(data_grp)
    write_dict_to_group(data_grp.require_group('sfh'), sfh_dict)
    write_snapshot_data(data_grp, structure_dict_light)
    data_grp['s_b'] = s_b
    _write_catalog_velocities(data_grp, basePath, snapNum, subfindID)

    data_grp.attrs['_complete'] = False
    data_grp.attrs['_skipped'] = False
    data_grp.attrs['_method'] = 'trace'
    data_grp.attrs['_stellar_mass'] = float(structure_dict_light['star']['mass'])
    data_grp.attrs['_krot'] = np.nan
    data_grp.attrs['_r50'] = float(structure_dict_light['star']['r50'])
    data_grp.attrs['_n_particles'] = n_parts
    data_grp.attrs['_n_dm_particles'] = structure_dict_light.get('dm', {}).get('n_particles', 0)
    data_grp.attrs['_dm_mass'] = structure_dict_light.get('dm', {}).get('mass', 0.0)


    # ── Evolution（trace 只能贡献 star 级别）──
    if track_evolution and evo is not None:
        galaxy = snap.container
        idx = snap_to_idx[snapNum]

        evo['star']['mass'][idx] = galaxy.s['mass'].sum()
        evo['star']['r50'][idx] = galaxy.s.r50
        evo['star']['vel_disp'][idx] = galaxy.s.vel_disp

        if iord_in is not None and iord_ex is not None:
            last_snap = _find_prev_snap_in_pairs(pairs, i_pair) if pairs is not None else 0

            # SFR: 增量区间 → 累加（后续 full 模式不再重复算这部分）
            flat_in_incr = flatten_iord(iord_in, last_snap, snapNum)
            # cumsum_sf: 从时间 0 累计 → 赋值（不是增量！）
            flat_in_cum = flatten_iord(iord_in, 0, snapNum)
            flat_ex_cum = flatten_iord(iord_ex, 0, snapNum)
            flat_ex = flatten_iord(iord_ex, last_snap, snapNum)
            # Preserve trace-period accretion for later full decompositions.
            if accum is not None:
                accum['ex_all'].extend(flat_ex.tolist())

            sid = galaxy.s['iord']
            evo['star']['cumsum_sf'][idx] = galaxy.s['GFM_InitialMass'][
                np.isin(sid, flat_in_cum)].sum()
            evo['star']['cumsum_ex_situ'][idx] = galaxy.s['mass'][
                np.isin(sid, flat_ex_cum)].sum()

            _, sfr_s = get_sfr(
                galaxy.s['GFM_InitialMass'][np.isin(sid, flat_in_incr)],
                galaxy.s['tform'][np.isin(sid, flat_in_incr)])
            evo['star']['SFR'] = sfr_s if evo['star']['SFR'] is None \
                else evo['star']['SFR'] + sfr_s

        evo_grp = data_grp.require_group('evolution')
        write_evolution_data(evo_grp, evo, evo_structures or ['star'])
    _commit_snapshot(h5, snapNum, subfindID, data_grp, structure_dict_light)
    del snap
    gc.collect()
    return data_grp, True, last_full_decomp_snap
