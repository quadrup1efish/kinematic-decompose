"""Snapshot indexing and in-memory evolution accumulation."""

import itertools

import numpy as np

from ..sfh import get_sfr
from ._storage import write_evolution_data


_SNAP_PARAMS = ['krot', 'beta', 'r50', 'R50', 'z50', 't50',
                'v_circ', 'v_rot', 'vel_disp', 'Mdyn', 'shape']
SNAPSHOTS = (13, 17, 21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99)
N_SNAPS_TOTAL = len(SNAPSHOTS)


def flatten_iord(iord_list, start, end):
    """将 iord_list[start:end] 展平为一个 int64 数组。
    iord_list 按 snapNum 索引，每项是该 snap 的 iord 列表。
    """
    if start > end or start < 0:
        return np.array([], dtype=np.int64)
    return np.array(list(itertools.chain.from_iterable(iord_list[start:end])),
                    dtype=np.int64)


def init_evolution_arrays(structures, n_snap):
    """初始化 evolution 累积数组"""
    evo = {}
    for s in structures:
        evo[s] = {
            'mass':           np.full(n_snap, np.nan, dtype=np.float32),
            'cumsum_sf':      np.full(n_snap, np.nan, dtype=np.float32),
            'cumsum_ex_situ': np.full(n_snap, np.nan, dtype=np.float32),
            'SFR': None,
        }
        for key in _SNAP_PARAMS:
            evo[s][key] = np.full(n_snap, np.nan, dtype=np.float32)

    for s in ['bulge', 'halo']:
        for f in ['from_disk', 'from_colddisk', 'from_warmdisk']:
            evo[s][f] = np.full(n_snap, np.nan, dtype=np.float32)
    for s in ['colddisk', 'warmdisk', 'star']:
        evo[s]['AM'] = np.full((n_snap, 3), np.nan, dtype=np.float32)
    evo['warmdisk']['from_colddisk'] = np.full(n_snap, np.nan, dtype=np.float32)
    return evo


def init_accumulators():
    """初始化 accum 累积器"""
    return {
        'born': {
            'disk': [], 'spheroid': [],
            'colddisk': [], 'warmdisk': [],
        },
        'sf_all': [],
        'sf_spheroid': [],
        'sf_disk': [],
        'ex_all': [],
    }


def _write_snap_params_to_evo(evo, structure_dict, idx, n_snaps, structures):
    """把 snapshot 属性写入 evo 数组对应 idx"""
    for s in structures:
        for key in _SNAP_PARAMS:
            val = np.asarray(structure_dict[s][key])
            if val.ndim == 0:
                evo[s][key][idx] = val.item()
            else:
                D = val.shape[0]
                if evo[s][key].ndim == 1:
                    evo[s][key] = np.full((n_snaps, D), np.nan, dtype=np.float32)
                evo[s][key][idx, :] = val


def _find_prev_snap_in_pairs(pairs, current_i):
    """返回 pairs 中前一个 snapNum（首项返回 0）"""
    return pairs[current_i - 1][0] if current_i > 0 else 0


def record_step(evo, galaxy, accum, idx, structures,
                flat_iord_in_full, flat_iord_ex, last_snap,
                flat_iord_in_incr=None,
                snap_to_idx=None, snapNums_list=None, iord_in=None, *, snapnum_to_age):
    """
    将一次完整的运动学分解结果写入 evo。

    参数
    ----
    flat_iord_in_full : 自上次 full-decomp 以来所有 in-situ 恒星 ID
                        （用于组件 SFR / accum / cumsum_sf）
    flat_iord_in_incr : 仅当前 [last_snap, snapNum) 区间的 in-situ ID
                        （用于 evo['star']['SFR']，避免与 trace 模式重复）
    last_snap :         前一个分析的 snapNum（用于 born 分类）
    """
    if flat_iord_in_incr is None:
        flat_iord_in_incr = flat_iord_in_full

    born = accum['born']
    last_time = snapnum_to_age(last_snap)

    # ── 新生星分类（born 追踪）──
    mask_d = (galaxy.disk['tform'] > last_time) & (galaxy.disk['jzojc'] > 0.5)
    born['disk'].extend(galaxy.disk['iord'][mask_d])
    mask_s = galaxy.spheroid['tform'] > last_time
    born['spheroid'].extend(galaxy.spheroid['iord'][mask_s])
    mask_c = (galaxy.colddisk['tform'] > last_time) & (galaxy.colddisk['jzojc'] > 0.85)
    born['colddisk'].extend(galaxy.colddisk['iord'][mask_c])
    mask_w = (galaxy.warmdisk['tform'] > last_time) & (galaxy.warmdisk['jzojc'] > 0.5)
    born['warmdisk'].extend(galaxy.warmdisk['iord'][mask_w])

    # ── accum 累积 ──
    if isinstance(flat_iord_ex, np.ndarray):
        accum['ex_all'].extend(flat_iord_ex.tolist())
    else:
        accum['ex_all'].extend(flat_iord_ex)
    # Keep the star-level in-situ pool independent of component membership.
    accum['sf_all'].extend(flat_iord_in_full)
    accum['sf_spheroid'].extend(
        galaxy.spheroid['iord'][np.isin(galaxy.spheroid['iord'],
                                        flat_iord_in_full)])
    accum['sf_disk'].extend(
        galaxy.disk['iord'][np.isin(galaxy.disk['iord'],
                                    flat_iord_in_full)])

    sf_sphe = accum['sf_spheroid']
    sf_disk = accum['sf_disk']
    ex_all = accum['ex_all']

    # ── 逐组件写入 ──
    for s in structures:
        obj = getattr(galaxy, s)
        sid = obj['iord']

        # 质量
        evo[s]['mass'][idx] = obj['mass'].sum()

        # SFR: star 用增量区间，组件用全区间
        flat_for_sfr = flat_iord_in_incr if s == 'star' else flat_iord_in_full
        _, sfr_s = get_sfr(
            obj['GFM_InitialMass'][np.isin(sid, flat_for_sfr)],
            obj['tform'][np.isin(sid, flat_for_sfr)])
        evo[s]['SFR'] = sfr_s if evo[s]['SFR'] is None else evo[s]['SFR'] + sfr_s

        # cumsum_sf / cumsum_ex_situ
        pool = (accum['sf_all'] if s == 'star' else
                sf_sphe if s in ['bulge', 'halo'] else sf_disk)
        evo[s]['cumsum_sf'][idx] = obj['GFM_InitialMass'][
            np.isin(sid, pool)].sum()
        evo[s]['cumsum_ex_situ'][idx] = obj['mass'][
            np.isin(sid, ex_all)].sum()

        # ── Backfill：把前面 trace 模式 snapshot 的组件 cumsum_sf 填上 ──
        if s != 'star' and snap_to_idx is not None \
                and snapNums_list is not None and iord_in is not None:
            for earlier_idx in range(idx):
                if np.isnan(evo[s]['cumsum_sf'][earlier_idx]):
                    earlier_snap = snapNums_list[earlier_idx]
                    if earlier_snap > 0:
                        earlier_flat = flatten_iord(iord_in, 0, earlier_snap)
                        evo[s]['cumsum_sf'][earlier_idx] = \
                            obj['GFM_InitialMass'][
                                np.isin(sid, pool) & np.isin(sid, earlier_flat)
                            ].sum()
                    else:
                        evo[s]['cumsum_sf'][earlier_idx] = 0.0

        # from_xxx 迁移质量
        if s in ['bulge', 'halo']:
            for src in ['disk', 'colddisk', 'warmdisk']:
                evo[s][f'from_{src}'][idx] = obj['mass'][
                    np.isin(sid, born[src]) & (obj['jzojc'] < 0.5)].sum()
        if s == 'warmdisk':
            evo[s]['from_colddisk'][idx] = obj['mass'][
                np.isin(sid, born['colddisk']) & (obj['jzojc'] < 0.85)].sum()

    return accum


def _run_evolution_step(h5, data_grp, snap, galaxy, structure_dict,
                         evo, accum, evo_structures, i_pair, pairs,
                         iord_in, iord_ex, snap_to_idx, snapNum,
                         last_full_decomp_snap=0, *, snapnum_to_age):
    """完整分解模式下的 evolution 写入"""
    idx = snap_to_idx[snapNum]

    # AM 写入
    evo['star']['AM'][idx] = galaxy.s.AM
    evo['colddisk']['AM'][idx] = galaxy.colddisk.AM
    evo['warmdisk']['AM'][idx] = galaxy.warmdisk.AM

    # 可视化（faceon 后会改变 galaxy 引用）
    snap.faceon(align_with='star',
                range=[3 * snap.properties['eps'], 5 * snap.s.r50],
                as_context=False)
    galaxy_faceon = snap.container
    _write_snap_params_to_evo(evo, structure_dict, idx,
                              N_SNAPS_TOTAL, evo_structures)

    # ★ 关键：flat_in_full = 自上次 full 分解以来的所有 in-situ 星
    #   第一个 full 分解时 last_full_decomp_snap=0 → 覆盖 [0:snapNum]
    flat_in_full = flatten_iord(iord_in, last_full_decomp_snap, snapNum)

    # flat_in_incr = 仅当前区间（用于 evo['star']['SFR']，避免重复）
    last_snap = _find_prev_snap_in_pairs(pairs, i_pair)
    flat_in_incr = flatten_iord(iord_in, last_snap, snapNum)

    flat_ex = flatten_iord(iord_ex, last_snap, snapNum)

    record_step(evo, galaxy_faceon, accum, idx, evo_structures,
                flat_in_full, flat_ex, last_snap,
                flat_iord_in_incr=flat_in_incr,
                snap_to_idx=snap_to_idx,
                snapNums_list=SNAPSHOTS,
                iord_in=iord_in, snapnum_to_age=snapnum_to_age)

    evo_grp = data_grp.require_group('evolution')
    write_evolution_data(evo_grp, evo, evo_structures)

    # time_140（只写一次，取首次 full 分解时的完整 SFH 时间轴）
    if 'time_140' not in h5:
        sf_idx = np.isin(galaxy_faceon.s['iord'], flat_in_full)
        t_bins, _ = get_sfr(
            galaxy_faceon.s['GFM_InitialMass'][sf_idx],
            galaxy_faceon.s['tform'][sf_idx])
        h5.create_dataset('time_140', data=t_bins)
