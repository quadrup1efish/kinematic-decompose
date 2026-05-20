import os
import gc
import h5py
import scipy
import pickle
import numpy as np

import agama

import illustris_python as il

from kinematic_decompose.config import BASEPATH
from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.visualize import visualize_decomposition
from kinematic_decompose.mixture import AutoGaussianMixtureModel, util, preprocessing
from kinematic_decompose.gravity.kinematic_solver import create_multipole_potential, calculate_kinematic_param

import itertools
import tempfile
import pathlib
from pathlib import Path

RCUT_RANGE = [1, 7]
OUTPUT_PATH = Path("/home/tnguser/output/TNG50-1")
OUTPUT_PATH.mkdir(exist_ok=True)
stellar_assembly_dir = pathlib.Path(f"/home/tnguser/gsf/output/TNG50-1/stellar_assembly")

snapNums = [21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99]

# ---------- snapshot → age 查表 ----------
_age_data = np.array([
    [0, 0.0476, 20], [1, 0.0625, 15], [2, 0.0769, 12], [3, 0.0833, 11],
    [4, 0.0909, 10], [5, 0.0964, 9.4], [6, 0.1, 9], [7, 0.1058, 8.5],
    [8, 0.1111, 8], [9, 0.1161, 7.6], [10, 0.1216, 7.2], [11, 0.125, 7],
    [12, 0.1334, 6.5], [13, 0.1429, 6], [14, 0.1464, 5.8], [15, 0.1533, 5.5],
    [16, 0.1606, 5.2], [17, 0.1667, 5], [18, 0.1763, 4.7], [19, 0.1847, 4.4],
    [20, 0.1935, 4.2], [21, 0.2, 4], [22, 0.2124, 3.7], [23, 0.2226, 3.5],
    [24, 0.2332, 3.3], [25, 0.25, 3], [26, 0.2561, 2.9], [27, 0.2677, 2.7],
    [28, 0.279, 2.6], [29, 0.2902, 2.4], [30, 0.3012, 2.3], [31, 0.3121, 2.2],
    [32, 0.3228, 2.1], [33, 0.3333, 2], [34, 0.3439, 1.9], [35, 0.3543, 1.8],
    [36, 0.3645, 1.7], [37, 0.3747, 1.7], [38, 0.3848, 1.6], [39, 0.3948, 1.5],
    [40, 0.4, 1.5], [41, 0.4147, 1.4], [42, 0.4246, 1.4], [43, 0.4344, 1.3],
    [44, 0.4441, 1.3], [45, 0.4538, 1.2], [46, 0.4635, 1.2], [47, 0.4731, 1.1],
    [48, 0.4827, 1.1], [49, 0.4923, 1], [50, 0.5, 1], [51, 0.5115, 0.96],
    [52, 0.521, 0.92], [53, 0.5306, 0.88], [54, 0.5401, 0.85], [55, 0.5496, 0.82],
    [56, 0.5591, 0.79], [57, 0.5687, 0.76], [58, 0.5782, 0.73], [59, 0.5882, 0.7],
    [60, 0.5973, 0.67], [61, 0.6069, 0.65], [62, 0.6164, 0.62], [63, 0.626, 0.6],
    [64, 0.6357, 0.57], [65, 0.6453, 0.55], [66, 0.655, 0.53], [67, 0.6667, 0.5],
    [68, 0.6744, 0.48], [69, 0.6841, 0.46], [70, 0.6939, 0.44], [71, 0.7037, 0.42],
    [72, 0.7143, 0.4], [73, 0.7235, 0.38], [74, 0.7334, 0.36], [75, 0.7434, 0.35],
    [76, 0.7534, 0.33], [77, 0.7635, 0.31], [78, 0.7692, 0.3], [79, 0.7838, 0.28],
    [80, 0.794, 0.26], [81, 0.8043, 0.24], [82, 0.8146, 0.23], [83, 0.825, 0.21],
    [84, 0.8333, 0.2], [85, 0.8459, 0.18], [86, 0.8564, 0.17], [87, 0.8671, 0.15],
    [88, 0.8778, 0.14], [89, 0.8885, 0.13], [90, 0.8993, 0.11], [91, 0.9091, 0.1],
    [92, 0.9212, 0.086], [93, 0.9322, 0.073], [94, 0.9433, 0.06], [95, 0.9545, 0.048],
    [96, 0.9657, 0.035], [97, 0.9771, 0.023], [98, 0.9885, 0.012], [99, 1.0, 0.0]])
_snaps_arr = _age_data[:, 0]
_scales_arr = _age_data[:, 1]

def _a_dot(a, h0, om_m, om_l):
    om_k = 1.0 - om_m - om_l
    return h0 * a * np.sqrt(om_m * a**-3 + om_k * a**-2 + om_l)

def _a_dot_recip(*args):
    return 1.0 / _a_dot(*args)

def snapNum_to_age(snap):
    """snapNum → lookback time in Gyr.  snap=None or NaN → 0"""
    if snap is None or np.isnan(snap):
        return 0.0
    from pynbody import units
    conv = units.Unit("0.01 s Mpc km^-1").ratio('Gyr')
    a = _scales_arr[int(np.where(_snaps_arr == snap)[0][0])]
    return scipy.integrate.quad(_a_dot_recip, 0, a,
                                (0.6774, 0.3089, 0.6911))[0] * conv

"""
Train model !
"""

def train_auto_gaussian_mixture_model(galaxy, pot, jzojc_cut=0.5):

    eoemin_index = 0
    jzojc_index = 1
    jpojc_index = 2
    X = np.column_stack([galaxy.s['eoemin'], galaxy.s['jzojc'], galaxy.s['jpojc']])
    keep_particle = (galaxy.s['eoemin']<0)&(np.abs(galaxy.s['jzojc'])<1.5)&(galaxy.s['jpojc']<1.5)

    sph, disk = util.JEHistogram(galaxy.s['eoemin'][keep_particle], galaxy.s['jzojc'][keep_particle], n_E=25, n_eps=50)
    sph = (sph) & (np.abs(galaxy.s['jzojc'][keep_particle])<=0.5)
    eoemin_cut= util.get_Ecut(galaxy.s['eoemin'][keep_particle][sph], galaxy.s['mass'][keep_particle][sph], M_bin=100, m_bin=25, Mmin=0.1)
    r = np.logspace(-1, 1, 100)
    points = np.column_stack((r*0, r*0, r))
    potential = pot.potential(points)
    max_eoemin_cut = (potential/np.abs(galaxy.s['e'].min()))[np.searchsorted(r, RCUT_RANGE[1])]
    min_eoemin_cut = (potential/np.abs(galaxy.s['e'].min()))[np.searchsorted(r, RCUT_RANGE[0])]
    if eoemin_cut == 0 or max_eoemin_cut < eoemin_cut or eoemin_cut < min_eoemin_cut:
        eoemin_cut = (potential/np.abs(galaxy.s['e'].min()))[np.searchsorted(r, 3.5)]

    scaler = preprocessing.RobustScaler()
    X_train= scaler.fit_transform(X[keep_particle])

    eoemin_cut_train = scaler.transform(eoemin_cut, columns=eoemin_index)
    jzojc_cut_train = scaler.transform(jzojc_cut, columns=jzojc_index)
    r_jzojc_cut_train = scaler.transform(-jzojc_cut, columns=jzojc_index)

    auto_gmm = AutoGaussianMixtureModel()
    auto_gmm = auto_gmm.fit(X_train, 
                            eoemin_cut=eoemin_cut_train, 
                            jzojc_cut=jzojc_cut_train,
                            r_jzojc_cut = r_jzojc_cut_train, 
                            sample_weight=galaxy.s['mass'][keep_particle],
                            max_iter=200, 
                            min_iter=50)

    best_model = scaler.inverse_transform_GMM(auto_gmm.best_model) 
    del r, points, sph, disk
    gc.collect()
    return X, best_model, eoemin_cut, jzojc_cut

def save_structure_properties(sim):
    """返回 {comp: {key: value}}，comp 含 total/star/dm/disk/... 等"""
    attrs = ['mass', 'krot', 'beta', 'AM', 'vel_disp', 'vr_disp',
             'vR_disp', 'vz_disp', 'v_circ', 'v_rot', 'ke',
             'r50', 'R50', 'z50', 't50', 'shape']
    comp_map = {'star': 's'}
    for c in ['disk', 'colddisk', 'warmdisk', 'spheroid', 'bulge', 'halo',
              'counter_rotating_disk']:
        comp_map[c] = c

    result = {"total": {}, "dm": {}, "star": {}}
    result.update({c: {} for c in comp_map if c != 'star'})

    for key, attr_name in comp_map.items():
        obj = getattr(sim, attr_name)
        for a in attrs:
            result[key][a] = getattr(obj, a)
        if key != 'star':
            result[key]['Mass_frac'] = obj.Mass_frac
    return result


def kinematic_decomposition_pipeline(run, snapNum, subID):
    basePath = f"{BASEPATH}/{run}/output"

    # --- potential ---
    snap = Snapshot(basePath, snapNum)
    snap.load_particle(ID=subID, load_particle_fields='potential')
    snap.physical_units()
    snap.load_group_catalog(ID=subID)
    snap.GC_physical_units()
    snap.center(cen=snap.group_catalog['SubhaloPos'])
    snap.faceon(align_with='star',
                range=[3*snap.properties['eps'], 5*snap.s.r50],
                as_context=False)
    galaxy = snap.container
    pot = create_multipole_potential(galaxy['pos'], galaxy['mass'])
    del galaxy, snap; gc.collect()

    # --- decompose ---
    snap = Snapshot(basePath, snapNum)
    snap.load_particle(ID=subID, load_particle_fields={
        "star": ['Coordinates', 'Velocities', 'Masses',
                 'ParticleIDs', 'GFM_StellarFormationTime']})
    snap.physical_units()
    snap.load_group_catalog(ID=subID)
    snap.GC_physical_units()
    snap.center(cen=snap.group_catalog['SubhaloPos'])
    with snap.faceon(align_with='star',
                     range=[3*snap.properties['eps'], 5*snap.s.r50],
                     as_context=True):
        galaxy = snap.container
        galaxy = calculate_kinematic_param(galaxy, pot)
        X, model, eoemin_cut, jzojc_cut = train_auto_gaussian_mixture_model(galaxy, pot)
        galaxy = util.decompose(X, galaxy, model, eoemin_cut, jzojc_cut,
                                predict_method='hard')
        structure_dict = save_structure_properties(galaxy)
        gmm_dict = util.decompose_mixture_model(model, eoemin_cut, jzojc_cut, -jzojc_cut)

    return structure_dict, gmm_dict, pot, snap

"""
安全写入工具
"""
def safe_write_group(grp, temp_prefix, final_name, writer_func):
    """writer_func(temp_grp) 写完自动 rename, 异常自动清理"""
    temp_name = f'{temp_prefix}_{final_name}'
    temp_grp = grp.create_group(temp_name)
    try:
        writer_func(temp_grp)
        grp.move(temp_name, final_name)
    except Exception:
        if temp_name in grp:
            del grp[temp_name]
        raise


def has_group(grp, name):
    return name in grp


# === 写入 ===
def write_snapshot_data(snap_grp, structure_dict):
    """structure_dict: {comp: {key: value}}"""
    for comp, sub in structure_dict.items():
        cg = snap_grp.create_group(comp)
        for k, v in sub.items():
            if isinstance(v, np.ndarray):
                cg.create_dataset(k, data=v)
            else:
                cg.attrs[k] = v


def write_potential(grp, pot):
    """pot → attribute"""
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
    tmp.close()
    pot.export(tmp.name)
    with open(tmp.name, 'r') as f:
        grp.attrs['potential_ini'] = f.read()
    os.unlink(tmp.name)

def read_potential(snap_grp):
    return agama.Potential(snap_grp.attrs['potential_ini'])

def write_dict_to_group(grp, d):
    """{key: value} → group 下的 dataset 或 attribute"""
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            grp.create_dataset(k, data=v)
        elif isinstance(v, (int, float, str, bool)):
            grp.attrs[k] = v

def write_evolution_data(evo_grp, evo_dict, structures):
    for s in structures:
        sg = evo_grp.create_group(s)
        for k, v in evo_dict[s].items():
            if isinstance(v, np.ndarray):
                sg.create_dataset(k, data=v)
    # 顶层
    for k in ['time_12', 'time_140', 'SFR']:
        v = evo_dict.get(k)
        if isinstance(v, np.ndarray):
            evo_grp.create_dataset(k, data=v)


"""
Evolution tracker
"""

def read_in_ex_situ(halo_id):
    filename = f"{halo_id}_stellar_assembly.pkl"
    full_path= stellar_assembly_dir / filename
    with open(full_path, "rb") as f:
        data = pickle.load(f)
    return data['iord_in'], data['iord_ex']

def get_sfr(mass, tform, bins=139, range=[0,14]):
    valid_mask = ~np.isnan(mass) & ~np.isnan(tform)
    mass_sum, bin_edges, binnumber = scipy.stats.binned_statistic(tform[valid_mask], mass[valid_mask], 
                                                  statistic='sum', bins=bins, 
                                                  range=range)
    delta_t = np.diff(bin_edges)
    SFR = mass_sum / delta_t / 1e9
    SFR = np.insert(SFR, 0, 0)
    return bin_edges, SFR

def get_cumsum_sf(mass, tform, bins=139, range=[0,14]):
    valid_mask = ~np.isnan(mass) & ~np.isnan(tform)
    mass_sum, bin_edges, binnumber = scipy.stats.binned_statistic(tform[valid_mask], mass[valid_mask], 
                                                  statistic='sum', bins=bins, 
                                                  range=range)
    delta_t = np.diff(bin_edges) 
    mass_sum  = np.cumsum(np.insert(mass_sum, 0, 0))

    return bin_edges, mass_sum

def init_evolution_arrays(structures, n_snap):
    evo = {'time_12': np.full(n_snap, np.nan),
           'time_140': None,
           'SFR': None}
    for s in structures:
        evo[s] = {
            'mass':          np.full(n_snap, np.nan),
            'cumsum_sf':     np.full(n_snap, np.nan),
            'cumsum_ex_situ': np.full(n_snap, np.nan),
            'SFR': None,
        }
    for s in ['bulge', 'halo']:
        for f in ['from_disk', 'from_colddisk', 'from_warmdisk']:
            evo[s][f] = np.full(n_snap, np.nan)
    for s in ['colddisk', 'warmdisk']:
        evo[s]['AM'] = np.full((n_snap, 3), np.nan)
    evo['warmdisk']['from_colddisk'] = np.full(n_snap, np.nan)
    return evo

def init_accumulators():
    return {
        'born': {
            'disk':     np.array([], dtype=np.int64),
            'spheroid': np.array([], dtype=np.int64),
            'colddisk': np.array([], dtype=np.int64),
            'warmdisk': np.array([], dtype=np.int64),
        },
        'sf_spheroid': [],   # 累积 in-situ spheroid iord
        'sf_disk':     [],   # 累积 in-situ disk iord
        'ex_all':      np.array([], dtype=np.int64),  # 累积 ex-situ iord
    }

def record_step(evo, galaxy, accum, i, structures,
                flat_iord_in, flat_iord_ex, last_time):
    born = accum['born']

    # --- born particles ---
    mask_d = (galaxy.disk['tform'] > last_time) & (galaxy.disk['jzojc'] > 0.5)
    born['disk'] = np.hstack([born['disk'], galaxy.disk['iord'][mask_d]])
    mask_s = galaxy.spheroid['tform'] > last_time
    born['spheroid'] = np.hstack([born['spheroid'], galaxy.spheroid['iord'][mask_s]])
    mask_c = (galaxy.colddisk['tform'] > last_time) & (galaxy.colddisk['jzojc'] > 0.85)
    born['colddisk'] = np.hstack([born['colddisk'], galaxy.colddisk['iord'][mask_c]])
    mask_w = (galaxy.warmdisk['tform'] > last_time) & (galaxy.warmdisk['jzojc'] > 0.5)
    born['warmdisk'] = np.hstack([born['warmdisk'], galaxy.warmdisk['iord'][mask_w]])

    # --- ex-situ accum ---
    accum['ex_all'] = np.hstack([accum['ex_all'], flat_iord_ex])
    accum['sf_spheroid'].extend(
        galaxy.spheroid['iord'][np.isin(galaxy.spheroid['iord'], flat_iord_in)])
    accum['sf_disk'].extend(
        galaxy.disk['iord'][np.isin(galaxy.disk['iord'], flat_iord_in)])

    sf_sphe = accum['sf_spheroid']
    sf_disk = accum['sf_disk']
    ex_all  = accum['ex_all']

    # --- global SFR ---
    sf_idx = np.isin(galaxy.s['iord'], flat_iord_in)
    t_bins, sfr = get_sfr(galaxy.s['mass'][sf_idx], galaxy.s['tform'][sf_idx])
    if evo['time_140'] is None:
        evo['time_140'] = t_bins
    evo['SFR'] = sfr if evo['SFR'] is None else evo['SFR'] + sfr

    # --- per structure ---
    for s in structures:
        obj = getattr(galaxy, s)
        sid  = obj['iord']
        evo[s]['mass'][i] = obj['mass'].sum()

        # SFR
        _, sfr_s = get_sfr(obj['mass'][np.isin(sid, flat_iord_in)],
                           obj['tform'][np.isin(sid, flat_iord_in)])
        evo[s]['SFR'] = sfr_s if evo[s]['SFR'] is None else evo[s]['SFR'] + sfr_s

        # cumsum_sf
        pool = sf_sphe if s in ['bulge', 'halo'] else sf_disk
        evo[s]['cumsum_sf'][i] = obj['mass'][np.isin(sid, pool)].sum()

        # cumsum_ex_situ
        evo[s]['cumsum_ex_situ'][i] = obj['mass'][np.isin(sid, ex_all)].sum()

        # from_disk / from_colddisk / from_warmdisk
        if s in ['bulge', 'halo']:
            for src in ['disk', 'colddisk', 'warmdisk']:
                evo[s][f'from_{src}'][i] = obj['mass'][
                    np.isin(sid, born[src]) & (obj['jzojc'] < 0.5)].sum()
        if s == 'warmdisk':
            evo[s]['from_colddisk'][i] = obj['mass'][
                np.isin(sid, born['colddisk']) & (obj['jzojc'] < 0.85)].sum()
    return accum

def load_sorted_tree(basePath, subID):
    tree = il.sublink.loadTree(basePath, snapNum=99, id=subID,
                               fields=['SubfindID', 'SnapNum'],
                               onlyMPB=True, onlyMDB=False,
                               treeName="SubLink_gal", cache=False)
    pairs = [(sn, id_) for sn, id_ in zip(tree['SnapNum'], tree['SubfindID'])
             if sn in snapNums]
    pairs.sort(key=lambda x: x[0])
    return pairs


def main(run, subID, h5_path=None):
    if h5_path is None:
        h5_path = str(OUTPUT_PATH / "results.h5")

    structures = ['bulge', 'halo', 'colddisk', 'warmdisk', 'counter_rotating_disk']
    sub_name = str(subID)

    with h5py.File(h5_path, 'a') as h5:
        if sub_name in h5 and 'evolution' in h5[sub_name]:
            print(f"  [{subID}] already done, skip")
            return

        pairs = load_sorted_tree(f"{BASEPATH}/{run}/output", subID)
        if not pairs:
            print(f"  [{subID}] no valid snapshots in tree")
            return

        sub_grp = h5.require_group(sub_name)
        evo = init_evolution_arrays(structures, len(pairs))
        accum = init_accumulators()

        iord_in, iord_ex = read_in_ex_situ(halo_id=subID)

        for i, (snapNum, ID) in enumerate(pairs):
            snap_str = str(snapNum)
            if snap_str in sub_grp:
                print(f"    snap {snapNum} cached, skip")
                continue

            last_snap = pairs[i-1][0] if i > 0 else 0
            last_time = snapNum_to_age(last_snap)

            # 本 snap 时间间隔内新形成的 in-situ / ex-situ 粒子
            flat_in = np.array(list(itertools.chain.from_iterable(
                iord_in[last_snap:snapNum]))) if last_snap < snapNum else np.array([], dtype=np.int64)
            flat_ex = np.array(list(itertools.chain.from_iterable(
                iord_ex[last_snap:snapNum]))) if last_snap < snapNum else np.array([], dtype=np.int64)

            # Step 1: decompose → AM (faceon 前)
            structure_dict, gmm_dict, pot, snapshot = \
                kinematic_decomposition_pipeline(run, snapNum, ID)
            galaxy = snapshot.container

            evo['colddisk']['AM'][i] = galaxy.colddisk.AM
            evo['warmdisk']['AM'][i] = galaxy.warmdisk.AM
            evo['time_12'][i] = snapNum_to_age(snapNum)

            # Step 2: faceon
            snapshot.faceon(align_with='star',
                            range=[3*snapshot.properties['eps'],
                                   5*snapshot.s.r50],
                            as_context=False)
            galaxy = snapshot.container

            # Step 3: 安全写入 snapshot
            temp_name = f'_tmp_{snapNum}'
            temp_grp = sub_grp.create_group(temp_name)
            try:
                write_snapshot_data(temp_grp, structure_dict)
                write_potential(temp_grp, pot)
                if gmm_dict:
                    write_dict_to_group(temp_grp.create_group('gmm'), gmm_dict)
                sub_grp.move(temp_name, snap_str)
            except Exception:
                if temp_name in sub_grp:
                    del sub_grp[temp_name]
                raise

            # Step 4: evolution 增量
            record_step(evo, galaxy, accum, i, structures,
                        flat_in, flat_ex, last_time)

            del snapshot, galaxy; gc.collect()
            print(f"    snap {snapNum} done")

        # Step 5: 安全写入 evolution
        evo_temp = '_tmp_evo'
        evo_tmp_grp = sub_grp.create_group(evo_temp)
        try:
            write_evolution_data(evo_tmp_grp, evo, structures)
            sub_grp.move(evo_temp, 'evolution')
        except Exception:
            if evo_temp in sub_grp:
                del sub_grp[evo_temp]
            raise

    print(f"  [{subID}] complete")

if __name__ == '__main__':
    run = "TNG50-1"
    subID = 198198
    main(run, subID)