"""Scientific calculation for a full snapshot; no catalog I/O."""

import numpy as np

from ...mixture import AutoGaussianMixtureModel, preprocessing, util
from ...pipeline import DEFAULT_GMM_MAX_ITER, DEFAULT_GMM_MIN_ITER


def train_auto_gaussian_mixture_model(galaxy, jzojc_cut=0.5, *, eoemin_distribution=None):
    """训练 AutoGMM 并返回 (X, best_model, eoemin_cut, jzojc_cut)"""
    X = np.column_stack([galaxy.s['eoemin'],
                         galaxy.s['jzojc'],
                         galaxy.s['jpojc']])
    keep = (galaxy.s['eoemin'] < 0) & (np.abs(galaxy.s['jzojc']) < 1.5) & (galaxy.s['jpojc'] < 1.5)
    sph, _ = util.JEHistogram(
        galaxy.s['eoemin'][keep], galaxy.s['jzojc'][keep], n_E=25, n_eps=50)

    sph &= np.abs(galaxy.s['jzojc'][keep]) <= 0.5

    eoemin_cut = util.get_Ecut_skewt(
        galaxy.s['eoemin'][keep][sph], galaxy.s['mass'][keep][sph],
        distribution=eoemin_distribution)

    if eoemin_cut is None:
        raise ValueError("Skew-t ECUT is unavailable; no fallback is configured")

    scaler = preprocessing.RobustScaler()
    X_train = scaler.fit_transform(X[keep])

    eoemin_cut_train = scaler.transform(eoemin_cut, columns=0)
    jzojc_cut_train = scaler.transform(jzojc_cut, columns=1)
    r_jzojc_cut_train = scaler.transform(-jzojc_cut, columns=1)

    auto_gmm = AutoGaussianMixtureModel()
    auto_gmm = auto_gmm.fit(
        X_train,
        eoemin_cut=eoemin_cut_train,
        jzojc_cut=jzojc_cut_train,
        r_jzojc_cut=r_jzojc_cut_train,
        sample_weight=galaxy.s['mass'][keep], max_iter=DEFAULT_GMM_MAX_ITER,
        min_iter=DEFAULT_GMM_MIN_ITER,
        scaler=scaler, use_float32=True)

    best_model = scaler.inverse_transform_GMM(auto_gmm.best_model)
    return X, best_model, eoemin_cut, jzojc_cut


def save_structure_properties(sim):
    """从已分解的 sim 中提取各组件属性字典"""
    attrs = ['mass', 'krot', 'beta', 'AM', 'vel_disp',
             'vr_disp', 'vR_disp', 'vz_disp', 'v_circ', 'v_rot',
             'ke', 'r50', 'R50', 'z50', 't50', 'Mdyn', 'shape']
    comp_map = {c: c for c in
                ['disk', 'colddisk', 'warmdisk', 'spheroid',
                 'bulge', 'halo', 'counter_rotating_disk']}
    comp_map['star'] = 's'

    result = {}
    for key, attr_name in comp_map.items():
        obj = getattr(sim, attr_name)
        result[key] = {a: getattr(obj, a) for a in attrs}
        if key != 'star':
            result[key]['Mass_frac'] = obj.Mass_frac
    return result
