"""Library port of the original in-situ/ex-situ classification."""

import pickle
from pathlib import Path

import numpy as np

from ..PyTNG import illustris_python as il


def read_in_ex_situ(assembly_dir, subfind_id):
    """Read snapshot-indexed buckets from a trusted legacy pickle."""
    path = Path(assembly_dir) / f'{int(subfind_id)}_stellar_assembly.pkl'
    with path.open('rb') as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or not {'iord_in', 'iord_ex'} <= data.keys():
        raise ValueError(f'Missing assembly buckets in {path}')
    buckets = data['iord_in'], data['iord_ex']
    if any(not isinstance(group, (list, tuple, np.ndarray)) or len(group) != 100
           for group in buckets):
        raise ValueError(f'Expected 100 snapshot buckets per classification in {path}')
    return buckets


def cal_in_ex_situ(MP, halo_id, start=0, end=99, *, base_path, snapnum_to_scale_factor=None):
    """Return the original snapshot-indexed buckets, without writing a file.

    By default, read the scale factor from the group catalog Header's ``Time``.
    Pass the original ``tools.snapnum_to_scale_factor`` to reproduce old thresholds.
    The original window semantics are retained: the cohort is always read at
    snapshot 99, and only the lower bound filters the tree. Use the original
    defaults unless intentionally reproducing a different legacy invocation.
    """
    length = end - start + 1
    fields = ['GFM_StellarFormationTime', 'ParticleIDs']
    halo_pIDs = il.snapshot.loadSubhalo(base_path, 99, halo_id, 'stars', fields=fields)

    ParticleIDs_halo = np.asarray(halo_pIDs['ParticleIDs'])
    ParticleIDs_mainleaf = []
    iord_in = [[] for _ in range(length)]
    iord_ex = [[] for _ in range(length)]

    prog_halo_ids = MP['SubfindID']
    prog_snapnums = MP['SnapNum']
    flag = 0
    last_snap_processed = None
    for snapshot_prog, prog_halo_id in zip(prog_snapnums[::-1], prog_halo_ids[::-1]):
        if snapshot_prog < start:
            continue
        subhalo_prog = il.snapshot.loadSubhalo(
            base_path, snapshot_prog, prog_halo_id, 'stars', fields=fields)
        try:
            tform_prog = np.asarray(subhalo_prog['GFM_StellarFormationTime'])
            ParticleIDs_prog = np.asarray(subhalo_prog['ParticleIDs'])
        except BaseException:
            # The legacy script skips field extraction failures, not load failures.
            continue

        exist_index = np.isin(ParticleIDs_prog, ParticleIDs_halo)
        tform_prog = tform_prog[exist_index]
        ParticleIDs_prog = ParticleIDs_prog[exist_index]

        if flag == 0:
            ParticleIDs_in_situ = ParticleIDs_prog
            flag = 1
        else:
            if snapnum_to_scale_factor is None:
                scale = il.groupcat.loadHeader(base_path, last_snap_processed)['Time']
            else:
                scale = snapnum_to_scale_factor(last_snap_processed)
            newstar_criterion = scale + 1e-10
            is_in_situ = ((tform_prog <= 0) | (tform_prog > newstar_criterion)) & ~np.isin(
                ParticleIDs_prog, ParticleIDs_mainleaf)
            ParticleIDs_in_situ = ParticleIDs_prog[is_in_situ]

        ParticleIDs_mainleaf.extend(ParticleIDs_in_situ)
        is_ex_situ = ~np.isin(ParticleIDs_prog, ParticleIDs_mainleaf)
        ParticleIDs_ex_situ = ParticleIDs_prog[is_ex_situ]
        ParticleIDs_mainleaf.extend(ParticleIDs_ex_situ)
        iord_in[snapshot_prog].extend(ParticleIDs_in_situ)
        iord_ex[snapshot_prog].extend(ParticleIDs_ex_situ)
        last_snap_processed = snapshot_prog

    return iord_in, iord_ex


def compute_stellar_assembly(base_path, subfind_id, *, snapnum_to_scale_factor=None,
                             tree_name='SubLink_gal'):
    """Run the legacy classification rule using Header scales by default."""
    tree = il.sublink.loadTree(
        base_path, 99, subfind_id,
        fields=['SnapNum', 'SubfindID', 'SubhaloMassType'],
        onlyMPB=True, treeName=tree_name,
    )
    return cal_in_ex_situ(tree, subfind_id, base_path=base_path,
                          snapnum_to_scale_factor=snapnum_to_scale_factor)
