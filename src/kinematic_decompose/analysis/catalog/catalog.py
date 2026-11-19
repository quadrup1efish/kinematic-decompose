"""Flexible skew-t catalog entry points for TNG simulations.

Use ``run_traceback`` for a z=0 galaxy, ``batch_compute`` for independent
snapshots, and ``build_tree`` to write the optional tree table.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np

from ...PyTNG import illustris_python as il
from ..inner_cgm import LOCAL_COOLING_TABLE_DIR
from ..stellar_assembly import compute_stellar_assembly, read_in_ex_situ
from ._evolution import SNAPSHOTS, N_SNAPS_TOTAL, init_accumulators, init_evolution_arrays
from ._snapshot import compute_and_store, _store_inner_cgm
from ._storage import (
    TREE_DTYPE, _cached_payload_valid, _check_catalog_schema, _repair_cached_catalogs,
    _set_traceback_complete, write_evolution_data,
)

__all__ = ['build_catalog', 'run_traceback', 'batch_compute', 'build_tree', 'SNAPSHOTS']


def build_catalog(base_path, h5_path=None, *, z0_targets=(), snapshot_targets=(),
                  snapnum_to_age=None, overwrite=False, fresh=False, stellar_assembly_dir=None,
                  cooling_table_dir=None, compute_inner_cgm=False):
    """Process main branches, then independent snapshots. Return batch counts.

    With no h5_path, write <run>_kinematic_catalog_flexible_ecut.h5 in the
    current directory, deriving the run name from the snapshot path.
    Pass the original snapshot-to-age (Gyr) callable for traceback targets.
    Pass stellar_assembly_dir to read trusted legacy assembly pickles.
    Set compute_inner_cgm=True to use the local cooling tables, or pass an
    explicit cooling_table_dir to enable the diagnostic with custom tables.
    """
    if h5_path is None:
        source = Path(base_path).resolve()
        run = source.parent.name if source.name == 'output' else source.name
        h5_path = Path(f'{run}_kinematic_catalog_flexible_ecut.h5')
    if compute_inner_cgm and cooling_table_dir is None:
        cooling_table_dir = LOCAL_COOLING_TABLE_DIR
    z0_targets = tuple(z0_targets)
    if z0_targets and not callable(snapnum_to_age):
        raise TypeError('snapnum_to_age is required for z=0 traceback targets')
    if fresh:
        with h5py.File(h5_path, 'x', libver='latest') as h5:
            h5.attrs.update(
                run_id=os.environ.get('KD_CATALOG_RUN_ID') or uuid4().hex,
                started_utc=datetime.now(timezone.utc).isoformat(),
                fresh_output=True,
                potential_policy='rebuild_from_snapshot_star_gas_dm',
            )
    for subfind_id in z0_targets:
        if not run_traceback(base_path, int(subfind_id), h5_path,
                             snapnum_to_age=snapnum_to_age, overwrite=overwrite,
                             stellar_assembly_dir=stellar_assembly_dir,
                             cooling_table_dir=cooling_table_dir):
            raise RuntimeError(f'Traceback incomplete for z=0 SubfindID {subfind_id}')
    return batch_compute(base_path, snapshot_targets, h5_path, overwrite=overwrite,
                         cooling_table_dir=cooling_table_dir)


def load_sorted_tree(basePath, subID):
    tree = il.sublink.loadTree(basePath, snapNum=99, id=subID,
                               fields=['SubfindID', 'SnapNum'],
                               onlyMPB=True, onlyMDB=False,
                               treeName="SubLink_gal", cache=False)
    pairs = [(sn, id_) for sn, id_ in zip(tree['SnapNum'], tree['SubfindID'])
             if sn in SNAPSHOTS]
    pairs.sort(key=lambda x: x[0])
    return pairs


def run_traceback(base_path, z0_subID, h5_path, *, snapnum_to_age, overwrite=False,
                  stellar_assembly_dir=None, cooling_table_dir=None):
    """Decompose a z=0 main branch and record its evolution.

    Read trusted legacy assembly buckets if a directory is supplied;
    otherwise compute them in memory. Missing caches are not recomputed
    silently. The HDF5 completion check is unchanged.
    """
    basePath = str(base_path)

    # Keep only this target's assembly in memory; failures occur before HDF5 writes.
    print(f"  [ASSEMBLY] z0_subID={z0_subID}", flush=True)
    if stellar_assembly_dir is None:
        iord_in, iord_ex = compute_stellar_assembly(basePath, int(z0_subID))
    else:
        iord_in, iord_ex = read_in_ex_situ(stellar_assembly_dir, z0_subID)

    with h5py.File(h5_path, 'a', libver='latest') as h5:
        _check_catalog_schema(h5)
        pairs = load_sorted_tree(basePath, z0_subID)
        if not pairs:
            print(f"  [z0_subID={z0_subID}] no valid snapshots in tree")
            return False

        # ── ★ 检查 z=0 snapshot 是否有 _traceback_complete 标记 ──
        z0_snapNum, z0_progenitorID = pairs[-1]   # 最后一个 = z=0
        z0_key = f"{z0_snapNum}/{z0_progenitorID}"
        z0_grp = h5.get(z0_key)

        if not overwrite and z0_grp is not None \
                and z0_grp.attrs.get('_traceback_complete', False) \
                and _cached_payload_valid(z0_grp) \
                and all(f'evolution/star/{key}' in z0_grp
                        and z0_grp[f'evolution/star/{key}'].shape == (N_SNAPS_TOTAL,)
                        for key in ('mass', 'cumsum_sf', 'cumsum_ex_situ')) \
                and 'time_12' in h5 and h5['time_12'].shape == (N_SNAPS_TOTAL,):
            _repair_cached_catalogs(h5, z0_snapNum, z0_progenitorID, z0_grp)
            if cooling_table_dir is not None:
                for snap, sid in pairs:
                    _store_inner_cgm(h5[f'{snap}/{sid}'], basePath, snap, sid, cooling_table_dir)
            print(f"  ✓ z0_subID={z0_subID}: trace-back already complete, skipping")
            return True

        _set_traceback_complete(h5, z0_snapNum, z0_progenitorID, False)

        print(f"\n=== Trace-back: z0_subID={z0_subID}, "
              f"{len(pairs)} ancestors ===\n")

        snap_to_idx = {sn: i for i, sn in enumerate(SNAPSHOTS)}
        found_snaps = {sn for sn, _ in pairs}
        missing = [sn for sn in SNAPSHOTS if sn not in found_snaps]
        if missing:
            print(f"  missing from tree: snap {missing}")

        structures = ['star', 'bulge', 'halo', 'colddisk',
                       'warmdisk', 'counter_rotating_disk']
        evo = init_evolution_arrays(structures, N_SNAPS_TOTAL)
        accum = init_accumulators()
        last_full = 0

        success = True

        for i_pair, (snapNum, progenitorID) in enumerate(pairs):
            print(f"\n  --- snap {snapNum}, subID={progenitorID} "
                  f"({i_pair + 1}/{len(pairs)}) ---")

            try:
                grp, _, last_full = compute_and_store(
                    h5, snapNum, progenitorID, basePath,
                    snapnum_to_age=snapnum_to_age,
                    overwrite=overwrite,
                    track_evolution=True,
                    evo=evo, accum=accum, evo_structures=structures,
                    i_pair=i_pair, pairs=pairs,
                    iord_in=iord_in, iord_ex=iord_ex,
                    snap_to_idx=snap_to_idx,
                    last_full_decomp_snap=last_full)
                _store_inner_cgm(grp, basePath, snapNum, progenitorID, cooling_table_dir,
                                 overwrite=overwrite)

            except Exception as e:
                print(f"\n  ✗ ERROR at snap {snapNum}, subID={progenitorID}: {e}")
                import traceback
                traceback.print_exc()
                success = False
                break    # 依赖链断裂，停止处理后续 snapshot

        # Publish chain completion only after final evolution and time writes.
        if success:
            z0_grp = h5.require_group(z0_key)
            z0_grp.attrs['_traceback_snap_count'] = len(pairs)
            evo_grp = z0_grp.require_group('evolution')
            write_evolution_data(evo_grp, evo, structures)

        # time_12
        times_12 = np.array([snapnum_to_age(sn) for sn in SNAPSHOTS],
                            dtype=np.float32)
        if 'time_12' in h5:
            ds = h5['time_12']
            if ds.shape == times_12.shape and ds.dtype == times_12.dtype:
                ds[...] = times_12
            else:
                del h5['time_12']
                h5.create_dataset('time_12', data=times_12)
        else:
            h5.create_dataset('time_12', data=times_12)

        if success:
            h5.flush()
            _set_traceback_complete(h5, z0_snapNum, z0_progenitorID, True)
            print(f"\n  ✓ z0_subID={z0_subID} complete! "
                  f"({len(pairs)} snapshots)")
        else:
            print(f"\n  ⚠ z0_subID={z0_subID} incomplete, re-run to retry")
        return success


def build_tree(base_path, z0_subIDs, h5_path):
    """构建 tree 表"""
    basePath = str(base_path)

    all_rows = []
    for z0_subID in z0_subIDs:
        pairs = load_sorted_tree(basePath, z0_subID)
        for snapNum, progenitorID in pairs:
            all_rows.append((z0_subID, snapNum, progenitorID))
        print(f"  ✓ z0_subID={z0_subID}: {len(pairs)} ancestors")

    arr = np.array(all_rows, dtype=TREE_DTYPE)

    with h5py.File(h5_path, 'a') as h5:
        if 'tree' in h5:
            del h5['tree']
        h5.create_dataset('tree', data=arr, chunks=True, compression='gzip')

    print(f"\n  ✓ Tree built! Total entries: {len(arr)}")


def batch_compute(base_path, targets, h5_path, *, overwrite=False, cooling_table_dir=None):
    """批量计算独立星系（不做 trace-back / evolution）。"""
    validated = []
    for t in targets:
        if isinstance(t, (list, tuple)) and len(t) == 2:
            s, sid = int(t[0]), int(t[1])
            if s in SNAPSHOTS:
                validated.append((s, sid))
            else:
                print(f"  ⚠ skip invalid snapNum={t[0]}, must be in {SNAPSHOTS}")
        else:
            print(f"  ⚠ skip invalid target: {t}")

    if not validated:
        print("No valid targets!")
        return 0, 0, 0

    n_total = len(validated)
    n_computed = n_skipped = n_errors = 0

    # Each object commits its payload and summary row before completion.
    # Row-key caches avoid a full index scan for every galaxy.
    with h5py.File(h5_path, 'a', libver='latest',
                   rdcc_nbytes=16 * 1024 * 1024) as h5:
        _check_catalog_schema(h5)

        for i, (snapNum, subfindID) in enumerate(validated):
            data_key = f"{snapNum}/{subfindID}"
            print(f"\n  [{i + 1}/{n_total}] {data_key}")
            try:
                if data_key in h5 and not overwrite:
                    grp = h5[data_key]
                    if _cached_payload_valid(grp):
                        _repair_cached_catalogs(h5, snapNum, subfindID, grp)
                        _store_inner_cgm(grp, str(base_path), snapNum, subfindID, cooling_table_dir)
                        method = grp.attrs.get('_method', 'full')
                        skipped = grp.attrs.get('_skipped', False)
                        status = 'skipped' if skipped else f'ok ({method})'
                        print(f"    ✓ already exists ({status})")
                        n_skipped += 1
                        continue
            except Exception as e:
                print(f"Wrong {subfindID}: {e}")
                n_errors += 1
                continue

            try:
                compute_and_store(h5, snapNum, subfindID, str(base_path),
                                  overwrite=overwrite,
                                  track_evolution=False)
                _store_inner_cgm(h5[data_key], str(base_path), snapNum, subfindID,
                                 cooling_table_dir, overwrite=overwrite)
                n_computed += 1
            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                n_errors += 1
                import traceback
                traceback.print_exc()


    print("\n  === Batch done ===")
    print(f"      Computed: {n_computed}")
    print(f"      Skipped:  {n_skipped}")
    print(f"      Errors:   {n_errors}")
    return n_computed, n_skipped, n_errors
