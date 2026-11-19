# Flexible skew-t catalog

From a research script:

```python
from kinematic_decompose.analysis.catalog import build_catalog

counts = build_catalog(
    '/data/TNG50-1/output',
    z0_targets=[433289],
    snapshot_targets=[(84, 1234), (99, 433289)],
    stellar_assembly_dir='/data/TNG50-1/stellar_assembly',
    fresh=True,
)
```
Replace the example IDs and paths with your targets. By default, the example writes `TNG50-1_kinematic_catalog_flexible_ecut.h5` in the current directory. Pass `h5_path` explicitly to choose a different name or location; the parent directory must exist. Set `fresh=False` (the default) to resume; `fresh=True` refuses to replace an existing file. `overwrite=True` recomputes selected objects. `counts` is `(computed, skipped, errors)` for independent snapshots; the combined runner raises if a main branch is incomplete. `run_traceback` handles one main branch. `build_tree(base_path, z0_targets, h5_path)` writes the optional tree table.

Traceback derives cosmic ages (Gyr) from the TNG group catalog headers and reuses them across targets in one call. By default, each traceback computes origin IDs in memory using group catalog Header scale factors. To reuse existing results, pass `stellar_assembly_dir` to `build_catalog` or `run_traceback`. This reads `<z0_subfind_id>_stellar_assembly.pkl` from that directory without rewriting it. Only use trusted pickle files from the same simulation run. A missing cache raises an error rather than silently recomputing. Assembly data are obtained before the HDF5 completion check. Independent snapshot targets do not use this option.

## Reading the implementation

New snapshot entries also store `/<snap>/<subfind_id>/vmax` and `v200c` in km/s. `vmax` is TNG's `SubhaloVmax`; `v200c` is derived from `Group_M_Crit200` and the physical `Group_R_Crit200` of the parent FoF group identified by `SubhaloGrNr`. For satellites the denominator still refers to the parent FoF group. Compute the ratio from the two scalars; undefined values are stored as `NaN`. Existing completed entries are not backfilled on a cache hit.

On TNG Lab, pass `stellar_assembly_dir=Path(f"/home/tnguser/gsf/output/{run}/stellar_assembly")` (import `Path` from `pathlib`). The library does not assume a server-specific default.

Newly computed snapshots store dimensionless `s_b` at `/<snap>/<subfind_id>/s_b`. Both full and trace modes use the baryon–total velocity offset inside 20 physical kpc, normalized by the circular speed from the total mass enclosed by the stellar `R90`. Undefined values are stored as `NaN`. Existing completed entries are not backfilled on a cache hit, and the summary table schema is unchanged.

1. `catalog/catalog.py`: run order, reuse checks and batch entry points. `__init__.py` only exports the public API.
2. `_snapshot.py`: choose full decomposition or lightweight SFH and connect results to storage.
3. `_decomposition.py`: skew-t energy cut, GMM, and component properties. Lightweight snapshot measurements use `analysis/sfh_trace.py` instead.
4. `_evolution.py`: particle-ID windows, cumulative histories and the simulation-frame angular momenta.
5. `_storage.py`: snapshot groups, catalog row layouts, and completion markers.

## Compatibility and limits

New files have one summary table, `/catalog`, keyed by `(snapNum, subfindID)`. It includes processing status, host ID and central status, particle counts, component masses, `krot`, `r50`, velocities, `s_b`, and the three CGM scalars. Missing CGM results are `NaN`; CGM quality attributes remain in the snapshot's `inner_cgm` group. Use a new output path for legacy files; they are not migrated in place.

Full entries retain `potential_ini`, GMM parameters, and `eoemin_distribution/fit_density` with scalar `first_center` and `bin_width` attributes. No per-star ID or label arrays are stored. Evolution conventions and right-exclusive ID windows are unchanged. Full recomputations rebuild potentials; an unavailable skew-t ECUT is reported as a failure.

The library writes the existing HDF5 directly and is not crash-atomic; use one writer and a separate output path for validation. Mocked integration tests do not establish numerical agreement on real TNG galaxies. Thread limits and target-file parsing belong in the calling research script, not import-time library code.
