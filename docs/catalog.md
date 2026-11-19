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
Replace the example IDs and paths with your targets. By default, the example writes `TNG50-1_kinematic_catalog_flexible_ecut.h5` in the current directory. Pass `h5_path` explicitly to choose a different name or location; the parent directory must exist. Set `fresh=False` (the default) to resume; `fresh=True` refuses to replace an existing file. `overwrite=True` recomputes selected objects. `counts` is `(computed, skipped, errors)` for independent snapshots. A failed main branch remains incomplete; the combined runner reports trace failures separately and continues with the remaining targets. In `99/<z0_subfind_id>`, `_traceback_error` records the latest failure stage and reason, and `_traceback_failed_snap` and `_traceback_failed_subfind_id` identify the failed frame (`-1` when no frame was reached). A successful retry clears these fields. Shared output errors still propagate. `run_traceback` handles one main branch. `build_tree(base_path, z0_targets, h5_path)` writes the optional tree table.

Traceback derives cosmic ages (Gyr) from the TNG group catalog headers and reuses them across targets in one call. By default, each traceback computes origin IDs in memory using group catalog Header scale factors. To reuse existing results, pass `stellar_assembly_dir` to `build_catalog` or `run_traceback`. This reads `<z0_subfind_id>_stellar_assembly.pkl` from that directory without rewriting it. Only use trusted pickle files from the same simulation run. A missing cache is reported as a failed target in a batch (and raises in a direct `run_traceback` call); it is not silently recomputed. Assembly data are obtained before the HDF5 completion check. Independent snapshot targets do not use this option.

Both main-branch tracing and independent targets use the 14-snapshot grid `(13, 17, 21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99)`. New trace catalogs store the corresponding cosmic ages in `/time_14`; the SFH bin edges remain in `/time_140`. Use a new output file rather than appending to a 12-snapshot trace catalog. The existing 1,000-stellar-particle threshold still determines whether a snapshot gets a full decomposition or only a lightweight SFH.

For a nonempty lightweight frame, the next full decomposition backfills `evolution/<component>/formed_mass_fraction_backfill`. At the earlier frame's cosmic age, it divides the initial mass formed by that age in the component by the initial mass formed by that age in all stars of the **same in-situ cohort** present in the full frame. Component labels come from that full frame, not from the earlier frame or from birth. Sum the cold and warm fractions for the disk. The arrays align with `/time_14`; `evolution/star/formed_mass_fraction_source_snap` records the reference snapshot (`-1` if not backfilled). Missing frames or frames without a usable later cohort stay `NaN`. This does not change snapshot `mass`, SFR, or `cumsum_sf`.

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

Full entries retain `potential_ini`, GMM parameters, and `eoemin_distribution/fit_density` with scalar `first_center` and `bin_width` attributes. No per-star ID or label arrays are stored. Assembly increments use `(previous_snapshot, current_snapshot]` (the first window starts at bucket 0), with separate windows for total and component SFHs. SFHs still bin by stellar formation time, not by the assembly bucket. The rightmost SFH edge covers the cosmic age at snapshot 99; interior bin edges are unchanged. Use a fresh output to recompute histories written with the old right-exclusive windows. Full recomputations rebuild potentials. If the skew-t estimator returns `None`, the catalog uses the legacy 1 kpc potential-based cut (sampled on the softening-limited radius grid) and records `_ecut_method=potential_1kpc_fallback`.

The library writes the existing HDF5 directly and is not crash-atomic; use one writer and a separate output path for validation. Mocked integration tests do not establish numerical agreement on real TNG galaxies. Thread limits and target-file parsing belong in the calling research script, not import-time library code.
