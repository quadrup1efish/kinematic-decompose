# Stellar-structure tracer history

Decompose a galaxy at a user-selected **seed snapshot** (not necessarily z=0).
Freeze the set of TNG Monte Carlo tracers carried by its real stars at that
snapshot. Follow their `TracerID`s into earlier snapshots, then resolve the
current `ParentID`s to gas, star, wind, or black-hole positions. Earlier
positions may be outside the main progenitor and its host halo. The seed
labels remain fixed; the figure does not show a new decomposition at each z.

The first version plots a grid of seed structures × redshifts, with one mark per
resolved tracer, colored by the parent's phase at the plotted snapshot. It
borrows the Lagrangian comparison from Obreja et al. (2013), Fig. 10 (ApJ 763,
26), but does not show the all-baryon environment or hot/cold gas subdivision.
Marks are not unique parents, stellar masses or exact mass weights. Missing
tracers and unresolved parents are reported separately, not interpreted as
physical disappearance. Zero-tracer seed stars count toward the coverage
denominator. `disk` combines the existing cold and warm disk labels and does
not include the counter-rotating disk.

## Small pilot

Use the complete TNG snapshots (including `PartType3`) for the seed and each
requested epoch. A local cutout containing only coordinates, velocities and
masses cannot supply the stellar IDs or tracer/parent history. The example
preflights the snapshots before either decomposition or tracing. It uses the
project's configured `BASEPATH` and unchanged decomposition pipeline unless
`--seed-particles` is supplied. This option accepts an NPZ with scalars `run`,
`seed_snap`, `subhalo_id`, and aligned `star_ids` (integer) and `star_labels`
(decomposition labels 0–4) for actual stars. Use this path to preserve the
exact memberships of an existing decomposition rather than refitting it. A
catalog containing only aggregate component masses is not sufficient.

Prepare a JSON object keyed by snapshot number, with `[x, y, z]` centers in
native `ckpc/h` (typically the `SubhaloPos` of the main progenitor at each
epoch). Supply those positions explicitly; the example does not choose a
merger-tree branch for you. Projections use fixed simulation-box axes and
same-sized square panels, with minimum-image periodic wrapping. The default
axes use physical kpc; `--comoving` switches to `ckpc/h`. The map has no
line-of-sight depth cut. The halo-centric center serves only to display the
results; it does not restrict the global parent lookup.

Example for full TNG50-1 data (provide your corresponding `centers.json`):

```sh
env -u PYTHONPATH MPLBACKEND=Agg OMP_NUM_THREADS=1 .venv/bin/python \
  tests/example_tracer_history.py --run TNG50-1 --seed 21 --subhalo 78482 \
  --snapshots 13 17 21 --centers centers.json --half-width 100 \
  --cache tracer-history.npz --figure tracer-history.png
```

For redraws use `--plot-only` with the same `--cache` and `--centers`; changing
`--structures` or `--half-width` then does not read any simulation data. The
NPZ contains run/seed provenance; the seed star IDs and hard labels; the
tracer IDs and their indices into the seed-star arrays; and, per snapshot,
`redshift`, `scale_factor`, `hubble`, `box_size`, `parent_ids`, `tracer_found`,
`parent_found`, `phase` and native absolute `coordinates`. `parent_ids` is
valid only under `tracer_found`; unresolved coordinates are NaN. No full
particle arrays, masses, or thermodynamic fields are cached. Keep the
centers JSON with the result to reproduce the same framing. This is a pilot
cache, written after all requested snapshots finish, not a batch checkpoint.

Library entry points: `PyTNG.tracer.seed_decomposed_galaxy`,
`PyTNG.tracer.seed_tracer_cohort`, `PyTNG.tracer.trace_snapshot`, and
`kinematic_decompose.visualization.tracer.visualize_tracer_history`. I/O is blockwise and bounded;
without an ID index, scanning full snapshots can be expensive. Test a single
galaxy and a few redshifts before increasing the sample.
