# Stellar birth histories by seed structure

This analysis retains the per-star labels from an existing decomposition at
any seed snapshot. Join `ParticleIDs` to the seed snapshot's `BirthPos`,
`GFM_StellarFormationTime` and `GFM_InitialMass`. Only real stars
(positive formation scale factor) are accepted. IDs remain integers. No Monte
Carlo tracer scan or new decomposition is needed.

## Coordinates and same-energy circularity

`analysis.stellar_birth.analyze_tng_stellar_birth` reads the birth records and
SubLink_gal main branch; `measure_stellar_birth` accepts preloaded arrays.
Supply a cosmic age function in Gyr for the simulation cosmology and a seed
axis in simulation-box coordinates. The main-branch `SubhaloPos` is unwrapped
and interpolated in cosmic time. The birth offset is wrapped to the nearest
periodic image and multiplied by `a_birth/h`; `cylindrical_radius` uses the
fixed seed axis, while `radius` is the 3D distance. No center is extrapolated.
A birth radius is not an assignment to a birth host.

Position-only analysis needs neither `BirthVel` nor branch velocities.
Use `analysis.stellar_birth.analyze_tng_birth_and_current` to measure both
epochs for the same input IDs. It builds reference frames with
`build_tng_circularity_frame`, reusing the current frame if it is also a birth
reference. The builder uses `SubhaloPos` and peculiar `SubhaloVel`, with the
spin axis from real stars within twice their half-mass radius. This bulk
velocity convention may differ from earlier exploratory plots. A custom
`load_frame(snap, subhalo_id)` can supply prepared frames. The joint entry
point does not assign birth hosts; pass a separately established membership
mask. For a known external host, `measure_circularity` accepts vectors
already centered on that host in physical kpc and peculiar km/s.

For circularity, supply `main_branch_birth`, a boolean mask aligned with the
input seed IDs, based on independent evidence of birth host membership. Also
supply `load_potential(snap, progenitor_subfind_id)` and
`max_reference_dt_gyr`. The callback returns a `BirthPotential` (potential in
physical kpc and km/s, circular-grid radii in kpc, and a right-handed
box-to-frame rotation), or `None` if no trustworthy frame is available.
The potential must be centered on its reference progenitor and built in the
same orientation as that rotation. Provide only resolved spin frames. The
closest main-branch snapshot in cosmic time is used, within the specified gap.
The callback runs once per used snapshot; potential construction and caching
are caller responsibilities. Host membership is not inferred by this module.

`BirthVel` records the parent gas velocity; multiply by `sqrt(a_birth)` and
subtract the interpolated peculiar `SubhaloVel` before rotating. Evaluate
`E = Phi(r) + |v|²/2` in the full reference potential, then compute
`(r × v)_z / jc(E)` using the same azimuthally averaged circular-orbit reference
as the decomposition. This is an estimate using a nearby snapshot potential,
not the exact time-dependent potential at formation. Unconfirmed hosts,
missing/distant reference frames, and energies outside the bound reference
grid yield NaN rather than zero or a clipped value. Output includes a status
and the signed reference-time offset in Gyr for each star.

`visualization.stellar_birth.visualize_stellar_birth` consumes the returned
arrays without reading snapshots. Supply common time, radius and circularity
bin edges, and optionally seed circularity in the same ID order. Initial
formed mass divided by the time-bin width colors the first two columns; the
birth-vs-seed panel shows formed mass per bin pair. This is the contribution
from stars selected at the seed snapshot, not the total progenitor SFR.
Coverage includes the available and actually plotted formed mass per structure.
For independent numerical results, call `analysis.stellar_birth.compute_birth_distributions`
and pass its result to `visualization.stellar_birth.plot_birth_distributions`.

For a GalaxyPose-style position-only view, call
`visualization.stellar_birth.plot_birth_spatial_history` once per structure.
The upper panel is an annulus-area-normalized SFR map; the lower grid shows
XY/XZ projections for all ages and the same birth-time bins, comparing seed
and birth positions in the same fixed seed orientation. Set `show_box_axes=True`
to add another projection of the same centered birth positions along the
simulation-box axes. Neither view is a time-varying birth alignment.
Pass `virial_history=(cosmic_time, R200c_pkpc)` for the FoF scale and
`merger_times` for selected SubLink descendant cosmic ages. The same snapshot
is marked only once, even if multiple progenitors join at that snapshot.
Returned masses distinguish the full cohort, the time window and the plotted
area. The All ages column is not restricted to the time window.
The upper map uses the GalaxyPose `SFHPlotConfig` defaults: 420 uniform time
bins and 90 uniform radial bins, with no smoothing. `time_bins` and
`radial_bins` override these independently of the projection columns. Pass
`redshift_at_time` to label the top axis and projection columns in redshift
while retaining the original time intervals and per-year SFR normalization.
`Current` means the selected analysis snapshot; the projection colorbar
uses initial-mass-weighted stellar surface density ($\Sigma_*$).
To compute or cache the maps separately, use
`analysis.stellar_birth.compute_birth_spatial_history` with explicit fine
lookback-time, radius and projection grids, then pass the result to
`visualization.stellar_birth.plot_birth_history`. This renderer returns a
figure without rebinning; use the result object to inspect the mass coverage.
To replace the upper radial panel with birth circularity, bin `birth.circularity`
and `birth.status` with `analysis.stellar_birth.compute_birth_circularity_history`
for the same ID-aligned cohort, seed epoch and fine lookback grid. Pass that
result as `circularity_history=` to `plot_birth_history` and set
`top_panel='jz/jc'` (`top_panel='R'` is the default). The map shows formed
mass divided by elapsed years and circularity-bin width, not by annulus area.
Only finite values with status `ok` enter. The result reports window, valid,
and plotted mass without clipping circularities or renormalizing for missing
data. The projections still include the full spatial cohort, and merger marks
are available; do not pass a physical `virial_history` for this axis.
See `tests/example_stellar_birth_history.py --help` for a six-interval redraw
from cached positions and merger times.

Coordinate-frame inspiration: GalaxyPose (`GalaxySimAnalytics/GalaxyPose`).
Field definitions: IllustrisTNG Data Specifications, PartType4 and Subhalos.
