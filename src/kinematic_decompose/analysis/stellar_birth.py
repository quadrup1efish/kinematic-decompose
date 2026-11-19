"""Birth coordinates and same-energy circularity for a fixed stellar cohort."""

from dataclasses import dataclass

import numpy as np

from ..PyTNG import illustris_python as il
from ..mixture.util import STRUCTURES


BIRTH_FIELDS = ('ParticleIDs', 'GFM_StellarFormationTime', 'BirthPos',
                'BirthVel', 'GFM_InitialMass')


def analyze_tng_stellar_birth(base_path, seed_snap, subhalo_id, particle_ids,
                              labels, *, seed_axis, age_gyr, **options):
    """Read a seed subhalo and its main branch without refitting labels.

    Optional `load_potential` takes `(snap, progenitor_subfind_id)`; see
    `measure_stellar_birth` for the required membership and frame contract.
    """
    base = str(base_path)
    loader = options.pop('load_potential', None)
    fields = il.snapshot.loadSubhalo(base, seed_snap, subhalo_id, 'stars',
                                   fields=[key for key in BIRTH_FIELDS
                                           if key != 'BirthVel' or loader is not None])
    tree = il.sublink.loadTree(base, seed_snap, subhalo_id, onlyMPB=True,
                              treeName='SubLink_gal', fields=['SnapNum', 'SubfindID'])
    if tree is None or not len(tree['SnapNum']):
        raise ValueError('No SubLink main branch for the requested seed')
    order = np.argsort(tree['SnapNum'])
    snaps = tree['SnapNum'][order]
    subfind_ids = tree['SubfindID'][order]
    if (int(snaps[-1]), int(subfind_ids[-1])) != (seed_snap, subhalo_id):
        raise ValueError('Main branch does not end at the requested seed')
    headers = [il.groupcat.loadHeader(base, int(snap)) for snap in snaps]
    catalog = [il.groupcat.loadSingle(base, int(snap), subhaloID=int(subfind))
               for snap, subfind in zip(snaps, subfind_ids)]
    header = headers[-1]
    progenitors = dict(zip(map(int, snaps), map(int, subfind_ids)))
    return measure_stellar_birth(
        fields, particle_ids, labels, branch_snaps=snaps,
        branch_scale=[h['Time'] for h in headers],
        branch_position=[row['SubhaloPos'] for row in catalog],
        branch_velocity=([row['SubhaloVel'] for row in catalog] if loader is not None else None),
        age_gyr=age_gyr,
        hubble=header['HubbleParam'], box_size=header['BoxSize'], seed_axis=seed_axis,
        load_potential=None if loader is None else lambda snap: loader(snap, progenitors[snap]),
        **options)


@dataclass(frozen=True)
class BirthPotential:
    """Potential and radial grid in the centered, rotated physical frame.

    Rows of rotation are the frame axes in box coordinates (z is the spin
    axis). Do not provide a frame for an epoch with no resolved spin axis.
    """
    potential: object
    radii: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class StellarBirth:
    particle_ids: np.ndarray
    labels: np.ndarray
    scale_factor: np.ndarray
    time_gyr: np.ndarray
    initial_mass: np.ndarray
    position: np.ndarray
    radius: np.ndarray
    cylindrical_radius: np.ndarray
    energy: np.ndarray
    circularity: np.ndarray
    reference_snap: np.ndarray
    reference_dt_gyr: np.ndarray
    status: np.ndarray


@dataclass(frozen=True)
class CircularityMeasurement:
    energy: np.ndarray
    circularity: np.ndarray
    status: np.ndarray


def measure_circularity(position, velocity, frame):
    """Evaluate jz/jc(E) for centered box-axis vectors in kpc and km/s."""
    from ..gravity.kinematic_solver import circular_orbit_reference

    r, v = (np.asarray(x, dtype=float) for x in (position, velocity))
    rotation = np.asarray(frame.rotation, dtype=float)
    if r.ndim != 2 or r.shape[1:] != (3,) or v.shape != r.shape:
        raise ValueError('Expected aligned (N, 3) positions and velocities')
    if (rotation.shape != (3, 3) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
        raise ValueError('Invalid reference rotation')
    finite = np.isfinite(r).all(axis=1) & np.isfinite(v).all(axis=1)
    energy = np.full(len(r), np.nan)
    epsilon = np.full_like(energy, np.nan)
    status = np.full(len(r), 'invalid_energy', dtype='U24')
    if not finite.any():
        return CircularityMeasurement(energy, epsilon, status)
    r, v = r[finite] @ rotation.T, v[finite] @ rotation.T
    e = np.asarray(frame.potential.potential(r)) + .5 * np.sum(v * v, axis=1)
    egrid, jcgrid = circular_orbit_reference(frame.potential, np.asarray(frame.radii))
    energy[finite] = e
    status[np.isfinite(energy)] = 'outside_energy_grid'
    status[np.isfinite(energy) & (energy >= 0)] = 'unbound'
    good = np.isfinite(e) & (e < 0) & (e >= egrid[0]) & (e <= egrid[-1])
    rows = np.flatnonzero(finite)[good]
    jc = np.exp(np.interp(-np.log(-e[good]), -np.log(-egrid), np.log(jcgrid)))
    epsilon[rows] = np.cross(r[good], v[good])[:, 2] / jc
    status[rows] = 'ok'
    return CircularityMeasurement(energy, epsilon, status)


def _ids(values):
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0):
        raise ValueError('ParticleIDs must be exact nonnegative integers')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate ParticleIDs')
    return ids.astype(np.uint64, copy=False)


@dataclass(frozen=True)
class TNGCircularityFrame:
    reference: BirthPotential
    particle_ids: np.ndarray
    position: np.ndarray
    velocity: np.ndarray

    def measure(self, particle_ids):
        """Measure current circularity in the caller's exact ID order."""
        ids = _ids(particle_ids)
        order = np.argsort(self.particle_ids)
        idx = np.searchsorted(self.particle_ids[order], ids)
        if np.any(idx == len(order)) or not np.array_equal(self.particle_ids[order[idx]], ids):
            raise ValueError('Requested stellar IDs missing from reference subhalo')
        rows = order[idx]
        return measure_circularity(self.position[rows], self.velocity[rows], self.reference)


def build_tng_circularity_frame(base_path, snap, subhalo_id, *, min_spin_stars=10):
    """Build a subhalo potential in its inner stellar-spin frame, or return None.

    The origin and peculiar bulk velocity are SubhaloPos/SubhaloVel. Spin
    uses real stars within twice their measured half-mass radius. The count
    threshold is a sampling floor, not a test of rotational support.
    """
    from ..PyTNG.snapshot_loader import Snapshot
    from ..gravity.kinematic_solver import construct_galaxy_potential_model

    if not isinstance(min_spin_stars, (int, np.integer)) or min_spin_stars < 1:
        raise ValueError('min_spin_stars must be a positive integer')
    snapshot = Snapshot(str(base_path), snap)
    snapshot.load_particle(ID=subhalo_id, load_particle_fields={
        'star': ['Coordinates', 'Velocities', 'Masses', 'ParticleIDs', 'GFM_StellarFormationTime'],
        'dm': ['Coordinates', 'Velocities'],
        'gas': ['Coordinates', 'Velocities', 'Masses'],
    })
    snapshot.physical_units()
    snapshot.load_group_catalog(ID=subhalo_id)
    snapshot.GC_physical_units()
    catalog = snapshot.group_catalog
    snapshot.center(cen=catalog['SubhaloPos'], vel_cen=catalog['SubhaloVel'])
    galaxy = snapshot.container
    stars = galaxy.s
    if len(stars) < min_spin_stars:
        return None
    p, v, mass = (np.asarray(stars[key], dtype=float) for key in ('pos', 'vel', 'mass'))
    real = np.asarray(stars['aform']) > 0
    ids = _ids(np.asarray(stars['iord'])[real])
    r = np.linalg.norm(p, axis=1)
    valid = real & np.isfinite(r) & np.isfinite(mass) & (mass > 0)
    order = np.flatnonzero(valid)
    if len(order) < min_spin_stars:
        return None
    order = order[np.argsort(r[order])]
    cumulative = np.cumsum(mass[order])
    half = r[order[np.searchsorted(cumulative, cumulative[-1] / 2)]]
    inner = valid & (r <= 2 * half) & np.isfinite(v).all(axis=1)
    if np.count_nonzero(inner) < min_spin_stars:
        return None
    angular = np.cross(p[inner], v[inner]) * mass[inner, None]
    spin = angular.sum(axis=0)
    norm = np.linalg.norm(spin)
    if not np.isfinite(norm) or norm <= np.finfo(float).eps * np.linalg.norm(angular, axis=1).sum():
        return None
    z = spin / norm
    x = np.cross(np.eye(3)[np.argmin(np.abs(z))], z)
    x /= np.linalg.norm(x)
    rotation = np.stack((x, np.cross(z, x), z))
    radii = np.linalg.norm(np.asarray(galaxy['pos']), axis=1)
    positive = radii[np.isfinite(radii) & (radii > 0)]
    if not positive.size:
        return None
    edges = np.logspace(np.log10(.9 * positive.min()), np.log10(1.1 * positive.max()), 100)
    with galaxy.rotate(rotation):
        potential = construct_galaxy_potential_model(galaxy)
    return TNGCircularityFrame(BirthPotential(potential, .5 * (edges[:-1] + edges[1:]), rotation),
                               ids, p[real], v[real])


@dataclass(frozen=True)
class StellarBirthAndCurrent:
    birth: StellarBirth
    current: CircularityMeasurement


def analyze_tng_birth_and_current(base_path, seed_snap, subhalo_id, particle_ids, labels, *,
                                  age_gyr, main_branch_birth, max_reference_dt_gyr,
                                  load_frame=None):
    """Compute both epochs in input ID order, without assigning birth hosts.

    An optional ``load_frame(snap, subhalo_id)`` supplies prepared frames.
    The seed frame is reused when selected as a birth reference.
    """
    def get_frame(snap, sub):
        if load_frame is not None:
            return load_frame(snap, sub)
        return build_tng_circularity_frame(base_path, snap, sub)

    seed = get_frame(seed_snap, subhalo_id)
    if seed is None:
        raise ValueError('No resolved stellar-spin frame at the current snapshot')
    current = seed.measure(particle_ids)

    def reference(snap, sub):
        frame = seed if (snap, sub) == (seed_snap, subhalo_id) else get_frame(snap, sub)
        return None if frame is None else frame.reference

    birth = analyze_tng_stellar_birth(
        base_path, seed_snap, subhalo_id, particle_ids, labels,
        seed_axis=seed.reference.rotation[2], age_gyr=age_gyr,
        main_branch_birth=main_branch_birth, max_reference_dt_gyr=max_reference_dt_gyr,
        load_potential=reference)
    return StellarBirthAndCurrent(birth, current)


def measure_stellar_birth(fields, particle_ids, labels, *, branch_snaps,
                          branch_scale, branch_position, branch_velocity=None,
                          age_gyr, hubble, box_size, seed_axis,
                          main_branch_birth=None, load_potential=None,
                          max_reference_dt_gyr=None):
    """Measure births of seed-labelled real stars; input and output align by ID.

    `fields` are native TNG PartType4 arrays. Branch positions use ckpc/h;
    branch velocities, when needed, must already be peculiar km/s. `age_gyr(a)` is the
    simulation cosmology. `seed_axis` defines the fixed plane for R_birth.
    For circularity, `main_branch_birth` is a boolean *confirmed* membership
    mask in seed-ID order (False includes unresolved births). `load_potential`
    receives the nearest branch snapshot in cosmic time and returns its
    BirthPotential or None. A required maximum time gap prevents selecting a
    distant model. Frames are loaded once per snapshot, not once per star.
    """
    ids = _ids(particle_ids)
    labels = np.asarray(labels)
    if labels.shape != ids.shape or labels.dtype.kind not in 'iu' or not np.isin(labels, range(5)).all():
        raise ValueError('Expected one structure label (0–4) per seed ID')
    source = _ids(fields['ParticleIDs'])
    order = np.argsort(source)
    index = np.searchsorted(source[order], ids)
    if np.any(index == len(source)) or np.any(source[order[index]] != ids):
        raise ValueError('Seed stellar IDs missing from birth fields')
    rows = order[index]
    a, bp, mass = (np.asarray(fields[key], dtype=float)[rows]
                   for key in ('GFM_StellarFormationTime', 'BirthPos', 'GFM_InitialMass'))
    n = len(ids)
    if (a.shape != (n,) or mass.shape != (n,) or bp.shape != (n, 3)
            or not np.isfinite(a).all() or np.any(a <= 0)
            or not np.isfinite(mass).all() or np.any(mass <= 0)):
        raise ValueError('Select real stars with positive birth scale and initial mass')
    snaps, scales = np.asarray(branch_snaps), np.asarray(branch_scale, dtype=float)
    pos = np.asarray(branch_position, dtype=float)
    if (scales.ndim != 1 or len(scales) < 2 or snaps.shape != scales.shape
            or snaps.dtype.kind not in 'iu' or np.any(snaps < 0) or np.any(snaps[1:] <= snaps[:-1])
            or np.any(scales <= 0) or np.any(np.diff(scales) <= 0)
            or pos.shape != (len(scales), 3)
            or not all(np.isfinite(x).all() for x in (scales, pos))
            or not np.isfinite([hubble, box_size]).all() or min(hubble, box_size) <= 0):
        raise ValueError('Invalid or unordered branch positions or cosmology')
    t, bt = np.asarray(age_gyr(a), dtype=float), np.asarray(age_gyr(scales), dtype=float)
    if (t.shape != a.shape or bt.shape != scales.shape or not np.isfinite(t).all()
            or not np.isfinite(bt).all() or np.any(np.diff(bt) <= 0)):
        raise ValueError('Invalid cosmic ages')
    axis = np.asarray(seed_axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis.shape != (3,) or not np.isfinite(axis_norm) or axis_norm == 0:
        raise ValueError('A finite nonzero seed axis is required for cylindrical radius')
    axis = axis / axis_norm
    inside = (a >= scales[0]) & (a <= scales[-1])
    status = np.full(n, 'outside_branch', dtype='U24')
    status[inside] = 'no_birth_position'
    usable = inside & np.isfinite(bp).all(axis=1)
    status[usable] = 'host_unconfirmed'

    def minimum_image(delta):
        return (delta + box_size / 2) % box_size - box_size / 2

    path = np.vstack((pos[0], pos[0] + np.cumsum(minimum_image(np.diff(pos, axis=0)), axis=0)))
    center = np.column_stack([np.interp(t[usable], bt, x) for x in path.T])
    relative = np.full((n, 3), np.nan)
    relative[usable] = minimum_image(bp[usable] - center) * (a[usable, None] / hubble)
    radius = np.linalg.norm(relative, axis=1)
    cylindrical = np.linalg.norm(np.cross(relative, axis), axis=1)
    energy, epsilon, dt = (np.full(n, np.nan) for _ in range(3))
    ref = np.full(n, -1, dtype=int)
    if load_potential is not None:
        if 'BirthVel' not in fields:
            raise ValueError('Circularity requires BirthVel')
        bv = np.asarray(fields['BirthVel'], dtype=float)[rows]
        vel = np.asarray(branch_velocity, dtype=float)
        if bv.shape != (n, 3) or vel.shape != pos.shape or not np.isfinite(vel).all():
            raise ValueError('Circularity requires aligned BirthVel and finite branch velocities')
        member = np.asarray(main_branch_birth)
        if member.dtype.kind != 'b' or member.shape != (n,):
            raise ValueError('Circularity requires a confirmed main-branch birth mask')
        if max_reference_dt_gyr is None or not np.isfinite(max_reference_dt_gyr) or max_reference_dt_gyr < 0:
            raise ValueError('Circularity requires a finite maximum reference gap (Gyr)')
        status[usable & member] = 'no_birth_velocity'
        selected = np.flatnonzero(usable & member & np.isfinite(bv).all(axis=1))
        if len(selected):
            right = np.searchsorted(bt, t[selected]).clip(1, len(bt) - 1)
            left = right - 1
            nearest = np.where(t[selected] - bt[left] <= bt[right] - t[selected], left, right)
            ref[selected] = snaps[nearest]
            dt[selected] = bt[nearest] - t[selected]
            status[selected] = 'reference_too_far'
            for snap in np.unique(ref[selected]):
                take = selected[(ref[selected] == snap) & (np.abs(dt[selected]) <= max_reference_dt_gyr)]
                if not len(take):
                    continue
                status[take] = 'no_reference_frame'
                frame = load_potential(int(snap))
                if frame is None:
                    continue
                origin_v = np.column_stack([np.interp(t[take], bt, x) for x in vel.T])
                measured = measure_circularity(relative[take],
                                               bv[take] * np.sqrt(a[take, None]) - origin_v, frame)
                energy[take], epsilon[take], status[take] = (
                    measured.energy, measured.circularity, measured.status)
    else:
        status[usable] = 'position_only'
    return StellarBirth(ids, labels, a, t, mass * 1e10 / hubble, relative, radius,
                        cylindrical, energy, epsilon, ref, dt, status)


@dataclass(frozen=True)
class BirthSpatialHistory:
    """Binned maps; projection rows are Current XY/XZ, then Birth XY/XZ.

    Optional box-axis Birth XY/XZ rows come last. Column zero covers all
    ages; the other columns follow the lookback intervals.
    """
    seed_time_gyr: float
    lookback_edges_gyr: np.ndarray
    time_edges_gyr: np.ndarray  # Lookback edges of the fine SFH grid.
    radius_edges_pkpc: np.ndarray
    sfr_surface_density: np.ndarray  # Time by radius; Msun / yr / pkpc².
    projection_edges_pkpc: np.ndarray
    stellar_surface_density: np.ndarray  # Row, age interval, y, x; Msun / pkpc².
    total_mass: float
    sfh_mass: float
    time_bin_mass: np.ndarray
    projected_mass: np.ndarray

    @property
    def window_mass(self):
        return float(self.time_bin_mass.sum())


def compute_birth_spatial_history(birth_position, seed_position, birth_time, initial_mass,
                                  *, seed_time, lookback_edges, rotation,
                                  time_edges, radius_edges, projection_edges,
                                  show_box_axes=False):
    """Bin aligned positions (pkpc), cosmic ages (Gyr) and initial masses (Msun).

    Rotation rows are fixed seed-frame axes. `lookback_edges` defines age
    slices, `time_edges` the finer lookback grid, `radius_edges` the SFH
    annuli, and `projection_edges` both axes of the square projected maps.
    """
    born, seed, time, mass, rot = map(np.asarray, (
        birth_position, seed_position, birth_time, initial_mass, rotation))
    edges, sfh_t, radii, grid = (_bin_edges(x) for x in (
        lookback_edges, time_edges, radius_edges, projection_edges))
    n = mass.size
    if (born.shape != (n, 3) or seed.shape != born.shape or time.shape != (n,)
            or mass.shape != (n,) or not n or not np.all(np.isfinite(mass) & (mass > 0))
            or not np.isfinite(seed_time) or min(edges[0], sfh_t[0], radii[0]) < 0
            or rot.shape != (3, 3) or not np.allclose(rot @ rot.T, np.eye(3))
            or not np.isclose(np.linalg.det(rot), 1)):
        raise ValueError('Invalid positions, masses, time bins, or rotation')

    lb = seed_time - time
    aligned = born @ rot.T
    radius = np.hypot(aligned[:, 0], aligned[:, 1])
    in_sfh = ((lb >= sfh_t[0]) & (lb <= sfh_t[-1])
              & (radius >= radii[0]) & (radius <= radii[-1]))
    sfr = np.histogram2d(lb[in_sfh], radius[in_sfh], bins=(sfh_t, radii), weights=mass[in_sfh])[0]
    sfr_mass = float(sfr.sum())
    sfr /= np.diff(sfh_t)[:, None] * 1e9 * np.pi * np.diff(radii**2)[None, :]

    index = np.searchsorted(edges, lb, side='right') - 1
    index[lb == edges[-1]] = len(edges) - 2
    samples = [np.ones(n, dtype=bool)] + [index == j for j in range(len(edges) - 1)]
    widths = np.diff(grid)
    # Keep uniform-grid arithmetic identical to the original projection.
    pix_area = widths[0]**2 if np.allclose(widths, widths[0], rtol=1e-12, atol=0) else np.outer(widths, widths)
    frames = [seed @ rot.T, aligned] + ([born] if show_box_axes else [])
    binned = np.array([[
        np.histogram2d(xyz[take, 0], xyz[take, dim], bins=(grid, grid), weights=mass[take])[0].T
        for take in samples] for xyz in frames for dim in (1, 2)])
    maps = binned / pix_area
    projected_mass = (maps.sum(axis=(-1, -2)) * pix_area if np.ndim(pix_area) == 0
                      else binned.sum(axis=(-1, -2)))
    return BirthSpatialHistory(float(seed_time), edges, sfh_t, radii, sfr, grid, maps,
                               float(mass.sum()), sfr_mass,
                               np.array([mass[take].sum() for take in samples[1:]], dtype=float),
                               projected_mass)


def _bin_edges(edges):
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError('Bin edges must be finite and strictly increasing')
    return edges


@dataclass(frozen=True)
class BirthDistributions:
    structures: tuple
    panels: tuple  # Per structure: (values, x edges, y edges) per panel.
    coverage: dict


@dataclass(frozen=True)
class BirthCircularityHistory:
    seed_time_gyr: float
    time_edges_gyr: np.ndarray  # Lookback time from the chosen snapshot.
    circularity_edges: np.ndarray
    sfr_per_circularity: np.ndarray  # Time by circularity; Msun / yr / unit circularity.
    window_mass: float
    valid_mass: float
    plotted_mass: float


def compute_birth_circularity_history(birth_time, initial_mass, circularity, status, *,
                                      seed_time, time_edges, circularity_edges,
                                      valid_status=('ok',)):
    """Bin one fixed structure. Only finite circularities with status 'ok' enter.

    Inputs are aligned, with cosmic ages in Gyr and initial masses in Msun.
    Coverage masses distinguish invalid circularities from those outside the
    display range. Neither category is clipped or renormalized away.
    """
    t, mass, eps = (np.asarray(x, dtype=float) for x in (birth_time, initial_mass, circularity))
    status = np.asarray(status)
    te, ce = map(_bin_edges, (time_edges, circularity_edges))
    if (mass.ndim != 1 or not mass.size or any(x.shape != mass.shape for x in (t, eps, status))
            or not np.all(np.isfinite(mass) & (mass > 0)) or not np.isfinite(seed_time) or te[0] < 0):
        raise ValueError('Provide aligned birth data, positive masses and a valid lookback grid')
    lb = seed_time - t
    window = (lb >= te[0]) & (lb <= te[-1])
    valid = window & np.isin(status, valid_status) & np.isfinite(eps)
    formed, _, _ = np.histogram2d(lb[valid], eps[valid], bins=(te, ce), weights=mass[valid])
    rate = formed / (np.diff(te)[:, None] * 1e9 * np.diff(ce)[None, :])
    return BirthCircularityHistory(float(seed_time), te, ce, rate,
                                   float(mass[window].sum()), float(mass[valid].sum()),
                                   float(formed.sum()))


def compute_birth_distributions(birth, *, time_edges, radius_edges, circularity_edges,
                                structures=('bulge', 'disk', 'halo'), seed_circularity=None):
    """Bin birth radii and circularities; optional seed circularities follow birth IDs."""
    time_edges, radius_edges, circularity_edges = map(_bin_edges, (time_edges, radius_edges, circularity_edges))
    matched = seed_circularity is not None
    if matched:
        seed_circularity = np.asarray(seed_circularity, dtype=float)
        if seed_circularity.shape != birth.circularity.shape:
            raise ValueError('Seed circularity must align with birth particle IDs')
    masses = birth.initial_mass
    items = [(name, np.isin(birth.labels, STRUCTURES[name])) for name in structures]
    items = [(name, mask) for name, mask in items if mask.any()]
    if not items:
        raise ValueError('No stars in the requested structures')
    panels, coverage = [], {}
    sources = [(birth.time_gyr, birth.cylindrical_radius, time_edges, radius_edges),
               (birth.time_gyr, birth.circularity, time_edges, circularity_edges)]
    if matched:
        sources.append((birth.circularity, seed_circularity, circularity_edges, circularity_edges))
    for name, member in items:
        histograms = []
        for x, y, xe, ye in sources:
            good = member & np.isfinite(x) & np.isfinite(y)
            h, _, _ = np.histogram2d(x[good], y[good], bins=(xe, ye), weights=masses[good])
            histograms.append((h, xe, ye))
        coverage[name] = dict(total_mass=float(masses[member].sum()),
                              radius_mass=float(masses[member & np.isfinite(birth.radius)].sum()),
                              circularity_mass=float(masses[member & (birth.status == 'ok')].sum()),
                              plotted_mass=tuple(float(h.sum()) for h, _, _ in histograms))
        panels.append(tuple((h / (1e9 * np.diff(xe)[:, None]) if i < 2 else h, xe, ye)
                            for i, (h, xe, ye) in enumerate(histograms)))
    return BirthDistributions(tuple(name for name, _ in items), tuple(panels), coverage)
