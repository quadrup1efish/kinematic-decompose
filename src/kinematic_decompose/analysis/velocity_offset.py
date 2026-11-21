"""Baryon-total velocity offset of a galaxy.

    s_b = |V_baryons - V_total| / V_c(R90),    V_c = sqrt(G M / R)

``V_baryons`` is the mass-weighted 3D velocity vector of all baryons (stars and
gas) inside the aperture, ``V_total`` the same for every particle type present,
including dark matter. ``R90`` encloses 90% of the stellar mass inside the
aperture. Distances are 3D (spherical aperture), not projected.
"""

import numpy as np

# pynbody's gravitational constant in kpc (km/s)^2 / Msol;
# the test pins this to pynbody.units.G so both stay consistent.
G_KPC_KMS2_MSOL = 4.300939179648743e-06
BARYONIC_SPECIES = ('star', 'gas')
CIRCULAR_MASS_APERTURES = ('r90', 'aperture')


def _as_block(name, block):
    try:
        pos = np.asarray(block['pos'], dtype=float)
        vel = np.asarray(block['vel'], dtype=float)
        mass = np.asarray(block['mass'], dtype=float)
    except KeyError as error:
        raise ValueError(f'{name}: missing {error.args[0]!r}') from None
    n = len(mass)
    if pos.shape != (n, 3) or vel.shape != (n, 3):
        raise ValueError(f'{name}: pos and vel must both have shape (N, 3)')
    if not (np.isfinite(pos).all() and np.isfinite(vel).all()
            and np.isfinite(mass).all() and (mass >= 0).all()):
        raise ValueError(f'{name}: arrays must be finite with non-negative masses')
    return pos, vel, mass


def _mass_weighted_velocity(velocity, mass):
    total = mass.sum()
    if total <= 0:
        raise ValueError('mass-weighted velocity needs a positive total mass')
    return (velocity * mass[:, None]).sum(axis=0) / total


def _enclosing_radius(radii, mass, fraction):
    order = np.argsort(radii, kind='stable')
    cumulative = np.cumsum(mass[order])
    if cumulative[-1] <= 0:
        return np.nan
    index = np.searchsorted(cumulative, fraction * cumulative[-1], side='left')
    return float(radii[order][min(index, len(radii) - 1)])


def baryonic_velocity_offset(particles, *, aperture=20.0, circular_mass='r90'):
    """Measure ``s_b`` from arrays already in kpc, km/s and Msol.

    ``particles`` maps a species name to ``{'pos', 'vel', 'mass'}``. Species
    listed in ``BARYONIC_SPECIES`` form the baryon sample; every species forms
    the total sample. ``circular_mass='r90'`` uses the mass enclosed within R90,
    the default; ``'aperture'`` uses all mass inside the aperture.
    """
    aperture = float(aperture)
    if not np.isfinite(aperture) or aperture <= 0:
        raise ValueError('aperture must be positive and finite')
    if circular_mass not in CIRCULAR_MASS_APERTURES:
        raise ValueError(f'circular_mass must be one of {CIRCULAR_MASS_APERTURES}')
    if not particles:
        raise ValueError('particles must not be empty')

    inside = {}
    for name, block in particles.items():
        pos, vel, mass = _as_block(name, block)
        radius = np.linalg.norm(pos, axis=1)
        keep = radius <= aperture
        inside[name] = (radius[keep], vel[keep], mass[keep])
    baryon_names = [name for name in inside if name in BARYONIC_SPECIES]
    if not baryon_names:
        raise ValueError('particles must include a star or gas species')

    def merged(names):
        return tuple(np.concatenate([inside[name][i] for name in names])
                     for i in range(3))

    baryon_radius, baryon_velocity, baryon_mass = merged(baryon_names)
    total_radius, total_velocity, total_mass_all = merged(list(inside))

    result = {
        'aperture': aperture,
        'circular_mass': circular_mass,
        'n_particles': {name: len(inside[name][2]) for name in inside},
        'mass_by_species': {name: float(inside[name][2].sum()) for name in inside},
        'mass_baryon': float(baryon_mass.sum()),
        'mass_total': float(total_mass_all.sum()),
    }
    if 'star' not in inside or inside['star'][2].sum() <= 0:
        result.update(valid=False, s_b=np.nan, r90=np.nan, v_circ=np.nan,
                      v_baryon=np.full(3, np.nan), v_total=np.full(3, np.nan))
        return result

    r90 = _enclosing_radius(*inside['star'][::2], fraction=0.9)
    if circular_mass == 'r90':
        enclosed = total_mass_all[total_radius <= r90].sum()
    else:
        enclosed = total_mass_all.sum()
    v_circ = float(np.sqrt(G_KPC_KMS2_MSOL * enclosed / r90))

    v_baryon = _mass_weighted_velocity(baryon_velocity, baryon_mass)
    v_total = _mass_weighted_velocity(total_velocity, total_mass_all)
    offset = v_baryon - v_total
    result.update(
        valid=True,
        s_b=float(np.linalg.norm(offset) / v_circ),
        r90=r90,
        mass_enclosed_r90=float(enclosed),
        v_circ=v_circ,
        v_baryon=v_baryon,
        v_total=v_total,
        velocity_offset=offset,
    )
    return result


def _physical(array, unit):
    """Return a plain float array in ``unit`` when units are attached."""
    if hasattr(array, 'in_units'):
        return np.asarray(array.in_units(unit), dtype=float)
    return np.asarray(array, dtype=float)


_FAMILY_NAMES = {'star': 's', 'gas': 'g', 'dm': 'dm', 'bh': 'bh'}


def _family_arrays(family):
    return {'pos': _physical(family['pos'], 'kpc'),
            'vel': _physical(family['vel'], 'km s^-1'),
            'mass': _physical(family['mass'], 'Msol')}


def particles_from_container(galaxy, species=('star', 'gas', 'dm')):
    """Extract physical-unit particle blocks from a pynbody galaxy view.

    The container must already be centred. Species with no particles are
    skipped, so callers can pass the same tuple for every galaxy.
    """
    missing = set(species) - set(_FAMILY_NAMES)
    if missing:
        raise ValueError(f'unknown species: {sorted(missing)}')
    blocks = {}
    for name in species:
        family = getattr(galaxy, _FAMILY_NAMES[name])
        if len(family['pos']):
            blocks[name] = _family_arrays(family)
    return blocks


def measure_streamed_dm_offset(galaxy, base_path, snap_num, subfind_id, *,
                               block_size=262144):
    """Measure s_b before rotation, reusing the centered DM coordinate array."""
    from ..PyTNG.illustris_python import snapshot

    baryons = baryonic_velocity_offset(particles_from_container(galaxy, ('star', 'gas')))
    r90 = baryons['r90']
    if not baryons['valid'] or not r90 > 0:
        return np.nan
    pos = galaxy.dm['pos']
    if not len(pos):
        return baryons['s_b']
    subset = snapshot.getSnapOffsets(base_path, snap_num, subfind_id, 'Subhalo')
    if int(subset['lenType'][1]) != len(pos):
        raise ValueError('DM coordinates and snapshot member interval differ')
    start = int(subset['offsetType'][1])
    scale = np.sqrt(galaxy.properties['a'])
    mass = _physical(galaxy.dm['mass'][:1], 'Msol')[0]
    n20 = n90 = 0
    velocity_sum = np.zeros(3)
    for lo in range(0, len(pos), block_size):
        hi = min(lo + block_size, len(pos))
        radius = np.linalg.norm(_physical(pos[lo:hi], 'kpc'), axis=1)
        inside = radius <= baryons['aperture']
        count = np.count_nonzero(inside)
        n90 += np.count_nonzero(radius <= r90)
        if count:
            subset['offsetType'][1] = start + lo
            subset['lenType'][1] = hi - lo
            vel = snapshot.loadSubset(base_path, snap_num, 'dm', ['Velocities'],
                                      subset, float32=True)
            velocity_sum += vel[inside].sum(axis=0, dtype=np.float64) * scale
            n20 += count
    mb = baryons['mass_total']
    vb = baryons['v_baryon']
    vt = (mb * vb + mass * velocity_sum) / (mb + n20 * mass)
    vc = np.sqrt(G_KPC_KMS2_MSOL * (baryons['mass_enclosed_r90'] + n90 * mass) / r90)
    return float(np.linalg.norm(vb - vt) / vc)


def measure_baryon_velocity_offset(base_path, snap_num, subfind_id, *,
                                   aperture=20.0, circular_mass='r90',
                                   species=('star', 'gas', 'dm')):
    """Load one TNG subhalo, centre it, and measure ``s_b``.

    The galaxy is centred on its group-catalog SubhaloPos and its bulk velocity
    (the standard library convention), so radii are measured from that centre.
    Black holes are excluded by default: TNG stores their coordinates relative
    to the host gas cell, so they would need a host-cell offset first.
    """
    from ..PyTNG.snapshot_loader import Snapshot

    fields = {'star': ['Coordinates', 'Velocities', 'Masses'],
              'gas': ['Coordinates', 'Velocities', 'Masses'],
              'dm': ['Coordinates', 'Velocities'],
              'bh': ['Coordinates', 'Velocities', 'Masses']}
    missing = set(species) - set(fields)
    if missing:
        raise ValueError(f'unknown species: {sorted(missing)}')
    snapshot = Snapshot(str(base_path), int(snap_num))
    snapshot.load_particle(ID=subfind_id,
                           load_particle_fields={name: fields[name] for name in species})
    snapshot.physical_units()
    snapshot.load_group_catalog(ID=subfind_id)
    snapshot.GC_physical_units()
    snapshot.center(cen=snapshot.group_catalog['SubhaloPos'])
    galaxy = snapshot.container
    particles = particles_from_container(galaxy, species)
    result = baryonic_velocity_offset(particles, aperture=aperture,
                                      circular_mass=circular_mass)
    result.update(snap_num=int(snap_num), subfind_id=int(subfind_id))
    return result
