"""Single-snapshot star formation history and light structural properties."""

import gc

from ..PyTNG.snapshot_loader import Snapshot
from .sfh import get_cumsum_sf, get_sfr

EMPTY_STELLAR_TRACE = object()


def compute_sfh_trace(basePath, snapNum, subID, *, include_gas=False):
    """Measure one subhalo without kinematic decomposition or output writes.

    Return (structure, sfh, snapshot), or (EMPTY_STELLAR_TRACE, dm_info)
    if no stellar particles are loaded. The returned snapshot is face-on.
    SFH here uses all loaded stars, not a terminal-member selection.
    """

    snap = Snapshot(basePath, snapNum)
    fields = {
        "star": ['Coordinates', 'Velocities', 'Masses',
                 'ParticleIDs', 'GFM_StellarFormationTime',
                 'GFM_InitialMass'],
        "dm": ['Coordinates', 'Velocities']}
    if include_gas:
        fields['gas'] = ['Coordinates', 'Velocities', 'Masses']
    snap.load_particle(ID=subID, load_particle_fields=fields)
    n_particles = len(snap.s['pos'])
    print(f"  [TRACE] n_particles={n_particles:.2e}")

    if n_particles == 0:
        snap.physical_units()
        n_dm = len(snap.dm['pos'])
        dm_info = {'n_dm_particles': n_dm,
                   'dm_mass': float(snap.container.properties['mDM'] * n_dm)}
        del snap
        gc.collect()
        return (EMPTY_STELLAR_TRACE, dm_info)

    snap.physical_units()
    snap.load_group_catalog(ID=subID)
    snap.GC_physical_units()
    snap.center(cen=snap.group_catalog['SubhaloPos'])
    snap.faceon(align_with='star',
                range=[3 * snap.properties['eps'], 5 * snap.s.r50],
                as_context=False)
    galaxy = snap.container

    n_dm_particles = len(galaxy.dm['pos'])
    dm_mass = float(galaxy.properties['mDM'] * n_dm_particles)

    mass_init = galaxy.s['GFM_InitialMass']
    mass_cur = galaxy.s['mass']
    tform = galaxy.s['tform']

    _, cumsum_sf = get_cumsum_sf(mass_init, tform)
    _, sfr = get_sfr(mass_init, tform)

    sfh_dict = {
        'cumsum_sf': cumsum_sf,
        'SFR': sfr,
        'total_mass': mass_cur.sum(),
        'n_particles': n_particles,
    }
    structure_dict_light = {
        'star': {
            'mass': mass_cur.sum(),
            'r50': galaxy.s.r50,
            'vel_disp': galaxy.s.vel_disp,
            'n_particles': n_particles,
        },
        'dm': {
            'n_particles': n_dm_particles,
            'mass': dm_mass,
        }
    }

    print(f'  [TRACE] SFH done: mass={mass_cur.sum():.2e}')
    return structure_dict_light, sfh_dict, snap
