"""Measure s_b from a local cutout (no full snapshot or group catalogue).

TNG stores Velocities as km/s * a^(1/2); the loader converts them to physical
km/s. Reading the raw HDF5 arrays without that factor inflates every velocity
by 1/sqrt(a) and, because R90 and V_c stay unchanged, inflates s_b too.

The catalogue SubhaloVel / SubhaloVelDisp values are the independent check that
the loaded velocities are physical: at snap 21 (a = 0.1997) the raw-file means
would be ~2.24x larger than the catalogue values.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.analysis.velocity_offset import (
    baryonic_velocity_offset, particles_from_container,
)

DEFAULT_CUTOUT = Path.home() / 'sims.TNG/TNG50-1/cutouts'
DEFAULT_BASE = Path.home() / 'sims.TNG/TNG50-1/output'


def measure(snap_num, subfind_id, cutout_dir=DEFAULT_CUTOUT, base_path=DEFAULT_BASE,
            aperture=20.0, circular_mass='r90'):
    """Centre on the cutout catalogue's SubhaloPos, then measure s_b."""
    catalog = Path(cutout_dir) / f'snapNum_{snap_num}_subID_{subfind_id}.json'
    meta = json.loads(catalog.read_text())
    center = np.array([meta[f'pos_{axis}'] for axis in 'xyz'])
    snapshot = Snapshot(str(base_path), snap_num, header_source='cutout')
    galaxy = snapshot.load_cutout(ID=subfind_id, snapNum=snap_num,
                                  cutout_dir=str(cutout_dir))
    galaxy.translate(-center)          # native ckpc/h, box frame
    snapshot.physical_units()
    particles = particles_from_container(galaxy)
    result = baryonic_velocity_offset(particles, aperture=aperture,
                                      circular_mass=circular_mass)
    result.update(meta=meta, hubble=float(galaxy.properties['h']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snap', type=int, default=21)
    parser.add_argument('--subfind-id', type=int, default=78482)
    parser.add_argument('--aperture', type=float, default=20.0,
                        help='3D aperture in physical kpc')
    args = parser.parse_args()

    results = {}
    for mode in ('r90', 'aperture'):
        results[mode] = measure(args.snap, args.subfind_id,
                                aperture=args.aperture, circular_mass=mode)
    first = results['r90']
    meta = first['meta']
    to_msol = 1e10 / first['hubble']

    print(f"\n[{args.snap}/{args.subfind_id}] aperture={args.aperture} kpc (3D)")
    print(f"  n_particles inside aperture : {first['n_particles']}")
    masses = first['mass_by_species']
    print(f"  M_star  (aperture)          : {masses.get('star', 0.0):.3e} Msol")
    print(f"  M_gas   (aperture)          : {masses.get('gas', 0.0):.3e} Msol")
    print(f"  M_dm    (aperture)          : {masses.get('dm', 0.0):.3e} Msol")
    print(f"  catalogue mass_stars        : {meta['mass_stars'] * to_msol:.3e} Msol "
          f"(does not match the particle sum; prefer the sum)")
    print(f"  M_baryon (aperture)         : {first['mass_baryon']:.4e} Msol")
    print(f"  M_total  (aperture)         : {first['mass_total']:.4e} Msol")
    print(f"  catalogue SubhaloVel        : "
          f"({meta['vel_x']:.1f}, {meta['vel_y']:.1f}, {meta['vel_z']:.1f}) km/s")
    print(f"  catalogue SubhaloVelDisp    : {meta['veldisp']:.1f} km/s")
    print(f"  R90                         : {first['r90']:.3f} kpc")
    print(f"  V_baryon                    : {np.round(first['v_baryon'], 2)} km/s")
    print(f"  V_total                     : {np.round(first['v_total'], 2)} km/s")
    print(f"  |V_b - V_t|                 : "
          f"{np.linalg.norm(first['velocity_offset']):.2f} km/s")
    for mode, result in results.items():
        print(f"  M for V_c = {mode:<8}       : "
              f"{result['mass_enclosed_r90']:.4e} Msol, "
              f"V_c = {result['v_circ']:.2f} km/s, "
              f"s_b = {result['s_b']:.4f}")


if __name__ == '__main__':
    main()
