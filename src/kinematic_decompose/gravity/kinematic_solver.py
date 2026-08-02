import numpy as np

from pynbody import units
from pynbody.array import SimArray
from pynbody.snapshot.simsnap import SimSnap

import agama
agama.setNumThreads(1)
# NOTE: units(1,1,1) means all inputs/outputs are in code units; the caller
# must convert the snapshot to physical units (kpc / km s^-1 / Msol) first
# so that the numerical values match the units attached to the arrays below.
agama.setUnits(length=1, mass=1, velocity=1)

# Multipole components for the galaxy potential model: (family, symmetry,
# lmax). dm is treated as spherical (lmax=0, monopole only) since its
# distribution is nearly round; gas and stars are axisymmetric (lmax=4,
# enough to capture the disk flattening). See Agama docs on Multipole.
_GALAXY_POTENTIAL_COMPONENTS = [
    ('dm',  's', 0),
    ('gas', 'a', 4),
    ('s',   'a', 4),
]

def create_multipole_potential(
    positions: np.ndarray,
    masses: np.ndarray,
    eps: float = 0.39,
    symmetry: str = 'a',
    rmin: float = 1e-2,
    rmax: float = 0,
    lmax: int = 4,
    gridsizeR: int = 40,
    export: bool = False,
    filename: str|None = None
) -> agama.Potential:
    """
    Compute gravitational potential field using Agama's multipole expansion.
    
    This function creates a multipole expansion potential from particle data,
    suitable for spherical or axisymmetric systems.
    
    Args:
        positions: Particle positions array of shape (N, 3) [required]
        masses: Particle masses array of shape (N,) [required]
        eps: Smoothing length parameter (default: 0.39)
        symmetry: Symmetry type: 's' (spherical), 'a' (axisymmetric), 
                 or 'n' (none) (default: 'a')
        rmin: Minimum radius for potential evaluation (default: 1e-2)
        lmax: Maximum angular order of expansion (default: 4)
        gridsizeR: Number of radial grid points (default: 40)
        export: Whether to export potential to file (default: False)
        filename: Output filename if export=True (default: auto-generated)
    
    Returns:
        agama.Potential: Potential object for gravity calculations
    
    Raises:
        ValueError: If input arrays have inconsistent shapes
    
    Example:
        >>> pot = multipole_expansion(positions, masses, eps=0.5, lmax=6)
        >>> force = pot.force(1.0, 0.0, 0.0)
    """
    # Input validation
    if len(positions) != len(masses):
        raise ValueError(f"positions ({len(positions)}) and masses ({len(masses)}) must have same length")
    
    if positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")
    
    # Create potential
    pot = agama.Potential(
        type='Multipole',
        particles=(positions, masses),
        symmetry=symmetry,
        smoothing=eps,
        rmin=rmin,
        rmax=rmax,
        lmax=lmax,
        gridsizeR=gridsizeR
    )
    
    # Export if requested
    if export:
        if filename is None:
            filename = f"multipole_sym{symmetry}_eps{eps}_lmax{lmax}.txt"
        pot.export(filename)
    
    return pot

def construct_galaxy_potential_model(galaxy):
    eps = galaxy.properties.get('eps', 0.39)

    potentials = []

    for name, symmetry, lmax in _GALAXY_POTENTIAL_COMPONENTS:
        pos = getattr(galaxy, name)['pos']
        if name == 'dm':
            mass = np.broadcast_to(
                np.asarray(galaxy.properties['mDM'], dtype=pos.dtype),
                len(pos)
            )
        else:
            mass = getattr(galaxy, name)['mass']

        if len(pos) < 10:
            print(f"Warning: {name} particles too few ({len(pos)}), skipping.")
            continue

        pot = create_multipole_potential(
            pos, mass,
            eps=eps,
            symmetry=symmetry,
            rmin=2*eps,
            rmax=galaxy.R_vir,
            lmax=lmax,
            gridsizeR=30
        )
        potentials.append(pot)

    if not potentials:
        raise ValueError("No valid potential components (all particle sets too small).")

    return agama.Potential(*potentials)


def calculate_kinematic_param(
    galaxy: SimSnap,
    potential: agama.Potential | None = None,
    partType: str = 'star',
    filename: str | None = None
) -> SimSnap:
    """
    Calculate kinematic parameters for galaxy particles.

    Computes potential, circular velocity, and angular momentum for given orbits.

    Args:
        galaxy: Galaxy snapshot with position and velocity data (must be in
            physical units, see module note on agama.setUnits)
        potential: Pre-computed potential; if None, built from the galaxy
            particle distribution (unless `filename` is given)
        partType: Family to attach the circular angular momentum to
            ('star' | 'gas' | 'dm')
        filename: Potential file to load (if None, compute from particles)

    Returns:
        Modified galaxy snapshot with added 'phi' and 'jc' fields
    """
    # 1. Create or load potential
    if potential is None:
        if filename is None:
            potential = construct_galaxy_potential_model(galaxy)
        else:
            potential = agama.Potential(filename)
    assert potential is not None  # built / loaded / passed in all branches
    # 2. Compute particle potentials
    galaxy['phi'] = SimArray(
        potential.potential(galaxy['pos']),
        units=units.km**2 / units.s**2
    )
    
    # 3. Prepare radial grid for circular orbits
    particle_radii = galaxy['r'].view(np.ndarray)
    positive_radii = particle_radii[particle_radii > 0]
    if len(positive_radii) == 0:
        raise ValueError("No particles with positive radius; cannot build circular-orbit grid.")
    
    r_min = 0.9 * np.min(positive_radii)
    r_max = 1.1 * np.max(particle_radii)
    
    r_bins = np.logspace(np.log10(r_min), np.log10(r_max), 100)
    r_midpoints = 0.5 * (r_bins[:-1] + r_bins[1:])
    
    # 4. Compute circular orbit quantities at grid points
    grid_points = np.column_stack([
        r_midpoints,
        np.zeros_like(r_midpoints),
        np.zeros_like(r_midpoints)
    ])
    
    circular_potentials = potential.potential(grid_points)
    radial_forces = np.linalg.norm(potential.force(grid_points)[:, :2], axis=1)
    circular_velocities = np.sqrt(r_midpoints * radial_forces)
    circular_energies = 0.5 * circular_velocities**2 + circular_potentials
    circular_angular_momenta = r_midpoints * circular_velocities
    
    # 5. Interpolate circular angular momentum for particle energies
    sort_idx = np.argsort(circular_energies)
    sorted_energies = circular_energies[sort_idx]
    # L_c(E) can be non-monotonic when the potential has a central cusp:
    # the same energy then admits two circular-orbit radii (inner/outer
    # branch). Every orbit satisfies |j| <= L_c(E), so interpolate on the
    # upper envelope (cumulative maximum) of the sorted circular angular
    # momenta - the plain sorted sequence would interpolate between the
    # two branches and underestimate jc, producing unphysical jz/jc >> 1.
    sorted_angular_momenta = np.maximum.accumulate(
        circular_angular_momenta[sort_idx])
    
    particle_data = {
        'star': galaxy.s,
        'gas': galaxy.g,
        'dm': galaxy.dm,
    }.get(partType, galaxy)
    particle_energies = particle_data['e']
    
    # np.interp: piecewise-linear in log-log space, constant extrapolation at
    # the ends. xp must be strictly increasing: log10(-E) decreases with E,
    # so interpolate in -log10(-E) instead (interp1d sorted internally, but
    # np.interp does not).
    log_jc = np.interp(
        -np.log10(-particle_energies),
        -np.log10(-sorted_energies),
        np.log10(sorted_angular_momenta),
        left=np.log10(sorted_angular_momenta[0]),
        right=np.log10(sorted_angular_momenta[-1]),
    )
    jc_values = 10**log_jc
    # unbound particles (e > 0) yield NaN in log10(-e); fall back to the
    # outermost circular angular momentum (same guard as the original code)
    jc_values = np.where(np.isnan(jc_values), sorted_angular_momenta[-1], jc_values)
    
    # 6. Store results
    particle_data['jc'] = SimArray(
        jc_values,
        units=particle_data['pos'].units * particle_data['vel'].units
    )
    
    return galaxy
