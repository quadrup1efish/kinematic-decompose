import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.interpolate import interp1d

from pynbody import units
from pynbody.array import SimArray
from pynbody.snapshot.simsnap import SimSnap

import agama
agama.setNumThreads(1)
agama.setUnits(length=1, mass=1, velocity=1)

def create_multipole_potential(
    positions: np.ndarray,
    masses: np.ndarray,
    eps: float = 0.39,
    symmetry: str = 's',
    rmin: float = 1e-3,
    lmax: int = 8,
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
                 or 'n' (none) (default: 's')
        rmin: Minimum radius for potential evaluation (default: 1e-3)
        lmax: Maximum angular order of expansion (default: 8)
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
        lmax=lmax,
        gridsizeR=gridsizeR
    )
    
    # Export if requested
    if export:
        if filename is None:
            filename = f"multipole_sym{symmetry}_eps{eps}_lmax{lmax}.txt"
        pot.export(filename)
    
    return pot

def calculate_kinematic_param(
    galaxy: SimSnap,
    potential = None,
    filename: str | None = None
) -> SimSnap:
    """
    Calculate kinematic parameters for galaxy particles.
    
    Computes potential, circular velocity, and angular momentum for given orbits.
    
    Args:
        galaxy: Galaxy snapshot with position and velocity data
        filename: Potential file to load (if None, compute from particles)
    
    Returns:
        Modified galaxy snapshot with added 'phi' and 'jc' fields
    """
    # 1. Create or load potential
    if potential is None:
        if filename is None:
            positions = galaxy['pos'].view(np.ndarray)
            masses = galaxy['mass'].view(np.ndarray)
            eps = galaxy.properties.get('eps', 0.39)
            potential = create_multipole_potential(
                positions, masses, eps, export=False
            )
        else:
            potential = agama.Potential(filename)
    else:
        pass
    # 2. Compute particle potentials
    galaxy['phi'] = SimArray(
        potential.potential(galaxy['pos']),
        units=units.km**2 / units.s**2
    )
    
    # 3. Prepare radial grid for circular orbits
    particle_radii = galaxy['r'].view(np.ndarray)
    positive_radii = particle_radii[particle_radii > 0]
    
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
    sorted_angular_momenta = circular_angular_momenta[sort_idx]
    
    jc_interpolator = interp1d(
        np.log10(-sorted_energies),
        np.log10(sorted_angular_momenta),
        fill_value='extrapolate',
        bounds_error=False
    )
    
    # Compute jc for each particle
    particle_energies = galaxy['e']
    log_jc = jc_interpolator(np.log10(-particle_energies))
    jc_values = 10**log_jc
    
    # Handle extrapolated values
    min_energy = circular_energies.min()
    max_energy = circular_energies.max()
    
    jc_values[particle_energies > max_energy] = circular_angular_momenta[-1]
    jc_values[particle_energies < min_energy] = circular_angular_momenta[0]
    
    # 6. Store results
    galaxy['jc'] = SimArray(
        jc_values,
        units=galaxy.s['pos'].units * galaxy.s['vel'].units
    )
    
    return galaxy
