import numpy as np

from pynbody import units
from pynbody.array import SimArray
from pynbody.snapshot.simsnap import SimSnap

from kinematic_decompose.potential import Potential, setUnits

# Keep the project unit convention unchanged from the former Agama backend.
setUnits(length=1, mass=1, velocity=1)

# Multipole components for the galaxy potential model: (family, symmetry,
# lmax). Retain fast spherical DM and axisymmetric gas approximations;
# stellar non-axisymmetry is included through lmax=mmax=4. Circular references
# use the azimuthally averaged midplane field, not one preferred direction.
_GALAXY_POTENTIAL_COMPONENTS = [
    ('dm',  's', 0),
    ('gas', 'a', 4),
    ('s',   'n', 4),
]

# TNG publishes collisionless softenings as Plummer-equivalent lengths.  The
# cubic-spline kernel used by the patched Agama source is Newtonian at
# h = 2.8 * epsilon_Plummer.
_KERNEL_SUPPORT_PER_PLUMMER_EQUIVALENT = 2.8


def _circular_velocity_from_radial_force(radii: np.ndarray, radial_force: np.ndarray) -> np.ndarray:
    """Return circular speeds, marking radii with outward/non-finite force invalid."""
    vc_squared = -np.asarray(radii, dtype=float) * np.asarray(radial_force, dtype=float)
    valid = np.isfinite(vc_squared) & (vc_squared > 0)
    return np.sqrt(np.where(valid, vc_squared, np.nan))

def _azimuthally_averaged_midplane(potential, radii, azimuth_samples=8):
    """Mean potential and radial force; do not average speed or force norm.

    Uniform periodic quadrature removes nonzero Fourier orders smaller than
    azimuth_samples. Eight points cover the default mmax=4. A higher-order
    supplied potential requires more points. Midplane values also define the
    z-reflection-symmetrized reference, without projecting particle energies.
    """
    radii = np.asarray(radii, dtype=float)
    if radii.ndim != 1 or not np.all(np.isfinite(radii) & (radii > 0)):
        raise ValueError("Reference radii must be a finite positive 1D array.")
    if not isinstance(azimuth_samples, (int, np.integer)) or azimuth_samples < 1:
        raise ValueError("azimuth_samples must be a positive integer.")
    angles = 2 * np.pi * np.arange(azimuth_samples) / azimuth_samples
    directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    points = (radii[:, None, None] * directions[None, :, :]).reshape(-1, 3)
    values = np.asarray(potential.potential(points)).reshape(len(radii), azimuth_samples)
    forces = np.asarray(potential.force(points)).reshape(len(radii), azimuth_samples, 3)
    radial_forces = np.einsum('rni,ni->rn', forces, directions)
    return values.mean(axis=1), radial_forces.mean(axis=1)


def create_multipole_potential(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float | None = None,
    symmetry: str = 'a',
    rmin: float | None = None,
    rmax: float = 0,
    lmax: int = 4,
    gridsizeR: int = 40,
    export: bool = False,
    filename: str|None = None
) -> Potential:
    """
    Compute gravitational potential field using Agama's multipole expansion.
    
    This function creates a multipole expansion potential from particle data,
    suitable for spherical or axisymmetric systems.
    
    Args:
        positions: Particle positions array of shape (N, 3) [required]
        masses: Particle masses array of shape (N,) [required]
        softening: Cubic-spline kernel support radius [kpc]. If None, use
            Agama's original point-particle source.
        symmetry: Symmetry type: 's' (spherical), 'a' (axisymmetric), 
                 or 'n' (none) (default: 'a')
        rmin: Innermost radial grid node. If omitted, softened particles use
            Agama's automatic choice (equivalent to 0), while point particles
            retain the legacy 1e-2 default.
        lmax: Maximum angular order of expansion (default: 4)
        gridsizeR: Number of radial grid points (default: 40)
        export: Whether to export potential to file (default: False)
        filename: Output filename if export=True (default: auto-generated)

    Notes:
        A scalar support radius uses fitted-density convolution when the native
        backend has enough noncentral positive-mass particles; sparse sources
        retain per-particle kernel integration. The threshold is 10*K*H, where
        K is the radial node count and H the active harmonic channel count
        (900 particles for K=30, symmetry='a', lmax=4).
        Direct ``Potential`` calls with a per-particle support array always use
        kernel integration, even when every array entry is equal. These
        approximations need not be identical.
        Requested radial endpoints may expand to include the kernel support.
        Symmetry 'a' also imposes plane-reflection symmetry.

    Returns:
        Potential: Potential object for gravity calculations
    
    Raises:
        ValueError: If input arrays have inconsistent shapes
    
    Example:
        >>> pot = create_multipole_potential(positions, masses, softening=0.5, lmax=6)
        >>> force = pot.force(1.0, 0.0, 0.0)
    """
    # Input validation
    if len(positions) != len(masses):
        raise ValueError(f"positions ({len(positions)}) and masses ({len(masses)}) must have same length")
    
    if positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")
    
    if rmin is None:
        rmin = 0 if softening is not None else 1e-2
    potential_args = dict(
        type='Multipole',
        particles=(positions, masses),
        symmetry=symmetry,
        rmin=rmin,
        rmax=rmax,
        lmax=lmax,
        gridsizeR=gridsizeR
    )
    if softening is not None:
        if softening <= 0:
            raise ValueError(f"softening must be positive, got {softening}")
        potential_args['softening'] = softening
    # Create potential
    pot = Potential(**potential_args)
    
    # Export if requested
    if export:
        if filename is None:
            filename = f"multipole_sym{symmetry}_lmax{lmax}.txt"
        pot.export(filename)
    
    return pot

def construct_galaxy_potential_model(galaxy, *, stellar_symmetry='n'):
    """Build the s/a/n, order-4, grid-30 model; 'a' restores the old stellar model."""
    eps = galaxy.properties.get('eps', 0.39)
    kernel_support = _KERNEL_SUPPORT_PER_PLUMMER_EQUIVALENT * eps

    potentials = []

    for name, symmetry, lmax in _GALAXY_POTENTIAL_COMPONENTS:
        if name == 's':
            symmetry = stellar_symmetry
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
            softening=kernel_support,
            symmetry=symmetry,
            rmin=0,
            rmax=galaxy.R_vir,
            lmax=lmax,
            gridsizeR=30
        )
        potentials.append(pot)

    if not potentials:
        raise ValueError("No valid potential components (all particle sets too small).")

    return Potential(*potentials)


def calculate_kinematic_param(
    galaxy: SimSnap,
    potential: Potential | None = None,
    partType: str = 'star',
    filename: str | None = None,
    *,
    azimuth_samples: int = 8,
) -> SimSnap:
    """
    Calculate kinematic parameters for galaxy particles.

    Computes potential, circular velocity, and angular momentum for given orbits.

    Args:
        galaxy: Galaxy snapshot with position and velocity data (must be in
            physical units (the backend applies its Agama-compatible `setUnits` scaling)
        potential: Pre-computed potential; if None, built from the galaxy
            particle distribution (unless `filename` is given)
        partType: Family to attach the circular angular momentum to
            ('star' | 'gas' | 'dm'). Also the family whose 'phi' is evaluated:
            the potential is only computed for this family's particles, since
            no other family's 'phi'/'e' is consumed downstream.
        filename: Potential file to load (if None, compute from particles)
        azimuth_samples: Ring samples for the circular reference. Must exceed
            mmax of a supplied potential; 8 covers the default order-4 model.

    Returns:
        Modified galaxy snapshot with 'phi' and 'jc' added to the `partType`
        family
    """
    # 1. Create or load potential
    if potential is None:
        if filename is None:
            potential = construct_galaxy_potential_model(galaxy)
        else:
            potential = Potential(filename)
    assert potential is not None  # built / loaded / passed in all branches
    # 2. Compute particle potentials
    # Only the family that actually consumes 'phi'/'e' is evaluated. In a
    # group-sized cutout the snapshot holds several times more DM and gas
    # particles than stars, and their potentials are never read: dump_container
    # stores target_families=['star'], and the pipeline only uses galaxy.s['e'].
    # Potential evaluation is point-by-point independent, so the values handed
    # to `partType` are bit-identical to evaluating every particle.
    particle_data = {
        'star': galaxy.s,
        'gas': galaxy.g,
        'dm': galaxy.dm,
    }.get(partType, galaxy)
    particle_data['phi'] = SimArray(
        potential.potential(particle_data['pos']),
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
    
    # 4. Average the field before computing speeds; particle energies remain
    # those of the full potential. One sample reproduces the legacy x-axis.
    circular_potentials, radial_forces = _azimuthally_averaged_midplane(
        potential, r_midpoints, azimuth_samples=azimuth_samples
    )
    circular_velocities = _circular_velocity_from_radial_force(
        r_midpoints, radial_forces
    )
    circular_energies = 0.5 * circular_velocities**2 + circular_potentials
    circular_angular_momenta = r_midpoints * circular_velocities
    
    # 5. Interpolate circular angular momentum for particle energies
    # Drop non-finite grid points BEFORE building the envelope: np.interp
    # requires finite xp/fp, and NaN would otherwise propagate through the
    # log-space interpolation. Restrict to bound circular orbits (E < 0) so
    # -log10(-E) stays finite and strictly increasing after the transform.
    valid_grid = (
        np.isfinite(circular_energies)
        & (circular_energies < 0)
        & np.isfinite(circular_angular_momenta)
    )
    finite_energies = circular_energies[valid_grid]
    finite_angular_momenta = circular_angular_momenta[valid_grid]
    if len(finite_energies) == 0:
        raise ValueError(
            "No finite, bound circular-orbit grid points; cannot interpolate jc.")

    sort_idx = np.argsort(finite_energies)
    sorted_energies = finite_energies[sort_idx]
    sorted_angular_momenta = finite_angular_momenta[sort_idx]
    # L_c(E) can be non-monotonic when the potential has a central cusp:
    # the same energy then admits two circular-orbit radii (inner/outer
    # branch). Preserve the historical monotone reference envelope, using the
    # per-energy upper envelope: for each unique branch energy take the
    # maximum angular momentum among the grid points AT that energy.
    # Then apply a running cumulative maximum over the per-energy maxima:
    # this enforces a monotonically non-decreasing upper envelope, so the
    # folded (non-monotonic) L_c(E) region cannot drag interpolated jc
    # below this reference envelope. It is NOT a rigorous angular-momentum
    # bound in the full non-axisymmetric potential. Deduplicating
    # energies with np.unique also guarantees xp is strictly increasing
    # (np.interp requirement) before the log transform.
    unique_energies, first_idx = np.unique(sorted_energies, return_index=True)
    # first_idx marks the start of each unique energy in the sorted array,
    # so maximum.reduceat evaluates the max over exactly that energy group.
    per_energy_max = np.maximum.reduceat(sorted_angular_momenta, first_idx)
    # Cummax of the per-energy max: without it, the low branch of the folded
    # L_c(E) can pollute the interpolation intervals of adjacent energies
    # and underestimate jc for real orbits.
    envelope_jc = np.maximum.accumulate(per_energy_max)
    # Guard against L=0 grid points: log10(0) = -inf would poison the
    # interpolation, so floor the envelope at a small positive value
    # (angular momenta are non-negative: r * sqrt(r * F_r)).
    envelope_jc = np.maximum(envelope_jc, np.finfo(float).eps)
    
    particle_energies = particle_data['e']
    
    # np.interp: piecewise-linear in log-log space, constant extrapolation at
    # the ends. xp must be strictly increasing: log10(-E) decreases with E,
    # so interpolate in -log10(-E) instead (interp1d sorted internally, but
    # np.interp does not).
    log_jc = np.interp(
        -np.log10(-particle_energies),
        -np.log10(-unique_energies),
        np.log10(envelope_jc),
        left=np.log10(envelope_jc[0]),
        right=np.log10(envelope_jc[-1]),
    )
    jc_values = 10**log_jc
    # unbound particles (e > 0) yield NaN in log10(-e); fall back to the
    # outermost circular angular momentum (same guard as the original code).
    # envelope_jc is guaranteed finite (finite grid points + eps floor), so
    # the fallback value cannot reintroduce NaN.
    jc_values = np.where(np.isnan(jc_values), envelope_jc[-1], jc_values)
    
    # 6. Store results
    particle_data['jc'] = SimArray(
        jc_values,
        units=particle_data['pos'].units * particle_data['vel'].units
    )
    
    return galaxy
