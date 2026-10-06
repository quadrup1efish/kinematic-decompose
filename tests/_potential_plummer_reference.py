"""Independent truncated-Plummer source and field for accuracy studies only."""
import numpy as np

PLUMMER_SCALE_KPC = 1.0
PLUMMER_TRUNCATION_KPC = 10.0
PLUMMER_TOTAL_MASS_MSUN = 1.0e10
RMIN_KPC = 1.0e-3
SOFTENING_AT_REFERENCE_KPC = 0.2
SOFTENING_REFERENCE_N = 1_000
ERROR_PROBE_COUNT = 64


def analytic_plummer(radii):
    """Potential and signed radial force of a finite, mass-normalized sphere."""
    radii = np.asarray(radii, dtype=float)
    scale = PLUMMER_SCALE_KPC
    truncation = PLUMMER_TRUNCATION_KPC
    enclosed_fraction = (truncation / np.sqrt(truncation**2 + scale**2))**3
    source_mass = PLUMMER_TOTAL_MASS_MSUN / enclosed_fraction
    gravity = 4.30091727067736e-6  # kpc (km/s)^2 / Msun
    inside = radii <= truncation
    phi = np.empty_like(radii)
    force = np.empty_like(radii)
    boundary_term = scale**2 / (truncation**2 + scale**2)**1.5
    phi[inside] = -gravity * source_mass * (
        1 / np.sqrt(radii[inside]**2 + scale**2) - boundary_term
    )
    force[inside] = -gravity * source_mass * radii[inside] / (radii[inside]**2 + scale**2)**1.5
    phi[~inside] = -gravity * PLUMMER_TOTAL_MASS_MSUN / radii[~inside]
    force[~inside] = -gravity * PLUMMER_TOTAL_MASS_MSUN / radii[~inside]**2
    return phi, force


def sample_truncated_plummer(count, rng):
    """Draw an isotropic Plummer sample within the fixed truncation radius."""
    fraction = (PLUMMER_TRUNCATION_KPC / np.sqrt(
        PLUMMER_TRUNCATION_KPC**2 + PLUMMER_SCALE_KPC**2
    ))**3
    q23 = (rng.random(count) * fraction)**(2.0 / 3.0)
    radius = PLUMMER_SCALE_KPC * np.sqrt(q23 / (1.0 - q23))
    cosine = rng.uniform(-1.0, 1.0, count)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    sine = np.sqrt(1.0 - cosine**2)
    return np.column_stack((radius * sine * np.cos(azimuth),
                            radius * sine * np.sin(azimuth), radius * cosine))
