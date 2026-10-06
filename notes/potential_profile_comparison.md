# Potential profile comparison — figure notes

The upper panel compares the unsoftened multipole profile, the cubic-spline-softened multipole profile, and a direct particle sum using the same cubic-spline gravity kernel. The lower panel shows signed relative differences after each profile has been independently normalized by its own sampled-range minimum. The softened multipole/direct-spline agreement is close; the unsoftened comparison also reflects the physical change introduced by softening and should not be interpreted as a solver-only error.

All components use `symmetry='n'` with `lmax=mmax=6`, so the models do not impose spherical or axisymmetric symmetry. The direct reference retains the particle distribution and is evaluated with the same spline gravity kernel. The comparison remains limited to the equatorial profile and sampled radii 0.01–5 kpc; it is not a global potential-error bound.

The figure is prepared at the A&A full-width maximum of 180 mm and 300 dpi. Serif typography, restrained axis/tick weights, explicit series labels, and distinct solid/marker versus dashed-line encodings are intended to remain legible at print size. The repository PNG is accompanied by a vector PDF for journal submission. To regenerate both, run `python tests/example_potential_profile_comparison.py` from the repository root.
