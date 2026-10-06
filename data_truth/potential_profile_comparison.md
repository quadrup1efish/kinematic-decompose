# Potential profile comparison figure — data truth

- **Input:** `tests/fixtures/potential_profile_symmetry_n.npz`, derived from the TNG50-1 snap 99, subID 715286 particle-potential comparison.
- **Components:** 335,520 particles total (DM 203,094; gas 112,386; stars 20,040).
- **Model configuration:** all components use Agama `symmetry='n'`, `lmax=mmax=6`, `gridSizeR=40`; softened profile uses the compact cubic-spline gravity kernel with Plummer-equivalent `epsilon=0.2879 kpc` (`h=2.8 epsilon`).
- **Radial samples:** 321 equatorial-plane query radii from 0.01 to 5.0 kpc.
- **Reference:** direct particle summation with the same cubic-spline gravity kernel. Each profile is independently normalized by the absolute value of its own minimum on the sampled interval. The lower panel is the signed relative difference of the independently normalized profile from the independently normalized direct-spline profile.
- **Measured normalized-profile differences:** softened p95 absolute relative difference `0.029811%`, maximum `0.065370%`; unsoftened p95 absolute relative difference `13.15835%`.
- **Figure output:** `images/potential_profile_comparison.png` (2125×2125 px, 300 dpi) and `images/potential_profile_comparison.pdf` (vector).
- **Reproduction:** `python tests/example_potential_profile_comparison.py`.
