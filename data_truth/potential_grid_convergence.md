# Potential radial-grid convergence data truth

- Figure: `images/potential_grid_convergence.png`
- Reproduction: `uv run --no-sync python tests/example_potential_grid_convergence.py`.
- Fixed model/sample: truncated Plummer, $a=1$ kpc, $R_t=10$ kpc, $M=10^{10}\,M_\odot$, $N=10^6$, RNG seed 2026; the identical positions and masses are reused at every grid size. Radial CDF maximum deviation is $7.8443\times10^{-4}$.
- Fixed source/softening settings: spherical $l_{\max}=0$, `rmin=0.001 kpc`, `rmax=10 kpc`, Native cubic-spline support $h=0.02$ kpc from the particle-count rule; stock Agama is unsoftened.
- Radial grid sizes: `gridSizeR` = 15, 30, 60, 120, 240. Error probes: 64 logarithmic radii from 0.05 to 10 kpc.
- Native matched analytic reference: the unsoftened truncated-Plummer potential and force convolved with the same normalized compact cubic-spline kernel. The 3-D kernel normalization quadrature returns 1.0. On five validation radii, changing quadrature order from (32 radial, 48 angular) to (64 radial, 96 angular) changed potential and force by at most $6.4\times10^{-16}$ and $5.3\times10^{-16}$ relative, respectively.
- Agama analytic reference: unsoftened truncated-Plummer potential and force.

## Field change from $N_r=120$ to $N_r=240

Values are median / 95th percentile / maximum of $|X_{120}-X_{240}|/|X_{240}|$ over the 64 probes.

| Backend | Field | Median | P95 | Maximum |
|---|---|---:|---:|---:|
| Native | Potential | $5.07\times10^{-7}$ | $2.48\times10^{-6}$ | $7.55\times10^{-6}$ |
| Native | Radial force | $8.38\times10^{-5}$ | $6.72\times10^{-4}$ | $1.06\times10^{-3}$ |
| Agama | Potential | $6.08\times10^{-7}$ | $2.52\times10^{-6}$ | $7.92\times10^{-6}$ |
| Agama | Radial force | $1.19\times10^{-4}$ | $6.53\times10^{-3}$ | $9.33\times10^{-3}$ |

## Error against each method's analytic target at $N_r=240

Values are median / P95 / maximum absolute relative error over the 64 probes.

| Backend | Field | Median | P95 | Maximum |
|---|---|---:|---:|---:|
| Native, matched softened Plummer | Potential | $5.65\times10^{-4}$ | $8.19\times10^{-4}$ | $8.34\times10^{-4}$ |
| Native, matched softened Plummer | Radial force | $1.36\times10^{-3}$ | $3.57\times10^{-2}$ | $4.44\times10^{-2}$ |
| Agama, unsoftened Plummer | Potential | $5.65\times10^{-4}$ | $8.20\times10^{-4}$ | $8.33\times10^{-4}$ |
| Agama, unsoftened Plummer | Radial force | $1.41\times10^{-3}$ | $3.31\times10^{-2}$ | $5.41\times10^{-2}$ |

At `gridSizeR=30`, median matched-target errors are Native $5.86\times10^{-4}$ (potential) and $1.40\times10^{-3}$ (force); Agama $5.86\times10^{-4}$ and $1.41\times10^{-3}$. Native-versus-unsoftened errors are separately retained in the JSON because they include the physical softening difference.
