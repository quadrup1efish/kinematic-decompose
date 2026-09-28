# Potential-performance figure data truth

- Figure: `images/potential_performance_scaling.png`
- Reproduction: `uv run --no-sync python tests/example_potential_performance.py`; the JSON used for this figure is a fixed run with seed 2026.
- Model: truncated Plummer sphere, scale radius 1 kpc, truncation radius 10 kpc, total mass $10^{10}\,M_\odot$.
- Sample: one nested realization, 13 logarithmically spaced particle counts from $10^5$ to $10^7$; radial CDF maximum deviation at $10^7$ is $1.5665\times10^{-4}$.
- Construction: spherical symmetry, $l_{\max}=0$, `gridSizeR=30`, fixed radial limits 0.001–10 kpc; Native support $h(N)=0.2\,\mathrm{kpc}(1000/N)^{1/3}$, stock Agama unsoftened.
- Timing: nine construction repeats per N and backend; plotted central values are medians with 16th–84th percentile error bars.
- Measured high-N log-log slopes for $N>10^5$: Native 0.91656; stock Agama 0.94223. These are finite-range fits.
- Endpoint median build times: at $N=10^5$, Native 0.01692 s and Agama 0.01265 s; at $N=10^7$, Native 1.0565 s and Agama 0.7997 s.
- Error panel: median absolute relative potential and radial-force errors across 64 logarithmic probes from 0.05 to 10 kpc, relative to the unsoftened analytic truncated-Plummer field. Native errors therefore include intentional softening bias.
