# Reading notes: potential-performance figure

The titleless, equal-panel figure separates build scaling (left) from field-error scaling (right). The left panel connects the measured medians, shows the 16–84% spread across timing repeats, and overlays dashed power-law fits for $N>10^5$. The displayed exponents are empirical slopes over this finite interval, not a proof of asymptotic complexity.

The right panel reports median errors over the listed radial probes; it does not show error spread over independent particle realizations. Every N uses a prefix of one seeded sample, and timing repetitions rebuild the same particles. Native is deliberately softened while stock Agama is not, so the error curves answer “deviation from the unsoftened Plummer truth,” not a matched-kernel numerical-equivalence question. See `data_truth/potential_performance_scaling.md` for exact reported values; the benchmark script can reproduce the full numeric output.
