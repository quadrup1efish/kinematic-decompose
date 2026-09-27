# TNG50-1 snap 99 subhalo 727693 — skew-t $e_{cut}$ check

The fit sample is selected from the cutout using the pipeline-style bound/rotation cuts, `JEHistogram(n_E=25, n_eps=50)`, and the additional `|jz/jc| <= 0.5` spheroid restriction; the physical-radius guard `r < 100 kpc` was applied first. The maintained two-skew-t fitter uses an unweighted energy histogram for its objective; particle masses enter its FindMin initialization and its component-location bounds.

- Fit sample: **10,800** stars.
- Returned $e_{cut}$: **-0.813047** (weighted component-density crossing).
- Independent recomputation of the fitted crossing: **-0.813047**.
- Distinguishability rule: fall back to `get_Ecut` when the Ashman separation index of the fitted modes is below **1.0**; measured separation is **2.003** ⇒ fallback **not triggered**.
- `get_Ecut` histogram valley on the same sample (comparison): **-0.626314**; the two cuts put **8,318 (77.0%)** vs **9,573 (88.6%)** stars below.
- Smoothed-histogram local minima: **-0.741, -0.614**.
- FindMin calls inside the fit (initialization and component-location bounds): **2**.
- Normalized fit objective relative to the single-component reference: **0.113**.

Interpretation: the two fitted components are distinguishable (Ashman separation index above the 1.0 threshold), so the reported $e_{cut}$ is the fitted component-density crossing; the `get_Ecut` line is drawn for comparison only.
 The retired degeneracy guards (component scale, component weight, component location near the fitted boundaries) can no longer substitute the cut; `removed_location_guard_margins_reference_only` only records how close the fitted locations sit to those retired thresholds.
