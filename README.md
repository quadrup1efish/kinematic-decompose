# kinematic-decompose

**From simulated star particles to a galaxy's kinematic components.**

[中文](README.zh-CN.md) · [Technical guide](docs/guide.md) · [Example notebook](tests/example_kinematic_decomposition.ipynb)

An end-to-end IllustrisTNG pipeline: load a galaxy, construct its gravitational potential, calculate orbital features, fit an adaptive Gaussian mixture, and separate **bulge, stellar halo, cold disk, warm disk, and counter-rotating disk**.

![Galaxy decomposition: orbital-feature distributions and spatial maps of the five kinematic components](images/example_decomposition.png)

*Example output. Top: fitted mixtures in orbital-feature space. Below: the total galaxy and its kinematic components. Open the image for full resolution.*

## What makes it different?

| Method | What it contributes |
|---|---|
| **Adaptive Gaussian mixtures** | Morphology-aware initialization and residual detection add Gaussian components where the current model underfits, rather than requiring a fixed component count for every galaxy. |
| **Adaptive energy separation** | A two-component skew-t fit estimates the bulge–halo energy division; fallback rules handle unresolved or unsuitable fits. |
| **Mini-batch EM** | Updates use bounded particle batches for large samples; full-batch EM remains available for comparison. Batch size depends on mixture complexity and the requested `tv_error` setting. |
| **Softened particle potentials** | An in-tree, Agama-derived native backend supports cubic-spline softening and configurable symmetry, without requiring an external Agama installation for the pipeline. |

The model uses normalized binding energy and angular momentum: $e/|e_{\min}|$, $j_z/j_c$, and $j_p/j_c$. Gaussian components are grouped into the five physical categories; these are **not** necessarily five Gaussians. The method still has documented thresholds and numerical settings—it is not parameter-free.

## Quick start

Requires **Python 3.11+**, a C/C++ compiler, `make`, and local TNG snapshots and group catalogs. The first native build downloads pinned GSL sources; no separate system GSL installation is needed.

```bash
git clone https://github.com/quadrup1efish/kinematic-decompose.git
cd kinematic-decompose
uv sync --locked
```

Set the simulation root in [`config.py`](src/kinematic_decompose/config.py), then run:

```python
from kinematic_decompose.pipeline import kinematic_decomposition_pipeline

model, galaxy, energy_cut, circularity_cut = kinematic_decomposition_pipeline(
    run="TNG50-1",
    snapNum=99,
    subID=307486,
    gravity_potential_path="./potentials",
    image_path="./results/images",
    structure_properties_output_path="./results/properties",
    mixture_model_output_path="./results/models",
)
```

Outputs include stellar classifications, a fitted mixture model, component properties, and diagnostic plots. Potential caches are isolated by simulation and snapshot, e.g. `potentials/TNG50-1/099/307486.ini`. After changing potential-construction settings, use `force_rebuild_potential=True`.

## Validation and benchmarks

- **Mixture fitting:** [mini-batch vs full-batch measurements](images/scaling_performance.png), [batch-size tests](images/batch_size.png), and [regression tests](tests/test_gaussian_mixture.py).
- **Potential accuracy:** [analytic models and a real-particle direct-sum reference](images/potential_analytic_models.png), with [reproducible calculations](tests/example_potential_analytic.py).
- **Potential build time:** [native backend vs Agama](images/potential_performance_scaling.png), with [benchmark code](tests/example_potential_performance.py).

Performance and fitting differences depend on the dataset, model settings, and hardware; these benchmarks are not a guarantee of identical classifications or a universal speedup.

Run the regression suite with `uv run pytest tests -q -rs`. Some checks require an external Agama reference or local simulation data; see the [testing instructions](docs/guide.md#testing).

<details>
<summary>Show potential-accuracy comparison</summary>

![Potential profiles and relative errors for analytic models and a real particle sample](images/potential_analytic_models.png)

Analytic and particle cases use different reference constructions and expansion orders. See the [technical guide](docs/guide.md#potential-construction-benchmarks) for the exact setup and limitations.

</details>

## Documentation and attribution

- [Technical guide](docs/guide.md): algorithm, potential conventions, cache behavior, references, and testing.
- [Native backend](src/kinematic_decompose/potential/native/README.md): supported operations, source provenance, and build details.
- **Author and maintainer:** Taiyu Li. Contributions remain credited in Git history; incorporated third-party code retains its upstream attribution.
- **License:** GPL-3.0-or-later. Please cite the software and the relevant [method papers](docs/guide.md#reference) when using it in research. Citation is a scholarly request, not an additional license restriction.
