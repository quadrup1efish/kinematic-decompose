# kinematic-decompose

**Automated kinematic decomposition of galaxies using adaptive Gaussian Mixture Models.**

This package provides an end-to-end pipeline for decomposing simulated galaxies (from cosmological simulations such as IllustrisTNG) into their kinematic components — cold disk, warm disk, bulge, stellar halo, and counter-rotating disk — using an **automated Gaussian Mixture Model (AutoGMM)** approach.

![Example Decomposition](image/example_decomposition.png)

## Overview

Traditional kinematic decomposition methods (e.g., Abadi + JEHistogram) rely on ad-hoc energy and angular momentum cuts. This package replaces hard cuts with an **adaptive, data-driven Gaussian Mixture Model** that:

1. **Automatically classifies morphology** — uses a 3-component GMM to distinguish disks from spheroids based on the $e/|e_{\min}|$–$j_z/j_c$ phase space.
2. **Initializes physically motivated components** — separates bulge, stellar halo, cold disk, warm disk, and counter-rotating disk using energy ($e/|e_{\min}|$) and circularity ($j_z/j_c$) thresholds.
3. **Detects residual underfitting** — uses a 2D histogram-based $\Delta L$ criterion to identify phase-space regions where the current mixture under-represents the data and adds new components automatically.
4. **Supports soft and hard classification** — assigns each star particle a probabilistic or hard label for the five kinematic categories.

### Kinematic Phase Space

The decomposition operates in three dimensions of the stellar orbital phase space:

| Variable | Definition | Description |
|----------|-----------|-------------|
| $e/|e_{\min}|$ | $E / |E_{\min}|$ | Orbital energy normalised by the magnitude of the minimum (most bound) energy; bound particles lie in $[-1, 0)$ |
| $j_z/j_c$ | $L_z / L_c(E)$ | Circularity: z-component of angular momentum over circular angular momentum at same energy |
| $j_p/j_c$ | $L_p / L_c(E)$ | Perpendicular angular momentum fraction |

<p align="center">
  <img src="image/phase_space_example.png" width="600" alt="Phase space decomposition example"/>
</p>

### Components

| Component | Phase-space signature |
|-----------|----------------------|
| **Cold disk** | $j_z/j_c > 0.85$ |
| **Warm disk** | $j_z/j_c > 0.5$ |
| **Bulge** | $e/|e_{\min}| < e_{\mathrm{cut}}$ and $|j_z/j_c| < 0.5$ |
| **Stellar halo** | $e/|e_{\min}| > e_{\mathrm{cut}}$ and $|j_z/j_c| < 0.5$ |
| **Counter-rotating disk** | $j_z/j_c < -0.5$ |

$e_{\mathrm{cut}}$ is determined adaptively from the gravitational potential and the stellar energy distribution; the circularity threshold is a fixed 0.5.

## Project Structure

```
kinematic-decompose/
├── src/kinematic_decompose/
│   ├── __init__.py                      # Entry point
│   ├── config.py                        # TNG simulation base path and defaults
│   ├── pipeline.py                      # End-to-end decomposition pipeline
│   ├── visualize.py                     # Publication-quality visualisation (phase space, surface density, LOS velocity)
│   ├── mixture/
│   │   ├── __init__.py
│   │   ├── _base.py                     # Base mixture model (from scikit-learn)
│   │   ├── _gaussian_mixture.py         # Gaussian Mixture Model (extended with soft prediction)
│   │   ├── _auto_gaussian_mixture.py    # **AutoGaussianMixtureModel** — the core algorithm
│   │   ├── preprocessing.py             # RobustScaler for phase-space normalisation
│   │   └── util.py                      # JEHistogram, Ecut, decomposition, structure properties
│   ├── gravity/
│   │   └── kinematic_solver.py          # Agama multipole potential + kinematic parameter calculation
│   └── PyTNG/
│       ├── snapshot_loader.py           # TNG snapshot loader (pynbody-based)
│       ├── derived_array.py             # Derived arrays (energy, angular momentum, etc.)
│       ├── extension.py                 # pynbody extensions (disk/spheroid filters, structural props)
│       ├── tng_config.py                # TNG simulation config
│       ├── simdict_getter.py            # Helper for simulation dict fields
│       └── illustris_python/            # Low-level TNG I/O routines
├── tests/
│   └── example_kinematic_decomposition.ipynb  # Example notebook
├── image/                               # Output and reference images
├── IDEA.md                              # Development notes (Chinese)
├── pyproject.toml
└── README.md
```

## Installation

### Prerequisites

- Python ≥ 3.11
- [Agama](https://github.com/GalacticDynamics-Oxford/Agama) (galaxy dynamics library)
- [pynbody](https://pynbody.github.io/) (N-body/SPH snapshot analysis)
- [IllustrisTNG](https://www.tng-project.org/) simulation data access

### Install with uv

```bash
uv pip install -e .
```

On macOS with Agama installed via Homebrew, you may need:

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib
```

## Usage

### Quick Start

```python
from kinematic_decompose.pipeline import kinematic_decomposition_pipeline

model, galaxy, eoemin_cut, jzojc_cut = kinematic_decomposition_pipeline(
    run='TNG50-1',
    snapNum=99,
    subID=307486,
    gravity_potential_path='./potentials/',
    image_path='./images/',
    structure_properties_output_path='./properties/',
    mixture_model_output_path='./models/',
)
```

### Step-by-step Usage

See [`tests/example_kinematic_decomposition.ipynb`](tests/example_kinematic_decomposition.ipynb) for a complete walkthrough.

```python
from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.gravity.kinematic_solver import (
    construct_galaxy_potential_model, calculate_kinematic_param
)
from kinematic_decompose.mixture import AutoGaussianMixtureModel, preprocessing, util
from kinematic_decompose.config import BASEPATH
from kinematic_decompose.visualize import visualize_decomposition

# 1. Load a galaxy from TNG
snapshot = Snapshot(f"{BASEPATH}/TNG50-1/output", snapNum=99)
snapshot.load_particle(ID=307486, load_particle_fields='default')
snapshot.physical_units()
snapshot.load_group_catalog(ID=307486)
snapshot.GC_physical_units()
snapshot.center(cen=snapshot.group_catalog['SubhaloPos'])
snapshot.faceon(align_with='star', range=[3*snapshot.properties['eps'], 5*snapshot.s.r50])

# 2. Construct gravitational potential (Agama Multipole)
pot = construct_galaxy_potential_model(galaxy)

# 3. Compute kinematic parameters (φ, j_c, e/|e|_max, j_z/j_c, j_p/j_c)
galaxy = calculate_kinematic_param(galaxy, potential=pot)

# 4. Build training data
X = np.column_stack([galaxy.s['eoemin'], galaxy.s['jzojc'], galaxy.s['jpojc']])
keep = (galaxy.s['eoemin'] < 0) & (np.abs(galaxy.s['jzojc']) < 1.5) & (galaxy.s['jpojc'] < 1.5)

# 5. Determine energy cut
sph, _ = util.JEHistogram(galaxy.s['eoemin'][keep], galaxy.s['jzojc'][keep])
eoemin_cut = util.get_Ecut(galaxy.s['eoemin'][keep][sph], galaxy.s['mass'][keep][sph])

# 6. Normalise and run AutoGMM
scaler = preprocessing.RobustScaler()
X_train = scaler.fit_transform(X[keep])
auto_gmm = AutoGaussianMixtureModel()
auto_gmm.fit(X_train, eoemin_cut=scaler.transform(eoemin_cut, columns=0),
             jzojc_cut=scaler.transform(0.5, columns=1),
             r_jzojc_cut=scaler.transform(-0.5, columns=1),
             sample_weight=galaxy.s['mass'][keep])
best_model = scaler.inverse_transform_GMM(auto_gmm.best_model)

# 7. Decompose the galaxy
galaxy = util.decompose(X, galaxy, best_model, eoemin_cut, jzojc_cut, predict_method='hard')

# 8. Visualise
visualize_decomposition(X, best_model, galaxy, eoemin_cut, jzojc_cut, threshold_line=True)
```

## Algorithm: Adaptive Gaussian Mixture Model (AutoGMM)

The core of the package is an **adaptive** GMM: instead of assuming a fixed number of components a priori, it **discovers the kinematic structure from the data** — morphology is classified first, then residual phase-space regions that the current mixture under-represents are detected automatically and new components are added until the model fits. The component count is therefore determined by the data, not by hand.

The custom `GaussianMixture` in `mixture/_gaussian_mixture.py` extends scikit-learn's implementation with:

- **`soft_predict(X)`**: probabilistic label assignment from responsibilities
- **`sample_weight` support**: mass-weighted fitting
- **`min_iter`**: minimum iteration guarantee before convergence check
- **Full precision matrix initialisation** from user-provided `precisions_init`

### AutoGMM Fitting Stages

1. **Morphology classification** — `_morphology_class()`: fits a 3-component GMM on ($e/|e_{\min}|$, $j_z/j_c$). If any component has $\mu_{j_z/j_c} > \text{cut}$, the galaxy is classified as `'disk'`; otherwise `'spheroid'`.

2. **Physical initialisation** — `_initialize()`: maps GMM components to bulge, halo, disk subgroups using energy and circularity cuts. Falls back to data-driven statistics when a morphological subgroup is missing.

3. **Residual component detection** — `_find_residual_component()`: the key innovation. It:
   - Constructs a 2D histogram of the true data in ($e/|e_{\min}|$, $j_z/j_c$)
   - Computes the model-predicted density from the current GMM
   - Calculates $\Delta L = \text{true}\cdot\log(\text{true}/\text{model}) - (\text{true} - \text{model})$ (a likelihood-ratio residual)
   - Thresholds outlier regions and estimates new Gaussian components for each
   - Uses a BIC-like gain criterion to select which new components to keep

4. **Final EM fitting** — runs the full GMM with the automatically determined number of components and initialisation.

## Dependencies

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `agama` | ≥ 1.0.0 | Gravitational potential (Multipole expansion) |
| `numpy` | ≥ 2.4.0 | Numerical computing |
| `scipy` | ≥ 1.17.0 | Statistics, interpolation, optimisation |
| `scikit-learn` | ≥ 1.8.0 | GMM base implementation |
| `scikit-image` | ≥ 0.26.0 | Image processing (label, watershed) |
| `pynbody` | ≥ 2.4.0 | Simulation snapshot analysis |
| `pandas` | ≥ 3.0.0 | Data structures (optional, for outputs) |
| `matplotlib` | ≥ 3.10.0 | Visualisation |
| `pytest` | ≥ 9.0.0 | Testing |

## Visualisation

`visualize.py` generates publication-quality multipanel figures:

- **Phase space** (top row): 2D histograms of ($j_z/j_c$, $e/|e_{\min}|$), ($j_z/j_c$, $j_p/j_c$), and ($j_p/j_c$, $e/|e_{\min}|$) with Gaussian ellipses colour-coded by component type.
- **Surface density** (middle row): projected stellar surface density maps ($\log_{10} \Sigma_*$) for each component, with face-on and edge-on views.
- **LOS velocity** (bottom row): line-of-sight stellar velocity maps ($v_{\text{los}} / \sqrt{v_{\text{los}}^2 + 3\sigma_{\text{los}}^2}$).

## Performance: Mini-Batch vs Full-Batch EM

The `GaussianMixture` supports **mini-batch EM whose per-iteration cost is independent of the number of particles**: it scales with `batch_size`, not with N. Full-batch EM must touch every particle each iteration (O(N) per iteration); mini-batch EM works on a fixed-size batch (O(batch) per iteration).

| | Full-batch | Mini-batch |
|---|---|---|
| Per-iteration cost | O(N) | **O(batch) — independent of N** |
| Per-iteration time @ N = 10⁴ | 1.5 ms | 0.4 ms |
| Per-iteration time @ N = 10⁷ | 3.79 s | **1.9 ms (~2000× faster)** |
| Cost growth as N: 10⁴ → 10⁷ (×1000) | ×2500 | **×5** |
| Converged lower bound | reference | within 0.5% (same iteration count) |

<p align="center">
  <img src="image/scaling_performance.png" width="700" alt="Mini-batch vs full-batch scaling"/>
</p>

The key enabler is a **statistically-grounded bounded initialisation**: initial parameters are estimated on a random subsample of size

$$S = \frac{K \cdot d(d+1)}{2\varepsilon^2}, \qquad \varepsilon = 0.05,$$

derived from the statistical power of the covariance estimator (≈5% relative precision of the initial covariance), so the initialisation cost is also independent of N. The scaling behaviour is verified by `tests/example_gaussian_mixture.py::test_scaling_with_n_samples` (N = 10⁴–10⁶, fixed 10 iterations, mini-batch repeated 10× per N: median curve with 16–84% spread bands). The right axis shows the **Bayes factor** BF = exp(ΔLB) of the full- vs mini-batch converged solutions on a log scale, with the y-range spanning the Jeffreys "no substantial evidence" region [0.1, 10]. The measured BF ≈ 1 (0.96–1.01 in this run) — **no evidence of a difference**: the two convergence paths are statistically indistinguishable, not merely close.

Beyond EM, the end-to-end pipeline (TNG50-1, 5.1M stars — automatic component selection, kinematic decomposition, publication-quality figures) runs in **22 s**, down from 37 s.

## Visualisation Style

All figures share a unified **Nature-journal style** (`NATURE_STYLE` in `visualize.py`): sans-serif Helvetica/Arial, restrained sizes, hairline axes, no grid, 300 dpi output. Surface-density and LOS-velocity maps are binned with an O(N) `searchsorted` + `bincount` routine (no `lexsort`) and rasterised in PDF output, keeping file sizes small.

## Testing

```bash
python -m pytest tests/example_gaussian_mixture.py tests/example_eoemin_cut.py -q
```

- `tests/example_eoemin_cut.py` — 21 tests locking the behaviour of the energy-cut algorithm across 19 synthetic scenarios (valley, seamless, uniform, noisy regimes), with TRUE (green dashed) vs DETECTED (red solid) overlays.
- `tests/example_gaussian_mixture.py` — GMM/AutoGMM behaviour tests plus the N-scaling benchmark described above.

## Reference

If you use this code in your research, please cite the relevant papers:

- **AutoGMM method**: (TODO — add paper reference when published)
- **IllustrisTNG**: Nelson et al. 2019, [CompAC, 6, 2](https://ui.adsabs.harvard.edu/abs/2019ComAC...6....2N)
- **Agama**: Vasiliev 2019, [MNRAS, 482, 1525](https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1525V)
- **pynbody**: Pontzen et al. 2013, [ApJS, 239, 39](https://ui.adsabs.harvard.edu/abs/2018MNRAS.473.4025P)

## License

This project is licensed under the BSD-3-Clause License — see the source headers for details. The GMM implementation in `mixture/_gaussian_mixture.py` and `mixture/_base.py` derives from scikit-learn (BSD-3-Clause).
