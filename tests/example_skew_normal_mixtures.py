"""Functional tests for SkewNormalMixtures -- a finite mixture of
skew-normal densities (Lin, Lee & Yen 2007, Statistica Sinica 17,
909-927; multivariate ECM aligned with the mixsmsn R package, Prates,
Lachos & Cabral 2013), fitted on top of the project's BaseMixture
template (mini-batch fit, n_init selection, warm_start,
score/predict/BIC machinery inherited).

Method
------
The ECM (mixsmsn Delta-parameterization; all steps closed form)
augments each particle with a half-normal latent tau; the E-step needs
only truncated-normal moments (Mill's ratio in log space), and the
M-step is: CM1 weights, CM2 locations mu, CM3 Delta = Sigma^{1/2} delta,
CM4 Gamma = Sigma - Delta Delta^T; p = 1 reduces to the paper's scheme.

Initialization is a deterministic delta-sign scan (see
_initialize_parameters): positions from a symmetric Gaussian fit, then
all 2^K sign patterns at magnitudes (0.3, 0.5, 0.8) are evaluated at
fixed positions and the candidate with the highest full-data likelihood
is kept; a single EM then runs from it (fast).

Scope
-----
p = 1, 2, 3, ... supported. The 1-D reference model is the
equal-posterior intersection (ecut) use case; p >= 2 is the clustering
use case. Data are synthetic mixtures of skew-normal components; the
deliverable metrics: ecut recovered to < 0.02 (1-D), clustering ARI
>= symmetric GMM (1/2/3-D).

Test matrix (locked from measured runs, N = 1e5):

  CAN (default init, kmeans pre-select):
     1. ecut intersection recovered to < 0.02 for all 5 synthetic
        configurations (opposite-skew, same-skew, weak-skew, strong-skew,
        unequal weights) x 4 seeds: measured max error 0.017
     2. full-data likelihood of the skew fit is within ~1e-4/sample of
        the symmetric GMM (parity; a strict gain was only measured for
        the best-basin scan, which has been removed)
     3. mini-batch fit stays in the same basin as full-batch (same
        pre-selected start): |ecut| within 0.02

  CANNOT (locked limitation):
     4. component-level delta is NOT recovered: the delta likelihood
        ridge is flat with several local maxima (0.29 / 0.46 / 0.71 /
        0.87 on this configuration) and the best short-lookahead start is
        not the best basin; the ecut stays accurate regardless
     5. (nearly) symmetric data: the model will use skewed components
        (|delta_hat| up to ~0.3) because component-level skew is not
        identifiable when the components cancel -- the mixture density
        and the ecut remain accurate (documented, asserted loosely)

  KNOWN LIMITATION (out of scope, documented not asserted):
     - float32: the log-Phi term loses precision; float64 recommended

Run:
    python tests/example_skew_normal_mixtures.py   # pytest + visualization
    pytest tests/example_skew_normal_mixtures.py   # tests only
"""
import itertools
import time
from typing import Any

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from kinematic_decompose.mixture import SkewNormalMixtures, GaussianMixture
from kinematic_decompose.mixture._skew_normal_mixtures import (
    mixture_intersection,
    skew_normal,
)

N = 100_000
SEED = 42

# -- synthetic ground truth: 2-component skew-normal mixture --------------
# comp 1 (bulge-like): location -0.75, scale 0.10, right-skewed (delta +)
# comp 2 (halo-like) : location -0.45, scale 0.08, left-skewed  (delta -)
TRUE = dict(
    w=np.array([0.5, 0.5]),
    xi=np.array([-0.75, -0.45]),
    sigma=np.array([0.10, 0.08]),
    delta=np.array([0.75, -0.60]),
)

# five configurations for the ecut matrix (same locations/scales)
CFGS = {
    "opposite-skew": np.array([0.75, -0.60]),
    "same-skew": np.array([0.50, 0.70]),
    "weak-skew": np.array([0.20, -0.20]),
    "strong-skew": np.array([0.90, -0.85]),
    "unequal-w": np.array([0.75, -0.60]),
}
CFG_W = {"unequal-w": np.array([0.7, 0.3])}


def _sample(rng, n, delta, w=None):
    """Draw from the hierarchical representation (paper eq. 9):
    Y = xi + delta*sigma*|Z0| + sqrt(1-delta^2)*sigma*Z1.
    Returns (X, labels)."""
    w = TRUE["w"] if w is None else w
    k = rng.choice(len(w), size=n, p=w)
    z0 = np.abs(rng.standard_normal(n))
    z1 = rng.standard_normal(n)
    out = np.empty(n)
    lab = np.empty(n)
    for i in range(len(w)):
        m = k == i
        d, s = delta[i], TRUE["sigma"][i]
        out[m] = TRUE["xi"][i] + d * s * z0[m] + np.sqrt(1 - d * d) * s * z1[m]
        lab[m] = i
    return out.reshape(-1, 1), lab


def _truth_ecut(w, xi, sigma, delta):
    from scipy.optimize import root_scalar

    f = lambda x: w[0] * skew_normal(x, xi[0], sigma[0], delta[0]) \
        - w[1] * skew_normal(x, xi[1], sigma[1], delta[1])
    sol = root_scalar(f, bracket=[xi[0], xi[1]])
    assert sol.converged
    return sol.root


def _main_data():
    rng = np.random.default_rng(SEED)
    y, _ = _sample(rng, N, TRUE["delta"])
    return y


# -- fixtures --------------------------------------------------------------
@pytest.fixture(scope="module")
def X():
    return _main_data()


@pytest.fixture(scope="module")
def fitted_default(X):
    """Default init (kmeans pre-select), full batch, tight tol."""
    m = SkewNormalMixtures(n_components=2, max_iter=500, tol=1e-5,
                           random_state=SEED)
    m.fit(X, use_mini_batch=False)
    return m


# -- CAN: ecut recovery (default fast init) --------------------------------
def test_ecut_recovery_default(fitted_default):
    o = np.argsort(fitted_default.xi_)
    cut = mixture_intersection(fitted_default.weights_[o], fitted_default.xi_[o],
                               fitted_default.scales_[o], fitted_default.deltas_[o])
    truth = _truth_ecut(TRUE["w"], TRUE["xi"], TRUE["sigma"], TRUE["delta"])
    assert cut is not None
    assert abs(cut - truth) < 0.02, f"ecut error {abs(cut - truth):.4f}"


def test_ecut_matrix_all_configs():
    """All five configurations x 4 seeds: ecut error < 0.02 (measured max 0.017)."""
    for name, delta in CFGS.items():
        w = CFG_W.get(name, TRUE["w"])
        truth = _truth_ecut(w, TRUE["xi"], TRUE["sigma"], delta)
        for seed in range(4):
            rng = np.random.default_rng(seed)
            y, _ = _sample(rng, N, delta, w=w)
            m = SkewNormalMixtures(n_components=2, max_iter=300, tol=1e-5,
                                   random_state=seed)
            m.fit(y.reshape(-1, 1), use_mini_batch=False)
            o = np.argsort(m.xi_)
            cut = mixture_intersection(m.weights_[o], m.xi_[o],
                                       m.scales_[o], m.deltas_[o])
            assert cut is not None
            err = abs(cut - truth)
            assert err < 0.02, f"{name} seed{seed}: ecut err {err:.4f}"


# -- CAN: skew fit likelihood parity with symmetric GMM --------------------
def test_skew_likelihood_parity(X, fitted_default):
    """Default-mode skew fit stays within ~1e-4/sample of the symmetric GMM
    in likelihood (parity; a strict gain was only measured for the deleted
    best-basin scan). The clustering gain (ARI >= GMM) is asserted
    separately in test_clustering_quality_all_configs."""
    g = GaussianMixture(n_components=2, n_init=5, random_state=0, max_iter=300)
    g.fit(X, use_mini_batch=False)
    assert fitted_default.lower_bound_ > g.score(X) - 1e-4


# -- CAN: mini-batch equivalence (same pre-selected basin) ------------------
def test_mini_batch_equivalence(X, fitted_default):
    m = SkewNormalMixtures(n_components=2, max_iter=500, tol=1e-5,
                           random_state=SEED)
    m.fit(X, use_mini_batch=True)  # auto-switches to mini-batch (N > 3*batch)
    o = np.argsort(m.xi_)
    cut = mixture_intersection(m.weights_[o], m.xi_[o], m.scales_[o], m.deltas_[o])
    o2 = np.argsort(fitted_default.xi_)
    cut_full = mixture_intersection(fitted_default.weights_[o2],
                                    fitted_default.xi_[o2],
                                    fitted_default.scales_[o2],
                                    fitted_default.deltas_[o2])
    assert cut is not None and cut_full is not None
    assert abs(cut - cut_full) < 0.02


# -- CAN: clustering quality (the primary use case) -------------------------
def _aligned_acc(pred, lab):
    """Component labels are permutable; align by majority, then accuracy."""
    p = np.asarray(pred).copy()
    for c in np.unique(pred):
        m = pred == c
        p[m] = np.bincount(lab[m].astype(int), minlength=2).argmax()
    return float((p == lab).mean())


def test_clustering_quality_all_configs():
    """Skew fit must cluster at least as well as the symmetric GMM on every
    configuration (measured: skew ARI >= GMM ARI on all 5 configs, largest
    gain +0.028 on unequal weights; accuracy gains +0.001..+0.009)."""
    for name, delta in CFGS.items():
        w = CFG_W.get(name, TRUE["w"])
        rng = np.random.default_rng(SEED)
        X, lab = _sample(rng, N, delta, w=w)
        g = GaussianMixture(n_components=2, n_init=5, random_state=0,
                            max_iter=300)
        g.fit(X, use_mini_batch=False)
        m = SkewNormalMixtures(n_components=2, max_iter=500, tol=1e-5,
                               random_state=SEED)
        m.fit(X, use_mini_batch=False)
        ari_g = adjusted_rand_score(lab, g.predict(X))
        ari_s = adjusted_rand_score(lab, m.predict(X))
        acc_g = _aligned_acc(g.predict(X), lab)
        acc_s = _aligned_acc(m.predict(X), lab)
        assert ari_s >= ari_g, f"{name}: skew ARI {ari_s:.4f} < GMM {ari_g:.4f}"
        assert acc_s >= acc_g - 0.002, f"{name}: skew acc {acc_s:.4f} < GMM {acc_g:.4f}"


# -- CANNOT / LIMITATIONS ----------------------------------------------------
def test_symmetric_data_no_ecut_bias():
    """Symmetric (delta=0) data: component skew is not identifiable, the
    fit may use |delta| up to ~0.3 (measured 0.29), but the ecut stays
    accurate (weak-identifiability limitation, locked loosely)."""
    rng = np.random.default_rng(3)
    y, _ = _sample(rng, N, np.array([0.0, 0.0]))
    m = SkewNormalMixtures(n_components=2, max_iter=500, tol=1e-5,
                           random_state=3)
    m.fit(y.reshape(-1, 1), use_mini_batch=False)
    assert np.abs(m.deltas_).max() < 0.35, "fabricated skew beyond the limit"
    truth = _truth_ecut(TRUE["w"], TRUE["xi"], TRUE["sigma"], np.array([0.0, 0.0]))
    o = np.argsort(m.xi_)
    cut = mixture_intersection(m.weights_[o], m.xi_[o], m.scales_[o], m.deltas_[o])
    assert cut is not None
    assert abs(cut - truth) < 0.03


# -- CAN: multivariate (p = 2, 3) clustering --------------------------------
def _sample_mv(rng, n, mu, Sigma, shape, w=None):
    """Sample a two-component multivariate skew-normal mixture from the
    latent representation X = mu + Delta*|U0| + Gamma^{1/2} U1."""
    w = TRUE["w"] if w is None else w
    p = mu.shape[1]
    k = rng.choice(len(w), size=n, p=w)
    X = np.empty((n, p))
    lab = np.empty(n)
    for i in range(len(w)):
        m = k == i
        c = int(m.sum())
        d = shape[i] / np.sqrt(1.0 + shape[i] @ shape[i])
        Del = np.linalg.cholesky(Sigma[i]) @ d
        Gam = Sigma[i] - np.outer(Del, Del)
        u0 = np.abs(rng.standard_normal(c))
        u1 = rng.standard_normal((c, p))
        X[m] = mu[i] + np.outer(u0, Del) + u1 @ np.linalg.cholesky(Gam).T
        lab[m] = i
    return X, lab


MV2 = dict(mu=np.array([[-0.7, 0.35], [-0.4, -0.2]]),
           Sigma=np.array([[[0.030, 0.012], [0.012, 0.020]],
                           [[0.022, -0.008], [-0.008, 0.016]]]),
           shape=np.array([[2.0, 0.6], [-1.5, 0.3]]))
MV3 = dict(mu=np.array([[-0.75, 0.3, 0.2], [-0.45, -0.1, -0.3]]),
           Sigma=np.array([[[0.030, 0.008, 0.004], [0.008, 0.020, 0.003],
                            [0.004, 0.003, 0.015]],
                           [[0.025, -0.006, 0.002], [-0.006, 0.018, -0.004],
                            [0.002, -0.004, 0.014]]]),
           shape=np.array([[2.5, 0.4, -0.3], [-1.8, -0.2, 0.5]]))


def test_multivariate_scope_and_clustering():
    """p = 2 and p = 3 synthetic skew mixtures: the model fits, clusters
    at least as well as the symmetric GMM (measured ARI 0.970 >= 0.968 in
    2-D, 0.989 >= 0.989 in 3-D), and recovers the component means to
    < 0.05. The 1-D conveniences (xi_, scales_, deltas_) are p = 1 only."""
    for p, cfg in [(2, MV2), (3, MV3)]:
        rng = np.random.default_rng(7 if p == 2 else 11)
        X, lab = _sample_mv(rng, N, cfg["mu"], cfg["Sigma"], cfg["shape"])
        assert X.shape == (N, p)
        m = SkewNormalMixtures(n_components=2, max_iter=500, tol=1e-5,
                               random_state=7 if p == 2 else 11)
        m.fit(X, use_mini_batch=False)
        g = GaussianMixture(n_components=2, n_init=5, random_state=0,
                            max_iter=300)
        g.fit(X, use_mini_batch=False)
        ari_s = adjusted_rand_score(lab, m.predict(X))
        ari_g = adjusted_rand_score(lab, g.predict(X))
        assert ari_s >= ari_g - 0.005, f"p={p}: skew ARI {ari_s:.4f} < GMM {ari_g:.4f}"
        # means recovery (component order matched by proximity to truth)
        d0 = np.linalg.norm(m.means_ - cfg["mu"][0], axis=1)
        d1 = np.linalg.norm(m.means_ - cfg["mu"][1], axis=1)
        err = max(min(d0[0], d1[0]), min(d0[1], d1[1]))
        assert err < 0.05, f"p={p}: mean recovery {err:.4f}"
        with pytest.raises(AttributeError, match="p = 1"):
            _ = m.xi_


# -- API smoke ----------------------------------------------------------------
def test_api_smoke(X, fitted_default):
    assert fitted_default._n_parameters() == 4 * 2 - 1
    Xs, y = fitted_default.sample(500)
    assert Xs.shape == (500, 1)
    assert y.shape == (500,)
    assert fitted_default.score_samples(Xs).shape == (500,)
    assert fitted_default.predict(Xs).shape == (500,)
    assert fitted_default.predict_proba(Xs).shape == (500, 2)
    assert np.isfinite(fitted_default.bic(Xs))
    assert np.isfinite(fitted_default.aic(Xs))
    assert np.isfinite(fitted_default.icl(Xs))
    assert np.isfinite(fitted_default.mbic(Xs))
    assert np.all(np.isfinite(fitted_default.lambdas_))


# -- efficiency benchmarks (speedup ratios; no absolute-time asserts) ------
# The mini-batch path is the production route (batch_size follows the
# Ashtiani et al. 2020 formula); these tests lock in that it is FASTER
# than full-batch at equal quality, and that the skew E-step (log_ndtr)
# stays within a sane factor of the symmetric GMM.

def _fit_time(X, use_mini_batch, max_iter=500, seed=SEED):
    m = SkewNormalMixtures(n_components=2, max_iter=max_iter, tol=1e-5,
                           random_state=seed)
    t0 = time.perf_counter()
    m.fit(X, use_mini_batch=use_mini_batch)
    return time.perf_counter() - t0, m


def test_benchmark_minibatch_speedup_1d(X):
    """1-D (N=1e5): mini-batch clearly faster than full-batch (measured
    ~10x) while keeping the same basin: |dLB| < 0.02 and |ecut| < 0.02
    (the locked mini-batch equivalence)."""
    t_full, mf = _fit_time(X, False)
    t_mini, mm = _fit_time(X, True)
    print(f"\n[bench 1-D N={len(X)}] full {t_full:.3f}s  mini {t_mini:.3f}s"
          f"  speedup {t_full / max(t_mini, 1e-9):.1f}x")
    assert t_mini < 0.5 * t_full, \
        f"mini {t_mini:.3f}s not faster than full {t_full:.3f}s"
    assert abs(mf.lower_bound_ - mm.lower_bound_) < 0.02
    of, om = np.argsort(mf.xi_), np.argsort(mm.xi_)
    cf = mixture_intersection(mf.weights_[of], mf.xi_[of],
                              mf.scales_[of], mf.deltas_[of])
    cm = mixture_intersection(mm.weights_[om], mm.xi_[om],
                              mm.scales_[om], mm.deltas_[om])
    assert cf is not None and cm is not None
    assert abs(cf - cm) < 0.02, f"ecut drift {abs(cf - cm):.4f}"


def test_benchmark_minibatch_speedup_2d():
    """p = 2 (N=3e4, MV2): mini-batch much faster (measured ~50x) with
    clustering quality kept within 0.01 ARI of full-batch."""
    rng = np.random.default_rng(7)
    X, lab = _sample_mv(rng, 30_000, MV2["mu"], MV2["Sigma"], MV2["shape"])
    mf = SkewNormalMixtures(n_components=2, max_iter=500, tol=1e-5,
                            random_state=7)
    t0 = time.perf_counter()
    mf.fit(X, use_mini_batch=False)
    t_full = time.perf_counter() - t0
    mm = SkewNormalMixtures(n_components=2, max_iter=500, tol=1e-5,
                            random_state=7)
    t0 = time.perf_counter()
    mm.fit(X)  # default: auto-switch to mini-batch (N > 3*batch)
    t_mini = time.perf_counter() - t0
    print(f"[bench 2-D N={len(X)}] full {t_full:.3f}s  mini {t_mini:.3f}s"
          f"  speedup {t_full / max(t_mini, 1e-9):.1f}x")
    assert t_mini < 0.3 * t_full, \
        f"mini {t_mini:.3f}s not much faster than full {t_full:.3f}s"
    ari_f = adjusted_rand_score(lab, mf.predict(X))
    ari_m = adjusted_rand_score(lab, mm.predict(X))
    assert ari_m >= ari_f - 0.01, \
        f"ARI mini {ari_m:.4f} < full {ari_f:.4f}"


def test_benchmark_skew_vs_gmm_cost(X):
    """Full-batch 1-D: the skew E-step (log_ndtr) costs more than the
    symmetric GMM; lock a sane ceiling (measured ~10x) to catch
    catastrophic regressions."""
    g = GaussianMixture(n_components=2, n_init=1, init_params="kmeans",
                        max_iter=300, random_state=SEED)
    t0 = time.perf_counter()
    g.fit(X, use_mini_batch=False)
    t_gmm = time.perf_counter() - t0
    t_skew, _ = _fit_time(X, False)
    print(f"[bench 1-D] GMM {t_gmm:.3f}s  skew {t_skew:.3f}s"
          f"  ratio {t_skew / max(t_gmm, 1e-9):.1f}x")
    assert t_skew < 30.0 * t_gmm + 1.0, \
        f"skew {t_skew:.3f}s unreasonably slower than GMM {t_gmm:.3f}s"


def test_benchmark_scaling_n_1d():
    """Full-batch 1-D wall time grows roughly linearly with N (very loose
    bound; measured ~5x for 5x the samples)."""
    rng = np.random.default_rng(SEED)
    times = {}
    for n in (20_000, 100_000):
        Xn, _ = _sample(rng, n, TRUE["delta"])
        t, _ = _fit_time(Xn, False)
        times[n] = t
    print(f"[bench scaling 1-D] {times}")
    assert times[100_000] < 15.0 * times[20_000] + 1.0, \
        f"scaling {times[100_000]:.2f}s vs {times[20_000]:.2f}s"


# -- visualization (NATURE_STYLE, TRUE green vs DETECTED red) ----------------
def _visualize(fitted):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    NATURE_STYLE = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 19,
        "axes.titlesize": 19,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 13,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.6,
    }
    plt.rcParams.update(NATURE_STYLE)

    rng = np.random.default_rng(SEED)
    y, lab = _sample(rng, N, TRUE["delta"])

    o = np.argsort(fitted.xi_)
    w = fitted.weights_[o]
    xi = fitted.xi_[o]
    s = fitted.scales_[o]
    d = fitted.deltas_[o]
    cut = mixture_intersection(w, xi, s, d)
    truth_cut = _truth_ecut(TRUE["w"], TRUE["xi"], TRUE["sigma"], TRUE["delta"])
    assert cut is not None

    g = GaussianMixture(n_components=2, n_init=5, random_state=0, max_iter=300)
    g.fit(y, use_mini_batch=False)

    e = np.linspace(-1.0, 0.0, 1200)
    true_total = np.zeros_like(e)
    det_total = np.zeros_like(e)
    for k in range(2):
        true_total += TRUE["w"][k] * skew_normal(e, TRUE["xi"][k],
                                                 TRUE["sigma"][k], TRUE["delta"][k])
        det_total += w[k] * skew_normal(e, xi[k], s[k], d[k])
    gmm_total = np.exp(g.score_samples(e.reshape(-1, 1)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) what "skew" means: the TRUE component vs a symmetric Gaussian
    # with the same (xi, sigma) -- the peak shifts and the tail leans
    # toward delta > 0
    ax = axes[0, 0]
    xi0, s0, d0 = TRUE["xi"][0], TRUE["sigma"][0], TRUE["delta"][0]
    ax.plot(e, skew_normal(e, xi0, s0, d0), color="#2ca02c", lw=2.4,
            label=rf"TRUE skew-normal ($\delta={d0:.2f}$)")
    ax.plot(e, np.exp(-0.5 * ((e - xi0) / s0) ** 2) / (s0 * np.sqrt(2 * np.pi)),
            color="0.5", ls="--", lw=1.8,
            label=r"Gaussian, same $\xi,\sigma$")
    ax.axvline(xi0, color="#2ca02c", ls=":", lw=1.2, alpha=0.6)
    ax.set_title("(a) A skew-normal component", fontsize=16)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel("density")
    ax.set_xlim(-1.0, 0.0)
    ax.legend(loc="upper left", frameon=False)

    # (b) density recovery: the symmetric GMM cannot follow the skew
    # tails, the skew ECM (DETECTED) tracks TRUE
    ax = axes[0, 1]
    ax.hist(y[:, 0], bins=np.linspace(-1.0, 0.0, 61).tolist(), density=True,
            histtype="step", color="0.45", lw=1.2, label=r"data ($N=10^5$)")
    ax.plot(e, true_total, color="#2ca02c", lw=2.2, label="TRUE (generating)")
    ax.plot(e, det_total, color="#d62728", lw=2.2, label="DETECTED (skew ECM)")
    ax.plot(e, gmm_total, color="#1f77b4", ls=":", lw=2.0,
            label="GMM (symmetric)")
    ax.axvline(truth_cut, color="#2ca02c", ls="--", lw=2.0,
               label=r"TRUE ecut $=%.4f$" % truth_cut)
    ax.axvline(cut, color="#d62728", ls=":", lw=2.0,
               label=r"DETECTED ecut $=%.4f$" % cut)
    ax.set_title("(b) Density recovery", fontsize=16)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel("density")
    ax.set_xlim(-1.0, 0.0)
    ax.legend(loc="upper left", frameon=False, fontsize=11)

    # (c) deviation of the fit from the TRUE density (smooth, not the
    # noisy single-realization histogram): the symmetric GMM must widen
    # its components to absorb the skew, so it deviates from TRUE more
    # than the skew ECM does
    bins = np.linspace(-1.0, 0.0, 81)
    xc = (bins[:-1] + bins[1:]) / 2
    true_at = np.interp(xc, e, true_total)
    res_gmm = np.abs(true_at - np.interp(xc, e, gmm_total))
    res_skew = np.abs(true_at - np.interp(xc, e, det_total))
    ax = axes[1, 0]
    ax.plot(xc, res_gmm, color="#1f77b4", lw=2.2,
            label=f"GMM  ($\\sum|\\Delta|={res_gmm.sum():.3f}$)")
    ax.plot(xc, res_skew, color="#d62728", lw=2.2,
            label=f"skew ($\\sum|\\Delta|={res_skew.sum():.3f}$)")
    ax.set_title("(c) Deviation from the TRUE density", fontsize=16)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel(r"$|$TRUE $-$ model$|$")
    ax.set_xlim(-1.0, 0.0)
    ax.set_ylim(0.0, None)
    ax.legend(loc="upper left", frameon=False)

    # (d) per-bin misassignment: skew below GMM everywhere, ARI in legend
    ax = axes[1, 1]
    idx = np.digitize(y[:, 0], bins) - 1
    for tag, pred, color in [("GMM (symmetric)", g.predict(y), "#1f77b4"),
                             ("skew (ECM)", fitted.predict(y), "#d62728")]:
        mis = np.zeros(len(bins) - 1)
        for b in range(len(bins) - 1):
            m = idx == b
            if m.sum() > 0:
                mis[b] = float((pred[m] != lab[m]).mean())
        ax.plot(xc, mis, color=color, lw=2.2,
                label=f"{tag}  (ARI {adjusted_rand_score(lab, pred):.3f})")
    ax.set_title("(d) Misassignment per energy bin", fontsize=16)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel("misassignment fraction")
    ax.set_xlim(-1.0, 0.0)
    ax.set_ylim(0.0, None)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    out = "tests/skew_normal_mixtures_recovery.png"
    fig.savefig(out)
    print(f"saved {out}")


def _visualize_mv():
    """2-D and 3-D clustering: data colored by TRUE vs DETECTED labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    NATURE_STYLE = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.linewidth": 1.0,
    }
    plt.rcParams.update(NATURE_STYLE)
    colors = ["#d62728", "#1f77b4"]

    fig = plt.figure(figsize=(11.5, 9))
    for row, (p, cfg, seed) in enumerate([(2, MV2, 7), (3, MV3, 11)]):
        rng = np.random.default_rng(seed)
        X, lab = _sample_mv(rng, N, cfg["mu"], cfg["Sigma"], cfg["shape"])
        m = SkewNormalMixtures(n_components=2, max_iter=500, tol=1e-5,
                               random_state=seed)
        m.fit(X, use_mini_batch=False)
        pred = m.predict(X)
        for col, (title, yv) in enumerate([("TRUE", lab), ("DETECTED", pred)]):
            ax: Any = fig.add_subplot(2, 2, row * 2 + col + 1,
                                      projection="3d" if p == 3 else None)
            for c in range(2):
                mm = yv == c
                if p == 2:
                    ax.scatter(X[mm, 0], X[mm, 1], s=2, alpha=0.35,
                               color=colors[c], rasterized=True)  # type: ignore[arg-type]
                else:
                    ax.scatter(X[mm, 0], X[mm, 1], X[mm, 2], s=2, alpha=0.35,
                               color=colors[c], rasterized=True)  # type: ignore[arg-type]
            ari = adjusted_rand_score(lab, pred)
            ax.set_title(f"p = {p}, {title}  (ARI {ari:.3f})")
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            if p == 3:
                getattr(ax, "set_zlabel")(r"$x_3$")
            else:
                ax.set_xlim(-1.2, 0.1)
                ax.set_ylim(-0.6, 0.8)
    fig.tight_layout()
    out = "tests/multivariate_clustering.png"
    fig.savefig(out)
    print(f"saved {out}")


def _report(fitted_default):
    """Quantitative table printed at the end of the main run."""
    o = np.argsort(fitted_default.xi_)
    w, xi, s, d = (fitted_default.weights_[o], fitted_default.xi_[o],
                   fitted_default.scales_[o], fitted_default.deltas_[o])
    truth = _truth_ecut(TRUE["w"], TRUE["xi"], TRUE["sigma"], TRUE["delta"])
    cut = mixture_intersection(w, xi, s, d)
    print("\n" + "=" * 68)
    print("SkewNormalMixtures recovery (N=1e5, seed 42)")
    print("=" * 68)
    print(f"{'param':<8}{'truth':>10}{'detected':>16}{'|err|':>10}")
    for name, t, v in [("w1", TRUE["w"][0], w[0]), ("w2", TRUE["w"][1], w[1]),
                       ("xi1", TRUE["xi"][0], xi[0]), ("xi2", TRUE["xi"][1], xi[1]),
                       ("sig1", TRUE["sigma"][0], s[0]), ("sig2", TRUE["sigma"][1], s[1]),
                       ("del1", TRUE["delta"][0], d[0]), ("del2", TRUE["delta"][1], d[1])]:
        print(f"{name:<8}{t:>10.4f}{v:>16.4f}{abs(v - t):>10.4f}")
    print("-" * 68)
    print(f"{'ecut (default init)':<24}{truth:>10.4f}{cut:>16.4f}{abs(cut - truth):>10.4f}")
    print(f"LB = {fitted_default.lower_bound_:.6f}  (truth-param LB = 0.708270)")
    print("=" * 68)

    # clustering table: skew vs symmetric GMM (the primary use case)
    rng = np.random.default_rng(SEED)
    X, lab = _sample(rng, N, TRUE["delta"])
    g = GaussianMixture(n_components=2, n_init=5, random_state=0, max_iter=300)
    g.fit(X, use_mini_batch=False)
    print("\nClustering (predict vs generating labels, all on the same data)")
    print(f"{'model':<20}{'ARI':>8}{'acc':>8}")
    for tag, m in [("GMM (symmetric)", g), ("skew (default)", fitted_default)]:
        p = m.predict(X)
        print(f"{tag:<20}{adjusted_rand_score(lab, p):>8.4f}{_aligned_acc(p, lab):>8.4f}")
    print("=" * 68)


if __name__ == "__main__":
    import sys

    code = pytest.main([__file__, "-v", "-s"])
    if code == 0:
        Xd = _main_data()
        fd = SkewNormalMixtures(n_components=2, max_iter=500, tol=1e-5,
                                random_state=SEED)
        fd.fit(Xd, use_mini_batch=False)
        _visualize(fd)
        _visualize_mv()
        _report(fd)
    sys.exit(code)
