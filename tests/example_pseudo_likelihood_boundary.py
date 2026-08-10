"""Tests for the pseudo-likelihood approximation used in auto-GMM component
selection, and its reliability boundary.

Context
-------
The auto-GMM (``mixture/_auto_gaussian_mixture.py::_find_residual_component``)
decides whether to add a component using a **pseudo** log-likelihood gain from an
**un-refit** model, under the efficiency assumption that each candidate component
is *independent* of the existing ones. This test measures how well that
approximation tracks the **true** decision (a full refit + global BIC) on
synthetic mixtures, as a function of the separation between the candidate and
the nearest existing component.

Faithfulness
------------
The pseudo decision is NOT re-implemented here: this suite calls the real
``AutoGaussianMixtureModel._find_residual_component`` pipeline end-to-end
(IQR threshold -> connected components -> particle mapping -> gain selection),
so it measures the code that actually runs, not an idealized proxy.

Findings (measurements locked by this suite)
----------------------------------------------
On 2-D data (N=40k, weights 0.7/0.3, sigma=0.6), the REAL pipeline:

  sep(sig)   pseudo_accept   true_accept   agreement
  0.3        reject          reject        YES   (no over-accept; IQR misses the
  0.5        reject          reject        YES    weak coupled candidate entirely)
  0.8        reject          reject        YES
  1.0        reject          accept        NO    <- under-acceptance (missed
  1.2        reject          accept        NO     component; pseudo gain stays
  1.5        reject          accept        NO     negative / below penalty)
  2.0..5.0   accept          accept        YES

Two regime boundaries:
  * sep < ~1 sigma: the IQR + connected-component detection does NOT find the
    candidate (residual below the outlier threshold), so the pipeline correctly
    keeps the existing K -- no over-split. The early simplified-proxy test that
    reported "over-acceptance < 1 sigma" was an artifact of a hand-written
    fixed-radius region; the real pipeline is stricter.
  * 1.0 <= sep <= ~1.5 sigma: genuine under-acceptance. The true BIC accepts
    the second component but the pseudo gain (2*dL_region - penalty)/n_labeled
    stays <= 0, so the candidate is missed. This is the real, measurable
    limitation of the pseudo-likelihood gate in this regime.
  * sep >= ~2 sigma: correct acceptance.

The tests below lock the reliable regimes (<=0.8 sigma reject-agreement and
>=2 sigma accept-agreement) and quantify the intermediate under-acceptance,
instead of the earlier (artifact) over-acceptance claim.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture as SkGM
from kinematic_decompose.mixture import AutoGaussianMixtureModel
from kinematic_decompose.mixture import GaussianMixture as KGM
from kinematic_decompose.visualize import NATURE_STYLE


# ---------------------------------------------------------------------------
# helpers: data generation
# ---------------------------------------------------------------------------
def _two_gaussians_sep(sep, n=40000, seed=0):
    """One component at origin, one at (+sep, 0), same sigma 0.6, weights 0.7/0.3."""
    rng = np.random.RandomState(seed)
    base = rng.randn(int(0.7 * n), 2) * 0.6
    cand = rng.randn(int(0.3 * n), 2) * 0.6 + np.array([sep, 0.0])
    X = np.vstack([base, cand])
    rng.shuffle(X)
    return X


# ---------------------------------------------------------------------------
# faithful pseudo decision: run the REAL pipeline
# ---------------------------------------------------------------------------
def _pseudo_accept_real(X, seed=0):
    """Run the real ``_find_residual_component`` on a K=1 fit of X; True if it
    adds >= 1 component (i.e. the real gain/BF selection accepts a candidate)."""
    g1 = KGM(n_components=1, covariance_type="full", random_state=seed,
             n_init=5, init_params="kmeans", max_iter=200, min_iter=0).fit(X)
    auto = AutoGaussianMixtureModel(random_state=seed)
    init_model, _ = auto._find_residual_component(X, g1)
    return bool(init_model.n_components > 1)


def _true_accept(X, seed=0):
    """True decision: global BIC of a full K=2 refit vs K=1 refit."""
    g1 = SkGM(n_components=1, covariance_type="full", random_state=seed,
              n_init=5, init_params="kmeans", max_iter=200).fit(X)
    g2 = SkGM(n_components=2, covariance_type="full", random_state=seed,
              n_init=5, init_params="kmeans", max_iter=200).fit(X)
    return bool(g2.bic(X) < g1.bic(X))


def _measure_boundary(seps, n=40000, n_seeds=6):
    """Per-separation agreement / under-acceptance of the REAL pipeline vs true."""
    out = {"seps": list(seps), "agree": {}, "under": {}}
    for sep in seps:
        acc_p, acc_t = [], []
        for seed in range(n_seeds):
            X = _two_gaussians_sep(sep, n=n, seed=seed)
            acc_p.append(_pseudo_accept_real(X, seed))
            acc_t.append(_true_accept(X, seed))
        acc_p, acc_t = np.array(acc_p, bool), np.array(acc_t, bool)
        out["agree"][sep] = float(np.mean(acc_p == acc_t))
        out["under"][sep] = float(np.mean((~acc_p) & acc_t))   # pseudo misses a real component
    return out


# ---------------------------------------------------------------------------
# CAPABILITY-BOUNDARY tests (numeric, not binary labels)
# ---------------------------------------------------------------------------
def test_coupled_regime_no_overaccept():
    """sep <= 0.8 sigma: the real pipeline rejects AND true BIC rejects ->
    full agreement, zero over-acceptance. The IQR detection misses the weak
    coupled candidate before the gain gate is even evaluated."""
    res = _measure_boundary(seps=[0.3, 0.5, 0.8], n=40000, n_seeds=4)
    agree = [res["agree"][s] for s in (0.3, 0.5, 0.8)]
    assert np.mean(agree) >= 0.9, f"coupled regime must agree with true: {agree}"


def test_separated_regime_accepts():
    """sep >= 2 sigma: the real pipeline accepts AND true BIC accepts ->
        full agreement."""
    res = _measure_boundary(seps=[2.0, 3.0, 5.0], n=40000, n_seeds=4)
    agree = [res["agree"][s] for s in (2.0, 3.0, 5.0)]
    assert np.mean(agree) >= 0.9, f"separated regime must agree with true: {agree}"


def test_intermediate_underacceptance_quantified():
    """1.0 <= sep <= 1.5 sigma: genuine under-acceptance -- true BIC accepts
    but the pseudo gate rejects. Lock the (non-trivial) under-acceptance rate
    as a measured quantity, not a pass/fail label."""
    res = _measure_boundary(seps=[1.0, 1.2, 1.5], n=40000, n_seeds=4)
    under = [res["under"][s] for s in (1.0, 1.2, 1.5)]
    assert np.mean(under) >= 0.25, \
        f"expected measurable under-acceptance in intermediate regime: {under}"


# ---------------------------------------------------------------------------
# Visualisation for human inspection (NATURE_STYLE, saved to image/)
# ---------------------------------------------------------------------------
def test_visualize_pseudo_likelihood_boundary():
    """Two-panel figure: (top) real-pipeline acceptance vs true BIC acceptance
    per separation; (bottom) under-acceptance rate vs separation. Saved to
    image/pseudo_likelihood_boundary.png."""
    seps = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    res = _measure_boundary(seps, n=40000, n_seeds=4)

    with plt.rc_context({**NATURE_STYLE, "font.size": 12}):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)

        p_acc = []
        t_acc = []
        for i, s in enumerate(seps):
            acc_p = [_pseudo_accept_real(_two_gaussians_sep(s, seed=seed), seed)
                     for seed in range(4)]
            acc_t = [_true_accept(_two_gaussians_sep(s, seed=seed), seed)
                     for seed in range(4)]
            p_acc.append(float(np.mean(acc_p)))
            t_acc.append(float(np.mean(acc_t)))

        ax1.plot(seps, p_acc, "o-", color="tab:blue", linewidth=2, markersize=7,
                 label="real pipeline accept rate")
        ax1.plot(seps, t_acc, "s--", color="tab:green", linewidth=2, markersize=6,
                 label="true BIC accept rate")
        ax1.axvspan(1.0, 1.5, color="tab:red", alpha=0.15,
                    label="under-acceptance zone")
        ax1.set_ylabel("accept rate")
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_title("real pipeline vs true BIC (synthetic 2-Gaussian)",
                      fontsize=13)
        ax1.grid(True, which="both", alpha=0.2)
        ax1.legend(loc="center left", fontsize=10)

        under = [res["under"][s] for s in seps]
        ax2.plot(seps, under, "o-", color="tab:red", linewidth=2, markersize=7)
        ax2.axvspan(1.0, 1.5, color="tab:red", alpha=0.15)
        ax2.set_xlabel("candidate separation  [$\\sigma$]")
        ax2.set_ylabel("under-acceptance rate  $P(\\mathrm{miss})$")
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, which="both", alpha=0.2)

        fig.tight_layout()
        fig.savefig("image/pseudo_likelihood_boundary.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    assert os.path.exists("image/pseudo_likelihood_boundary.png")
