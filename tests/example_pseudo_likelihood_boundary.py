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
from scipy.ndimage import label

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


def _pseudo_gain_real(X, seed=0):
    """Mirror ``_find_residual_component``'s gain computation line by line
    (histogram -> KL residual -> IQR threshold -> connected components ->
    particle mapping -> BF cumulative selection -> MIN_POINTS/MIN_WEIGHT),
    so the scatter plot can use the code's ACTUAL per-candidate gain values.

    Returns (max_positive_gain, n_kept) -- n_kept counts candidates that pass
    the full real selection; decisions match ``_pseudo_accept_real``."""
    N = len(X)
    eoemin, jzojc = X[:, 0], X[:, 1]
    wid0 = 2.0 * np.subtract(*np.percentile(eoemin, [75, 25])) * eoemin.size ** (-1.0/3.0)
    wid1 = 2.0 * np.subtract(*np.percentile(jzojc, [75, 25])) * jzojc.size ** (-1.0/3.0)
    b0 = min(int(np.ptp(eoemin)/wid0), 150); b1 = min(int(np.ptp(jzojc)/wid1), 300)
    x_range = np.percentile(eoemin, [0.1, 99.9]).tolist()
    y_range = np.percentile(jzojc, [0.1, 99.9]).tolist()
    true_prob, x_edges, y_edges = np.histogram2d(eoemin, jzojc, bins=[b0, b1],
                                                 density=True, range=[x_range, y_range])
    xc = 0.5*(x_edges[:-1]+x_edges[1:]); yc = 0.5*(y_edges[:-1]+y_edges[1:])
    xx, yy = np.meshgrid(xc, yc, indexing='ij')
    pts = np.column_stack([xx.ravel(order='C'), yy.ravel(order='C')])
    g1 = KGM(n_components=1, covariance_type="full", random_state=seed,
             n_init=5, init_params="kmeans", max_iter=200, min_iter=0).fit(X)
    model_prob = np.exp(np.nan_to_num(g1.score_samples(pts))).reshape(xx.shape)
    dx, dy = x_edges[1]-x_edges[0], y_edges[1]-y_edges[0]
    ba = dx*dy
    tc, mc = true_prob*N*ba, model_prob*N*ba
    eps = 1.0
    ratio = (tc+eps)/(mc+eps)
    dL = np.clip(np.nan_to_num(tc*np.log(ratio) - (tc-mc)), 0, None)
    pos = dL[dL>0]
    if len(pos) == 0:
        return 0.0, 0
    q1, q3 = np.percentile(pos, [25, 75]); iqr = q3 - q1
    mask = dL > (q3 + 1.5*iqr)
    labs, _ = label(mask)
    ix = np.digitize(eoemin, x_edges)-1; iy = np.digitize(jzojc, y_edges)-1
    valid = (ix>=0)&(ix<len(x_edges)-1)&(iy>=0)&(iy<len(y_edges)-1)
    pl = np.zeros(N, int); pl[valid] = labs[ix[valid], iy[valid]]
    rids = np.unique(labs); rids = rids[rids != 0]
    n_lab = np.sum(pl > 0)
    if n_lab == 0 or len(rids) == 0:
        return 0.0, 0
    k = 0.5*2*3 + 2 + 1
    pen = k*np.log(N)
    dsums = np.array([dL[labs == r].sum() for r in rids])
    gains = (2.0*dsums - pen)/n_lab
    order = np.argsort(gains)[::-1]
    sg = gains[order]
    pg = sg[sg > 0]
    if len(pg) == 0:
        return 0.0, 0
    cum_ratio = np.exp(0.5*(np.cumsum(pg) - np.sum(pg)))
    ns = int(np.searchsorted(cum_ratio, 0.951, side='right'))
    ns = min(ns, len(pg))
    q1g, q3g = np.percentile(pg, [25, 75]); iqrg = q3g - q1g
    if ns == 0 and pg[0] > q3g + 1.5*iqrg:
        ns = 1
    if ns == 0 and len(pg) > 0:
        ns = 1
    sel = np.argsort(gains)[::-1][:ns]
    kept = 0
    for r in sel:
        npts = int(np.sum(pl == rids[r]))
        if npts >= 10 and npts/N >= 0.01:
            kept += 1
    return float(pg[0]), kept


def _true_improvement(X, seed=0):
    """True per-particle log-likelihood improvement of K=2 refit over K=1:
    2*[ln L(K=2) - ln L(K=1)] / N  [nats particle^-1]."""
    g1 = SkGM(n_components=1, covariance_type="full", random_state=seed,
              n_init=5, init_params="kmeans", max_iter=200).fit(X)
    g2 = SkGM(n_components=2, covariance_type="full", random_state=seed,
              n_init=5, init_params="kmeans", max_iter=200).fit(X)
    return float(2.0 * (g2.score(X) - g1.score(X)))


def _measure_boundary(seps, n=20000, n_seeds=3):
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
    res = _measure_boundary(seps=[0.3, 0.5, 0.8])
    agree = [res["agree"][s] for s in (0.3, 0.5, 0.8)]
    assert np.mean(agree) >= 0.9, f"coupled regime must agree with true: {agree}"


def test_separated_regime_accepts():
    """sep >= 2 sigma: the real pipeline accepts AND true BIC accepts ->
        full agreement."""
    res = _measure_boundary(seps=[2.0, 3.0, 5.0])
    agree = [res["agree"][s] for s in (2.0, 3.0, 5.0)]
    assert np.mean(agree) >= 0.9, f"separated regime must agree with true: {agree}"


def test_intermediate_underacceptance_quantified():
    """1.0 <= sep <= 1.5 sigma: genuine under-acceptance -- true BIC accepts
    but the pseudo gate rejects. Lock the (non-trivial) under-acceptance rate
    as a measured quantity, not a pass/fail label."""
    res = _measure_boundary(seps=[1.0, 1.2, 1.5])
    under = [res["under"][s] for s in (1.0, 1.2, 1.5)]
    assert np.mean(under) >= 0.25, \
        f"expected measurable under-acceptance in intermediate regime: {under}"


# ---------------------------------------------------------------------------
# Visualisation for human inspection (NATURE_STYLE, saved to image/)
# ---------------------------------------------------------------------------
def test_visualize_pseudo_likelihood_boundary():
    """Scatter: estimated (pseudo-likelihood) gain vs TRUE per-particle
    log-likelihood improvement, one point per (separation, seed) trial;
    point colour encodes the candidate separation in sigma.

    The diagonal (y = x) is the ideal where the pseudo gain equals the true
    improvement. Points below the diagonal under-estimate the gain; points
    with x > 0 but y <= 0 are the under-acceptance cases (true BIC accepts,
    the real pipeline rejects). Saved to image/pseudo_likelihood_boundary.png.
    """
    seps = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    n_seeds = 3
    xs, ys, cs = [], [], []
    for sep in seps:
        for seed in range(n_seeds):
            X = _two_gaussians_sep(sep, n=20000, seed=seed)
            gain, kept = _pseudo_gain_real(X, seed)
            true_imp = _true_improvement(X, seed)
            xs.append(true_imp)
            ys.append(gain)
            cs.append(sep)
    xs, ys, cs = np.array(xs), np.array(ys), np.array(cs)

    with plt.rc_context({**NATURE_STYLE, "font.size": 12}):
        fig, ax = plt.subplots(figsize=(7, 6))
        lo = min(xs.min(), ys.min()) - 0.05
        hi = max(xs.max(), ys.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1,
                label=r"$y = x$ (perfect approximation)")
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=60, edgecolor="k",
                        linewidth=0.5, zorder=3)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel(r"True per-particle improvement, "
                      r"$2\,\Delta\ln L\,/\,N$  [nat particle$^{-1}$]")
        ax.set_ylabel(r"Pseudo gain, $(2\,\Delta L_{KL} - k\ln N)/n_{\mathrm{lab}}$  "
                      r"[nat particle$^{-1}$]")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, which="both", alpha=0.2)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(r"separation  [$\sigma$]")
        ax.legend(loc="upper left", fontsize=10)
        fig.tight_layout()
        fig.savefig("image/pseudo_likelihood_boundary.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    assert os.path.exists("image/pseudo_likelihood_boundary.png")
