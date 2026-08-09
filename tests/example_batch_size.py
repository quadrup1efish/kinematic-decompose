"""Tests for the mini-batch size selection theory.

The auto-derived ``batch_size`` (when ``batch_size=None``) follows the
near-optimal sample complexity of learning a k-Gaussian mixture in R^d to
total-variation error eps (Ashtiani, Ben-David, Harvey, Liaw, Mehrabian &
Plan 2020, "Near-optimal Sample Complexity Bounds for Robust Learning of
Gaussian Mixtures via Compression Schemes", Theorem 1.5):

    n = polylog(kd/eps) * k*d^2 / eps^2

without the high-probability polylog factor by default (``use_polylog=False``;
expectation-level precision is empirically sufficient for mini-batch/full-batch
convergence equivalence). These tests pin the formula, its scaling in (k, d,
eps), the fit-time auto-derivation, the explicit-override precedence, and the
mini-batch/full-batch convergence equivalence under the auto-derived size.
"""
import numpy as np
import matplotlib.pyplot as plt

from kinematic_decompose.mixture import GaussianMixture
from kinematic_decompose.mixture._base import BaseMixture
from kinematic_decompose.visualize import NATURE_STYLE


def _two_gaussians_1d(n_samples=100000, seed=0):
    """Two well-separated 1-D Gaussians with fixed seed (deterministic)."""
    return _two_gaussians_d(1, n_samples=n_samples, separation=4.0, seed=seed)


def _two_gaussians_d(d, n_samples=100000, separation=4.0, sigma=0.5, seed=0):
    """Two d-dim Gaussians, means at +/-separation/2 along every axis.

    separation is in units of sigma: separation=4 (default) is well
    separated; separation=1 is a hard (barely separable) mixture.
    """
    rng = np.random.RandomState(seed)
    n1 = int(0.6 * n_samples)
    n2 = n_samples - n1
    mu = separation / 2
    X = np.vstack([
        rng.randn(n1, d) * sigma - mu,
        rng.randn(n2, d) * sigma + mu,
    ])
    return X


def _true_density_d(pts, d, separation=4.0, sigma=0.5):
    """Ground-truth d-dim mixture density (diagonal covariances):
    0.6*N(-mu, sigma^2 I) + 0.4*N(+mu, sigma^2 I)."""
    from scipy.stats import norm
    mu = separation / 2
    return (0.6 * np.prod(norm.pdf(pts, -mu, sigma), axis=1)
            + 0.4 * np.prod(norm.pdf(pts, +mu, sigma), axis=1))


def _tv_between_gmms(m1, m2, lo=-6.0, hi=6.0, n_grid=4000):
    """Total-variation distance between two fitted 1-D GMM densities.

    TV(P, Q) = (1/2) * int |p(x) - q(x)| dx, evaluated by rectangle
    quadrature on a uniform grid (the same eps as Ashtiani et al. 2020:
    a distribution-level error, unlike a parameter gap).
    """
    grid = np.linspace(lo, hi, n_grid)[:, None]
    p1 = np.exp(m1.score_samples(grid))
    p2 = np.exp(m2.score_samples(grid))
    dx = grid[1, 0] - grid[0, 0]
    return 0.5 * np.sum(np.abs(p1 - p2)) * dx


def _tv_to_true_density(m, d=1, separation=4.0, lo=-6.0, hi=6.0):
    """TV distance between a fitted d-dim GMM and the ground-truth density
    (the strict meaning of eps in Ashtiani et al. 2020). Rectangle
    quadrature on a uniform grid: 4000 points (1-D), 200x200 (2-D), 40^3
    (3-D) keep ~4e4-6e4 evaluation points per dimension."""
    n_side = {1: 4000, 2: 200, 3: 40}.get(d, 40)
    axes = [np.linspace(lo, hi, n_side)] * d
    grid = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([g.ravel() for g in grid], axis=1)
    p_model = np.exp(m.score_samples(pts))
    p_true = _true_density_d(pts, d, separation=separation)
    cell = (2 * hi / (n_side - 1)) ** d
    return 0.5 * np.sum(np.abs(p_model - p_true)) * cell


# ---------------------------------------------------------------------------
# Formula pinning: S = ceil(k*d^2 / eps^2), polylog optional
# ---------------------------------------------------------------------------
def test_suggested_batch_size_formula_table():
    """The auto size equals ceil(k*d^2/eps^2) on a reference table."""
    cases = [
        (6, 3, 0.05, 21600),   # typical galaxy problem size
        (2, 1, 0.05, 800),     # 2 components in 1-D
        (2, 2, 0.05, 3200),    # 2 components in 2-D
        (6, 3, 0.10, 5400),    # looser TV error -> 4x smaller batch
        (1, 3, 0.05, 3600),    # single component: k=1
    ]
    for k, d, eps, expected in cases:
        got = BaseMixture.suggested_batch_size(k, d, eps)
        assert got == expected, f"({k},{d},{eps}): got {got}, expected {expected}"


def test_suggested_batch_size_is_int_ceil():
    """Result is an integer ceiling, never a float or floor."""
    s = BaseMixture.suggested_batch_size(6, 3, 0.03)
    assert isinstance(s, int)
    expected = np.ceil(6 * 3 ** 2 / 0.03 ** 2)
    assert s == expected


def test_polylog_variant():
    """use_polylog=True multiplies by ln(kd/(eps*delta)) (first order)."""
    import math
    s_plain = BaseMixture.suggested_batch_size(6, 3, 0.05)
    s_poly = BaseMixture.suggested_batch_size(6, 3, 0.05, delta=0.05,
                                              use_polylog=True)
    factor = max(1.0, math.log(6 * 3 / (0.05 * 0.05)))
    assert s_poly == int(np.ceil(s_plain * factor))
    # conservative high-probability factor, but still feasible for N ~ 5e6
    assert s_plain < s_poly < 5e6


def test_polylog_off_by_default():
    """Default use_polylog=False: no logarithmic inflation."""
    s = BaseMixture.suggested_batch_size(6, 3, 0.05)
    assert s == 21600


# ---------------------------------------------------------------------------
# Scaling laws
# ---------------------------------------------------------------------------
def test_scaling_with_eps():
    """S scales as 1/eps^2: halving eps quadruples the batch."""
    s1 = BaseMixture.suggested_batch_size(6, 3, 0.10)
    s2 = BaseMixture.suggested_batch_size(6, 3, 0.05)
    assert s2 == 4 * s1


def test_scaling_with_k_and_d():
    """S scales linearly in k and quadratically in d."""
    s_k2 = BaseMixture.suggested_batch_size(2, 3, 0.05)
    s_k6 = BaseMixture.suggested_batch_size(6, 3, 0.05)
    assert s_k6 == 3 * s_k2
    s_d1 = BaseMixture.suggested_batch_size(6, 1, 0.05)
    s_d3 = BaseMixture.suggested_batch_size(6, 3, 0.05)
    assert s_d3 == 9 * s_d1


# ---------------------------------------------------------------------------
# Fit-time auto-derivation and override
# ---------------------------------------------------------------------------
def test_fit_auto_derives_batch_size():
    """batch_size=None: derived at fit time from (n_components, d, tv_error)."""
    X = _two_gaussians_1d()
    m = GaussianMixture(n_components=2, init_params="kmeans", max_iter=5,
                        random_state=42)
    m.fit(X, use_mini_batch=True)
    assert m.batch_size == BaseMixture.suggested_batch_size(2, 1, 0.05)  # 800
    assert 3 * m.batch_size < len(X)  # mini-batch path actually active


def test_explicit_batch_size_priority():
    """An explicit batch_size overrides the tv_error-derived default."""
    X = _two_gaussians_1d()
    m = GaussianMixture(n_components=2, batch_size=500, max_iter=5,
                        random_state=42)
    m.fit(X, use_mini_batch=True)
    assert m.batch_size == 500


def test_tv_error_controls_batch_size():
    """Smaller tv_error -> larger auto batch -> closer to full-batch."""
    X = _two_gaussians_1d()
    m1 = GaussianMixture(n_components=2, tv_error=0.10, max_iter=5,
                         random_state=42)
    m2 = GaussianMixture(n_components=2, tv_error=0.02, max_iter=5,
                         random_state=42)
    m1.fit(X, use_mini_batch=True)
    m2.fit(X, use_mini_batch=True)
    assert m2.batch_size == 25 * m1.batch_size  # (0.10/0.02)^2


def test_mini_batch_activation_threshold():
    """CAN: n > 3*batch_size activates mini-batch; CANNOT below it (full)."""
    # tiny data: auto batch (800) > n/3 -> full-batch path, batch unused
    X_small = _two_gaussians_1d(n_samples=1000)
    m = GaussianMixture(n_components=2, init_params="kmeans", max_iter=5,
                        random_state=42)
    m.fit(X_small, use_mini_batch=True)
    assert m.batch_size == 800           # still derived
    assert 3 * m.batch_size > len(X_small)  # mini-batch not activated


# ---------------------------------------------------------------------------
# Convergence equivalence under the auto-derived size
# ---------------------------------------------------------------------------
def test_mini_full_convergence_equivalence_auto_batch():
    """The auto-derived batch (tv_error=0.05) yields a mini-batch model whose
    TV distance to both the full-batch model and the ground-truth density is
    below the requested error."""
    X = _two_gaussians_1d(n_samples=300000)
    m_full = GaussianMixture(n_components=2, init_params="kmeans",
                             max_iter=60, min_iter=40, random_state=42)
    m_mini = GaussianMixture(n_components=2, init_params="kmeans",
                             max_iter=60, min_iter=40, random_state=42,
                             batch_size=None)
    m_full.fit(X, use_mini_batch=False)
    m_mini.fit(X, use_mini_batch=True)

    tv_full = _tv_between_gmms(m_full, m_mini)
    tv_true = _tv_to_true_density(m_mini)
    assert tv_full < 0.05, f"TV(mini, full) = {tv_full:.4f} > tv_error=0.05"
    assert tv_true < 0.05, f"TV(mini, true) = {tv_true:.4f} > tv_error=0.05"


# ---------------------------------------------------------------------------
# Extended theory checks: dimensions, separability, N-invariance, polylog
# ---------------------------------------------------------------------------
def test_tv_dimensions_1_2_3():
    """TV(mini, true) stays below tv_error in 1-D, 2-D and 3-D under the
    auto-derived batch (d enters the formula, so each d gets its own size)."""
    for d in (1, 2, 3):
        X = _two_gaussians_d(d, n_samples=100000)
        m = GaussianMixture(n_components=2, init_params="kmeans",
                            max_iter=60, min_iter=30, random_state=42,
                            batch_size=None)
        m.fit(X, use_mini_batch=True)
        expected = BaseMixture.suggested_batch_size(2, d, 0.05)
        tv = _tv_to_true_density(m, d=d)
        assert m.batch_size == expected, f"d={d}: batch {m.batch_size}"
        assert tv < 0.05, f"d={d}: TV(mini,true)={tv:.4f} > 0.05"


def test_hard_separation_tv_decays_with_batch():
    """The -1/2 law is a worst-case upper envelope: TV decays with batch
    size for BOTH separations, and the barely separable mixture (sep=1
    sigma, closer to the worst case) tracks the -1/2 slope within a factor
    ~2. Note the TV absolute value is not comparable across distributions
    (a sep=1 mixture is easier for a 2-GMM to approximate)."""
    X_easy = _two_gaussians_d(1, n_samples=200000, separation=4.0)
    X_hard = _two_gaussians_d(1, n_samples=200000, separation=1.0)
    tvs_easy, tvs_hard = [], []
    for S in (800, 3200, 12800):
        m_easy = GaussianMixture(n_components=2, init_params="kmeans",
                                 max_iter=80, min_iter=40, random_state=42,
                                 batch_size=S).fit(X_easy, use_mini_batch=True)
        m_hard = GaussianMixture(n_components=2, init_params="kmeans",
                                 max_iter=80, min_iter=40, random_state=42,
                                 batch_size=S).fit(X_hard, use_mini_batch=True)
        tvs_easy.append(_tv_to_true_density(m_easy, separation=4.0))
        tvs_hard.append(_tv_to_true_density(m_hard, separation=1.0))
    # monotone decay in S for both separations
    assert tvs_easy[0] > tvs_easy[-1], f"easy={tvs_easy}"
    assert tvs_hard[0] > tvs_hard[-1], f"hard={tvs_hard}"
    # hard mixture stays within ~2x of the -1/2 theory line through its
    # midpoint (the worst-case slope is an upper envelope, not a floor)
    S_list = (800, 3200, 12800)
    theory = tvs_hard[1] * np.sqrt(S_list[1] / np.array(S_list))
    assert all(t < 2 * th for t, th in zip(tvs_hard, theory)), \
        f"hard={tvs_hard} theory={theory}"


def test_batch_size_independent_of_n():
    """CAN: the auto batch is derived from (k, d, eps) only; for a fixed
    batch size the TV error does NOT grow with N (mini-batch property)."""
    tvs = []
    for n in (50000, 300000):
        X = _two_gaussians_d(1, n_samples=n)
        m = GaussianMixture(n_components=2, init_params="kmeans",
                            max_iter=60, min_iter=40, random_state=42,
                            batch_size=800).fit(X, use_mini_batch=True)
        tvs.append(_tv_to_true_density(m))
    # same batch, 6x more data: TV should not grow (certainly not 2x)
    assert tvs[1] < 2 * tvs[0], f"TV(N=5e4)={tvs[0]:.4f} TV(N=3e5)={tvs[1]:.4f}"


def test_polylog_batch_is_more_conservative():
    """The polylog-inflated batch yields TV <= the plain batch's TV."""
    X = _two_gaussians_1d(n_samples=200000)
    m_plain = GaussianMixture(n_components=2, init_params="kmeans",
                              max_iter=60, min_iter=40, random_state=42,
                              batch_size=None).fit(X, use_mini_batch=True)
    m_poly = GaussianMixture(n_components=2, init_params="kmeans",
                             max_iter=60, min_iter=40, random_state=42,
                             tv_error=0.05, use_polylog=True,
                             batch_size=None).fit(X, use_mini_batch=True)
    assert m_poly.batch_size > m_plain.batch_size
    tv_plain = _tv_to_true_density(m_plain)
    tv_poly = _tv_to_true_density(m_poly)
    assert tv_poly <= tv_plain + 1e-6, \
        f"TV plain={tv_plain:.4f} polylog={tv_poly:.4f}"


def test_multicomponent_equivalence():
    """k=3 mixture: the auto-derived batch keeps TV(mini, true) below the
    requested error (k enters the formula)."""
    rng = np.random.RandomState(0)
    n1, n2, n3 = 50000, 30000, 20000
    X = np.concatenate([
        rng.randn(n1) * 0.4 - 2.5, rng.randn(n2) * 0.5 + 0.5,
        rng.randn(n3) * 0.4 + 3.0])[:, None]
    # ground truth density for the k=3 mixture
    from scipy.stats import norm
    def p_true_3(pts):
        return (0.5 * norm.pdf(pts, -2.5, 0.4)
                + 0.3 * norm.pdf(pts, 0.5, 0.5)
                + 0.2 * norm.pdf(pts, 3.0, 0.4))
    m = GaussianMixture(n_components=3, init_params="kmeans",
                        max_iter=80, min_iter=40, random_state=42,
                        batch_size=None)
    m.fit(X, use_mini_batch=True)
    expected = BaseMixture.suggested_batch_size(3, 1, 0.05)
    assert m.batch_size == expected, f"k=3 batch {m.batch_size}"
    grid = np.linspace(-6, 6, 4000)[:, None]
    p_model = np.exp(m.score_samples(grid))
    dx = grid[1, 0] - grid[0, 0]
    tv = 0.5 * np.sum(np.abs(p_model - p_true_3(grid[:, 0]))) * dx
    assert tv < 0.05, f"k=3 TV={tv:.4f} > 0.05"


# ---------------------------------------------------------------------------
# Visualisation for human inspection (NATURE_STYLE, saved to image/)
# ---------------------------------------------------------------------------
def test_visualize_batch_size_theory():
    """Three-panel figure (d = 1, 2, 3): theory vs measurement per panel.

    x = batch size S (log, 1e2..1e5), y = TV(mini, ground-truth density)
    (log). Theory: eps ~ sqrt(k*d^2/S) (Ashtiani et al. 2020 minimax rate,
    inverted), drawn as a -1/2 slope dashed line through the well-separated
    median midpoint. Measured: TV(mini, true) vs S, repeated N_REPEATS times
    per (d, separation, S) point -> median with a 16-84% band (stable
    curves). Well-separated (sep = 4 sigma, blue) and barely separable
    (sep = 1 sigma, red); the auto-derived point (tv_error=0.05) is starred
    per panel (S=800/3200/7200 for d=1/2/3). Saved to
    image/batch_size_selection.png (same style as the scaling fig).
    """
    N_REPEATS = 3
    S_LIST = [100, 300, 1000, 3000, 10000, 30000, 100000]  # 1e2 .. 1e5
    panels = [(1, 1000000), (2, 400000), (3, 400000)]  # (d, n) keep 3*S<N
    with plt.rc_context({**NATURE_STYLE, "font.size": 13}):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)

        med_e_all = med_h_all = None
        for ax, (d, n) in zip(axes, panels):
            X_easy = _two_gaussians_d(d, n_samples=n, separation=4.0)
            X_hard = _two_gaussians_d(d, n_samples=n, separation=1.0)
            med_e, lo_e, hi_e = [], [], []
            med_h, lo_h, hi_h = [], [], []
            for S in S_LIST:
                te, th = [], []
                for i in range(N_REPEATS):
                    for sep, tv_list, X in ((4.0, te, X_easy), (1.0, th, X_hard)):
                        m = GaussianMixture(n_components=2,
                                            init_params="kmeans",
                                            max_iter=50, min_iter=25,
                                            random_state=42 + i,
                                            batch_size=S)
                        m.fit(X, use_mini_batch=True)
                        tv_list.append(_tv_to_true_density(
                            m, d=d, separation=sep))
                te, th = np.array(te), np.array(th)
                med_e.append(np.median(te))
                lo_e.append(np.percentile(te, 16))
                hi_e.append(np.percentile(te, 84))
                med_h.append(np.median(th))
                lo_h.append(np.percentile(th, 16))
                hi_h.append(np.percentile(th, 84))
            med_e, med_h = np.array(med_e), np.array(med_h)
            med_e_all, med_h_all = med_e, med_h
            S_arr = np.array(S_LIST, dtype=float)

            # measured: sep=4 (blue) and sep=1 (red), median + 16-84% band
            ax.loglog(S_arr, med_e, "tab:blue", marker="o", linewidth=2.2,
                      markersize=6, label="sep = 4$\\sigma$")
            ax.fill_between(S_arr, lo_e, hi_e, color="tab:blue", alpha=0.15)
            ax.loglog(S_arr, med_h, "tab:red", marker="s", linewidth=2.2,
                      markersize=6, label="sep = 1$\\sigma$")
            ax.fill_between(S_arr, lo_h, hi_h, color="tab:red", alpha=0.15)

            # theory slope -1/2 through the easy median midpoint
            mid = len(S_LIST) // 2
            S_th = np.logspace(np.log10(S_LIST[0]), np.log10(S_LIST[-1]), 50)
            ax.loglog(S_th, med_e[mid] * np.sqrt(S_LIST[mid] / S_th),
                      "k--", linewidth=1.8, alpha=0.7,
                      label="theory $\\varepsilon \\propto \\sqrt{k\\,d^2/S}$")
            # slope annotation (above the theory line, clear of curves)
            ax.annotate("$-1/2$",
                        xy=(S_th[-2], med_e[mid] * np.sqrt(S_LIST[mid] / S_th[-2])),
                        xytext=(S_th[-2] * 0.5,
                                med_e[mid] * np.sqrt(S_LIST[mid] / S_th[-2]) * 2.5),
                        fontsize=13, color="k", alpha=0.7)

            # auto-derived point per panel
            auto_S = BaseMixture.suggested_batch_size(2, d, 0.05)
            auto_tv = np.interp(np.log10(auto_S), np.log10(S_arr), np.log10(med_e))
            ax.plot(auto_S, 10 ** auto_tv, "*", color="k", markersize=16,
                    label="auto (tv_error=0.05)" if d == 1 else None)
            ax.annotate(f"S={auto_S}", xy=(auto_S, 10 ** auto_tv),
                        xytext=(auto_S * 0.9, 10 ** auto_tv * 1.6),
                        fontsize=11, color="k")

            ax.set_xlabel("batch size $S$")
            ax.set_title(f"$d$={d}  (auto $S$={auto_S})", fontsize=14)
            ax.grid(True, which="both", alpha=0.2)

        axes[0].set_ylabel("TV(mini, true)")
        h, l = axes[0].get_legend_handles_labels()
        # dedupe (auto label only on d=1)
        seen, hl = set(), []
        for hh, ll in zip(h, l):
            if ll not in seen:
                seen.add(ll); hl.append((hh, ll))
        fig.legend([x[0] for x in hl], [x[1] for x in hl],
                   loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4,
                   fontsize=11, framealpha=0.9)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig("image/batch_size_selection.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # soft assertions: figure exists; median TV decays with S; largest batch
    # below the requested error
    import os
    assert os.path.exists("image/batch_size_selection.png")
    assert med_e_all[0] > med_e_all[-1], f"easy medians {med_e_all}"
    assert med_h_all[0] > med_h_all[-1], f"hard medians {med_h_all}"
    assert min(med_e_all) < 0.05   # largest batch below the requested error
    assert min(med_h_all) < 0.05


# ---------------------------------------------------------------------------
# Parameter scan of the batch-size formula (pure evaluation, no fitting)
# ---------------------------------------------------------------------------
def test_visualize_batch_size_parameter_scan():
    """Three-panel figure: how the auto batch size S = k*d^2/eps^2
    (Ashtiani et al. 2020, Thm 1.5, polylog off) varies with each of its
    three parameters (pure formula evaluation, no fitting -> fast):

    left:   S vs d (k=6), eps in {0.01, 0.02, 0.05, 0.1} -> S ~ d^2
    middle: S vs k (d=3), same eps                         -> S ~ k
    right:  S vs eps for (k,d) in {(6,3), (2,1), (15,5)}   -> S ~ 1/eps^2
    The default reference point (k=6, d=3, eps=0.05 -> S=21600) is starred.
    Saved to image/batch_size_scan.png (NATURE_STYLE, like the other figs).
    """
    with plt.rc_context({**NATURE_STYLE, "font.size": 13}):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
        eps_list = [0.01, 0.02, 0.05, 0.1]
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(eps_list)))

        # ---- left: S vs d (k=6) ----
        ax = axes[0]
        d_range = np.arange(1, 6)
        for eps, c in zip(eps_list, colors):
            S = [BaseMixture.suggested_batch_size(6, d, eps) for d in d_range]
            ax.semilogy(d_range, S, "o-", color=c, linewidth=2, markersize=5,
                        label=f"$\\varepsilon$={eps}")
        ax.set_xlabel("dimension $d$  (k=6)")
        ax.set_ylabel("batch size $S$")
        ax.set_xticks(d_range)
        ax.grid(True, which="both", alpha=0.2)
        ax.annotate("$S \\propto d^2$", xy=(1.02, 2e5), fontsize=13)
        ax.plot(3, 21600, "*", color="k", markersize=16)
        ax.annotate("default $S$=21600", xy=(3, 21600),
                    xytext=(3.1, 21600 * 2.5), fontsize=11)

        # ---- middle: S vs k (d=3) ----
        ax = axes[1]
        k_range = np.arange(1, 16)
        for eps, c in zip(eps_list, colors):
            S = [BaseMixture.suggested_batch_size(k, 3, eps) for k in k_range]
            ax.semilogy(k_range, S, "o-", color=c, linewidth=2, markersize=5,
                        label=f"$\\varepsilon$={eps}")
        ax.set_xlabel("components $k$  (d=3)")
        ax.set_xticks(k_range[::2])
        ax.grid(True, which="both", alpha=0.2)
        ax.annotate("$S \\propto k$", xy=(12, 3000), fontsize=13)

        # ---- right: S vs eps for (k,d) variants ----
        ax = axes[2]
        eps_range = np.logspace(-2, -1, 40)  # 0.01 .. 0.1
        for (k, d), c in zip([(6, 3), (2, 1), (15, 5)],
                             ["tab:blue", "tab:orange", "tab:green"]):
            S = [BaseMixture.suggested_batch_size(k, d, e) for e in eps_range]
            ax.loglog(eps_range, S, color=c, linewidth=2,
                      label=f"k={k}, d={d}")
        ax.set_xlabel("TV error $\\varepsilon$")
        ax.grid(True, which="both", alpha=0.2)
        ax.annotate("$S \\propto 1/\\varepsilon^2$", xy=(0.06, 1.5e6),
                    fontsize=13)

        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=4, fontsize=11, framealpha=0.9)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig("image/batch_size_scan.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    import os
    assert os.path.exists("image/batch_size_scan.png")

