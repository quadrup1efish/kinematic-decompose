"""
Functional tests for get_Ecut - the energy-cut algorithm used in kinematic decomposition.

get_Ecut(eb, masses) separates bound spheroidal components (low energy) from
the disk (high energy) by locating the valley bottom in the energy (eoemin)
histogram (FindMin: sign change of the histogram slope).

Every scenario is a *synthetic combination* of simple distributions, so the
true valley is analytically known. Each scenario declares its sub-distribution
components, which are drawn directly in the visualization (colored curves
overlaid on the histogram).

Test matrix (verified against the ORIGINAL algorithm, 20 seeds each):

  CAN detect - valley distributions (FindMin finds the local minimum)
     1. bimodal           : Gaussian + Gaussian          -> valley between peaks
     2. trimodal          : 3 Gaussians                  -> deepest valley
     3. peak -> plateau   : Gaussian + Uniform           -> peak/plateau boundary
     4. peak-plateau-peak : G + U + G                    -> Ecut inside plateau
     5. double valley     : 3 Gaussians                  -> deeper valley chosen
     6. mass filter       : 3 Gaussians                  -> deep low-mass valley
                            rejected (E < -0.9, < 5% mass below)

  CANNOT detect -> returns 0
     7. unimodal          : single Gaussian              -> no valley
     8. monotonic up/down : linear ramps                 -> no valley
     9. tiny right tail   : valley with too few particles on the right
    10. valley at window edge : valley truncated by M_E = quantile(0.9)

  KNOWN LIMITATIONS (locked assertions on the ORIGINAL behavior)
    11. uniform plateau   : pure Uniform -> FALSE POSITIVE (noise-driven
                            fake valley, e.g. -0.41 on seed 0); regression
                            guard so improvements must fix this
    12. seamless tail+peak : flat tail + peak, NO local minimum
    13. low-count plateau  : short Uniform + Gaussian
    14. ramp tail + peak   : linear ramp + Gaussian
    15. decline tail +peak : decreasing ramp + Gaussian
    16. exp tail + peak    : exp(3(E+1)) + Gaussian
    17. wide + narrow      : wide low + narrow high Gaussian (overlap)
    18. plateau + soft peak: Uniform + low Gaussian
        (scenarios 12-18 are two-component mixtures WITHOUT a valley: the
         original algorithm cannot lock onto the junction - hit rates are
         2-11/20 with wide scatter. Assertions LOCK this unreliable behavior)

    19. small sample (N <= 100) -> 0

Run:
    python tests/example_eoemin_cut.py   # pytest + single-figure visualization
    pytest tests/example_eoemin_cut.py   # tests only
"""

import sys
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import pytest

from kinematic_decompose.mixture.util import get_Ecut

N = 30000
HALF = N // 2
THIRD = N // 3
SEED = 0  # default random seed for reproducible synthetic data

_EGRID = np.linspace(-1.0, 0.0, 4000)  # shared energy grid, restricted to [-1, 0]


# =====================================================================
# Sub-distribution PDF helpers (unnormalized, on _EGRID)
# =====================================================================
def _gpdf(e, mean, std, amp=1.0):
    """Gaussian component PDF."""
    return amp * np.exp(-0.5 * ((e - mean) / std) ** 2)


def _updf(e, a, b, h=1.0):
    """Uniform plateau component PDF."""
    return np.where((e >= a) & (e <= b), h, 0.0)


def _rpdf(e, a, b, h_lo, h_hi):
    """Linear ramp component PDF (h_lo at a, h_hi at b)."""
    return np.where((e >= a) & (e <= b), h_lo + (h_hi - h_lo) * (e - a) / (b - a), 0.0)


def _epdf(e, k, cutoff=-0.7):
    """Exponential tail component PDF exp(k(E+1)) up to cutoff."""
    return np.exp(k * (e + 1.0)) * (e <= cutoff)


# =====================================================================
# Data generators - every scenario is an explicit combination
# =====================================================================
def _sample_pdf(rng, e, pdf, n=N):
    """Inverse-CDF sample of a PDF, forced monotone non-decreasing
    (np.maximum.accumulate) -> no local minimum: the tail connects into the
    peak seamlessly."""
    pdf = np.maximum.accumulate(pdf)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, e)


def _gauss(rng, mean, std, n):
    return rng.normal(mean, std, n)


def _uniform(rng, a, b, n):
    return rng.uniform(a, b, n)


def _flat_tail_peak(rng, n=N):
    """Seamless: Uniform(-1.0,-0.7) flat tail + Gaussian(-0.5, 0.08) peak."""
    pdf = _updf(_EGRID, -1.0, -0.7) + 8.0 * _gpdf(_EGRID, -0.5, 0.08)
    return _sample_pdf(rng, _EGRID, pdf, n)


def _ramp_tail_peak(rng, n=N):
    """Seamless: linear ramp tail (0 -> 1 over [-1.0,-0.7]) + Gaussian peak."""
    pdf = _rpdf(_EGRID, -1.0, -0.7, 0.0, 1.0) + 8.0 * _gpdf(_EGRID, -0.5, 0.08)
    return _sample_pdf(rng, _EGRID, pdf, n)


def _decline_tail_peak(rng, n=N):
    """Seamless: decreasing ramp tail (1 -> 0.3) + Gaussian peak."""
    pdf = _rpdf(_EGRID, -1.0, -0.7, 1.0, 0.3) + 8.0 * _gpdf(_EGRID, -0.5, 0.08)
    return _sample_pdf(rng, _EGRID, pdf, n)


def _exp_tail_peak(rng, k=3.0, n=N):
    """Seamless: exponential tail exp(k(E+1)) + Gaussian peak."""
    pdf = _epdf(_EGRID, k) + 8.0 * _gpdf(_EGRID, -0.5, 0.08)
    return _sample_pdf(rng, _EGRID, pdf, n)


def _wide_narrow_peaks(rng, n=N):
    """Two overlapping Gaussians (wide low + narrow high), no valley."""
    pdf = 1.2 * _gpdf(_EGRID, -0.85, 0.25) + 8.0 * _gpdf(_EGRID, -0.5, 0.06)
    return _sample_pdf(rng, _EGRID, pdf, n)


def _soft_peak_plateau(rng, n=N):
    """Seamless: Uniform plateau + LOW Gaussian peak (subtle junction)."""
    pdf = _updf(_EGRID, -1.0, -0.7) + 2.0 * _gpdf(_EGRID, -0.5, 0.12)
    return _sample_pdf(rng, _EGRID, pdf, n)


@dataclass
class Scenario:
    """One synthetic energy distribution.

    components : list[(label, pdf)]
        Sub-distributions (unnormalized PDFs on _EGRID) that make up the
        total distribution - drawn as colored curves in the visualization.
    truth : float | tuple[float, float] | None
        Analytically-known valley position, or a plateau interval [lo, hi]
        in which Ecut is expected. None for scenarios with no valley.
    rejected : float | None
        Optional second valley that the algorithm is expected to reject
        (drawn in gray in the visualization).
    """

    name: str
    components: list[tuple[str, np.ndarray]]
    gen: Callable[[np.random.RandomState], np.ndarray]
    truth: Optional[Union[float, tuple[float, float]]] = None
    tol: float = 0.10
    rejected: Optional[float] = None
    seed: int = SEED
    kind: str = ""
    total_peak: float = 0.0  # precomputed after scenario lists are built


# =====================================================================
# CAN detect - with valley (FindMin)
# =====================================================================
DETECTABLE_VALLEY = [
    Scenario(
        name="bimodal",
        components=[
            ("Gaussian(-0.9, 0.06)", _gpdf(_EGRID, -0.9, 0.06)),
            ("Gaussian(-0.4, 0.12)", _gpdf(_EGRID, -0.4, 0.12)),
        ],
        gen=lambda rng: np.concatenate(
            [_gauss(rng, -0.9, 0.06, HALF), _gauss(rng, -0.4, 0.12, HALF)]
        ),
        truth=-0.65,  # valley between the two peaks
        tol=0.10,
    ),
    Scenario(
        name="trimodal",
        components=[
            ("Gaussian(-1.0, 0.05)", _gpdf(_EGRID, -1.0, 0.05)),
            ("Gaussian(-0.7, 0.05)", _gpdf(_EGRID, -0.7, 0.05)),
            ("Gaussian(-0.35, 0.05)", _gpdf(_EGRID, -0.35, 0.05)),
        ],
        gen=lambda rng: np.concatenate(
            [
                _gauss(rng, -1.0, 0.05, THIRD),
                _gauss(rng, -0.7, 0.05, THIRD),
                _gauss(rng, -0.35, 0.05, THIRD),
            ]
        ),
        truth=-0.52,  # deepest valley (its right side holds 2/3 of the mass)
        tol=0.10,
    ),
    Scenario(
        name="peak_to_plateau",
        components=[
            ("Gaussian(-0.8, 0.08)", _gpdf(_EGRID, -0.8, 0.08)),
            ("Uniform(-0.6, -0.4)", _updf(_EGRID, -0.6, -0.4)),
        ],
        gen=lambda rng: np.concatenate(
            [_gauss(rng, -0.8, 0.08, HALF), _uniform(rng, -0.6, -0.4, HALF)]
        ),
        truth=-0.6,  # peak / plateau boundary
        tol=0.05,
    ),
    Scenario(
        name="peak_plateau_peak",
        components=[
            ("Gaussian(-0.95, 0.05)", _gpdf(_EGRID, -0.95, 0.05)),
            ("Uniform(-0.85, -0.75)", _updf(_EGRID, -0.85, -0.75)),
            ("Gaussian(-0.5, 0.08)", _gpdf(_EGRID, -0.5, 0.08)),
        ],
        gen=lambda rng: np.concatenate(
            [
                _gauss(rng, -0.95, 0.05, THIRD),
                _uniform(rng, -0.85, -0.75, THIRD),
                _gauss(rng, -0.5, 0.08, THIRD),
            ]
        ),
        truth=(-0.85, -0.75),  # expect Ecut inside the plateau
        tol=0.05,
    ),
    Scenario(
        name="double_valley",
        components=[
            ("Gaussian(-0.85, 0.05) 12k", _gpdf(_EGRID, -0.85, 0.05)),
            ("Gaussian(-0.65, 0.04) 4k", _gpdf(_EGRID, -0.65, 0.04)),
            ("Gaussian(-0.35, 0.08) 14k", _gpdf(_EGRID, -0.35, 0.08)),
        ],
        gen=lambda rng: np.concatenate(
            [
                _gauss(rng, -0.85, 0.05, 12000),
                _gauss(rng, -0.65, 0.04, 4000),
                _gauss(rng, -0.35, 0.08, 14000),
            ]
        ),
        truth=-0.5,  # deeper of the two valleys
        tol=0.10,
    ),
]

MASS_FILTER = Scenario(
    name="mass_filter",
    components=[
        ("Gaussian(-1.05, 0.015) 500", _gpdf(_EGRID, -1.05, 0.015)),
        ("Gaussian(-0.88, 0.03) 15k", _gpdf(_EGRID, -0.88, 0.03)),
        ("Gaussian(-0.5, 0.1) 14.5k", _gpdf(_EGRID, -0.5, 0.10)),
    ],
    gen=lambda rng: np.concatenate(
        [
            _gauss(rng, -1.05, 0.015, 500),     # low-energy clump, ~1.7% of mass
            _gauss(rng, -0.88, 0.03, 15000),    # main peak
            _gauss(rng, -0.50, 0.10, 14500),    # high-energy peak
        ]
    ),
    truth=-0.77,     # high-energy valley (deep valley at -0.95 is rejected)
    rejected=-0.95,  # deep valley: E < -0.9 and < 5% mass below -> filtered out
    tol=0.05,
)

# =====================================================================
# CANNOT detect -> returns 0
# =====================================================================
UNDETECTABLE = [
    Scenario(
        name="unimodal",
        components=[("Gaussian(-0.6, 0.15)", _gpdf(_EGRID, -0.6, 0.15))],
        gen=lambda rng: _gauss(rng, -0.6, 0.15, N),
    ),
    Scenario(
        name="monotonic_up",
        components=[("Uniform(-1.0, -0.1)", _updf(_EGRID, -1.0, -0.1))],
        gen=lambda rng: np.linspace(-1.0, -0.1, N),
    ),
    Scenario(
        name="monotonic_down",
        components=[("Uniform(-1.0, -0.1)", _updf(_EGRID, -1.0, -0.1))],
        gen=lambda rng: np.linspace(-0.1, -1.0, N),
    ),
    Scenario(
        name="tiny_right_tail",
        components=[
            ("Gaussian(-0.95, 0.05) 15k", _gpdf(_EGRID, -0.95, 0.05)),
            ("Gaussian(-0.3, 0.1) 300", _gpdf(_EGRID, -0.3, 0.1)),
        ],
        gen=lambda rng: np.concatenate(
            [_gauss(rng, -0.95, 0.05, 15000), _gauss(rng, -0.3, 0.1, 300)]
        ),
    ),
    Scenario(
        name="valley_at_window_edge",
        components=[
            ("Gaussian(-0.9, 0.08) 25k", _gpdf(_EGRID, -0.9, 0.08)),
            ("Gaussian(-0.2, 0.05) 5k", _gpdf(_EGRID, -0.2, 0.05)),
        ],
        gen=lambda rng: np.concatenate(
            [_gauss(rng, -0.9, 0.08, 25000), _gauss(rng, -0.2, 0.05, 5000)]
        ),
        seed=2,  # empirically, this seed keeps the valley just at the window edge
    ),
]

# =====================================================================
# KNOWN LIMITATIONS - assertions LOCK the original algorithm's behavior.
# Update these if the algorithm is improved.
# =====================================================================
UNIFORM_PLATEAU = Scenario(
    name="uniform_plateau",
    components=[("Uniform(-1.0, -0.1)", _updf(_EGRID, -1.0, -0.1))],
    gen=lambda rng: _uniform(rng, -1.0, -0.1, N),
)

NO_VALLEY_UNRELIABLE = [
    Scenario(
        name="seamless_tail_peak",
        components=[
            ("Uniform(-1.0, -0.7)", _updf(_EGRID, -1.0, -0.7)),
            ("Gaussian(-0.5, 0.08) x8", _gpdf(_EGRID, -0.5, 0.08, 8.0)),
        ],
        gen=lambda rng: _flat_tail_peak(rng),
        truth=-0.7,  # plateau / peak junction (algorithm misses it)
        tol=0.12,
        kind="unreliable",
    ),
    Scenario(
        name="low_count_plateau",
        components=[
            ("Uniform(-0.9, -0.7) 2k", _updf(_EGRID, -0.9, -0.7)),
            ("Gaussian(-0.55, 0.08) 15k", _gpdf(_EGRID, -0.55, 0.08)),
        ],
        gen=lambda rng: np.concatenate(
            [_uniform(rng, -0.9, -0.7, 2000), _gauss(rng, -0.55, 0.08, 15000)]
        ),
        truth=-0.7,
        tol=0.12,
        kind="unreliable",
    ),
    Scenario(
        name="ramp_tail_peak",
        components=[
            ("linear ramp 0->1", _rpdf(_EGRID, -1.0, -0.7, 0.0, 1.0)),
            ("Gaussian(-0.5, 0.08) x8", _gpdf(_EGRID, -0.5, 0.08, 8.0)),
        ],
        gen=lambda rng: _ramp_tail_peak(rng),
        truth=-0.7,
        tol=0.12,
        kind="unreliable",
    ),
    Scenario(
        name="decline_tail_peak",
        components=[
            ("linear decline 1->0.3", _rpdf(_EGRID, -1.0, -0.7, 1.0, 0.3)),
            ("Gaussian(-0.5, 0.08) x8", _gpdf(_EGRID, -0.5, 0.08, 8.0)),
        ],
        gen=lambda rng: _decline_tail_peak(rng),
        truth=-0.7,
        tol=0.12,
        kind="unreliable",
    ),
    Scenario(
        name="exp_tail_peak",
        components=[
            ("exp(3(E+1))", _epdf(_EGRID, 3.0)),
            ("Gaussian(-0.5, 0.08) x8", _gpdf(_EGRID, -0.5, 0.08, 8.0)),
        ],
        gen=lambda rng: _exp_tail_peak(rng),
        truth=-0.7,
        tol=0.12,
        kind="unreliable",
    ),
    Scenario(
        name="wide_narrow_peaks",
        components=[
            ("Gaussian(-0.85, 0.25) x1.2", _gpdf(_EGRID, -0.85, 0.25, 1.2)),
            ("Gaussian(-0.5, 0.06) x8", _gpdf(_EGRID, -0.5, 0.06, 8.0)),
        ],
        gen=lambda rng: _wide_narrow_peaks(rng),
        truth=-0.62,
        tol=0.10,
        kind="unreliable",
    ),
    Scenario(
        name="soft_peak_plateau",
        components=[
            ("Uniform(-1.0, -0.7)", _updf(_EGRID, -1.0, -0.7)),
            ("Gaussian(-0.5, 0.12) x2", _gpdf(_EGRID, -0.5, 0.12, 2.0)),
        ],
        gen=lambda rng: _soft_peak_plateau(rng),
        truth=-0.7,
        tol=0.12,
        kind="unreliable",
    ),
]


# Precompute total_peak of each scenario (avoid recomputing per panel)
for _sc in (
    DETECTABLE_VALLEY + [MASS_FILTER] + UNDETECTABLE
    + [UNIFORM_PLATEAU] + NO_VALLEY_UNRELIABLE
):
    _sc.total_peak = float(np.sum([p for _, p in _sc.components], axis=0).max())


# =====================================================================
# CAN detect - with valley (FindMin)
# =====================================================================
@pytest.mark.parametrize("sc", DETECTABLE_VALLEY, ids=lambda s: s.name)
def test_detectable_valley(sc: Scenario):
    """Ecut should land near the analytically-known valley."""
    rng = np.random.RandomState(sc.seed)
    eb = sc.gen(rng)
    cut = get_Ecut(eb, np.ones(len(eb)))
    assert sc.truth is not None
    if isinstance(sc.truth, tuple):
        lo, hi = sc.truth
        assert lo - sc.tol <= cut <= hi + sc.tol, (
            f"{sc.name}: Ecut={cut}, expected inside [{lo}, {hi}]"
        )
    else:
        assert abs(cut - sc.truth) < sc.tol, (
            f"{sc.name}: Ecut={cut}, expected ~ {sc.truth}"
        )


def test_mass_filter_rejects_deep_small_valley():
    """Deep valley with E < -0.9 and < 5% mass below it must be rejected.

    Valley A ~ -0.95: E < -0.9, ~2% mass below  -> rejected (default Mmin=0.05)
    Valley B ~ -0.77: E > -0.9                   -> selected
    Control:        Mmin=0.0 disables the filter -> valley A selected again.
    """
    rng = np.random.RandomState(MASS_FILTER.seed)
    eb = MASS_FILTER.gen(rng)
    mass = np.ones(len(eb))
    assert np.mean(eb < -0.95) < 0.05, "precondition: valley A must hold < 5% mass"

    cut = get_Ecut(eb, mass)
    assert cut > -0.9, f"deep valley A was NOT rejected: Ecut={cut}"
    assert abs(cut - MASS_FILTER.truth) < MASS_FILTER.tol, f"Ecut={cut}"

    cut0 = get_Ecut(eb, mass, Mmin=0.0)
    assert cut0 < -0.95, f"Mmin=0 should reselect deep valley A: Ecut={cut0}"


# =====================================================================
# CANNOT detect -> returns 0
# =====================================================================
@pytest.mark.parametrize("sc", UNDETECTABLE, ids=lambda s: s.name)
def test_undetectable(sc: Scenario):
    """Distributions without a detectable valley must return 0."""
    rng = np.random.RandomState(sc.seed)
    eb = sc.gen(rng)
    assert get_Ecut(eb, np.ones(len(eb))) == 0, f"{sc.name}"


def test_uniform_plateau_false_positive():
    """LIMITATION (locked): a pure Uniform has no valley, but Poisson noise
    produces a fake one - the ORIGINAL algorithm returns a spurious Ecut
    (e.g. -0.41 on seed 0). A fixed algorithm should return 0 here."""
    rng = np.random.RandomState(UNIFORM_PLATEAU.seed)
    eb = UNIFORM_PLATEAU.gen(rng)
    cut = get_Ecut(eb, np.ones(len(eb)))
    assert cut != 0, "uniform plateau should produce a false positive in the original algorithm"


@pytest.mark.parametrize("sc", NO_VALLEY_UNRELIABLE, ids=lambda s: s.name)
def test_no_valley_unreliable(sc: Scenario):
    """LIMITATION (locked): two-component mixtures WITHOUT a valley cannot be
    separated by the original valley-search algorithm - the junction is not
    found reliably (< 90% hit rate over 20 seeds, wide scatter)."""
    vals = np.array(
        [
            get_Ecut(
                (eb := sc.gen(np.random.RandomState(s))),
                np.ones(len(eb)),
            )
            for s in range(20)
        ]
    )
    assert isinstance(sc.truth, float)
    hits = np.sum(np.abs(vals - sc.truth) < sc.tol)
    spread = vals.max() - vals.min()
    assert hits < 18, f"{sc.name}: unexpectedly reliable ({hits}/20)"
    assert spread > 0.15, f"{sc.name}: unexpectedly stable (spread={spread:.2f})"


def test_small_sample_no_cut():
    """N = 100: too few particles for a reliable valley -> 0."""
    rng = np.random.RandomState(SEED)
    eb = np.concatenate([_gauss(rng, -0.9, 0.05, 30), _gauss(rng, -0.4, 0.1, 70)])
    assert get_Ecut(eb, np.ones(len(eb))) == 0


def test_less_than_100_particles():
    """N < 100: early return of 0."""
    eb = np.full(99, -0.5)
    assert get_Ecut(eb, np.ones(99)) == 0


# =====================================================================
# Visualization - ONE figure: sub-distributions drawn as curves, TRUE vs
# DETECTED cut. No per-panel titles, no composition text at the bottom.
# =====================================================================
def visualize():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    groups = [
        ("CAN detect - with valley\n(green = TRUE, red = DETECTED)",
         DETECTABLE_VALLEY + [MASS_FILTER]),
        ("CANNOT detect\n(no valley -> Ecut = 0)",
         UNDETECTABLE + [UNIFORM_PLATEAU]),
        ("NO valley - unreliable\n(locked limitation)",
         NO_VALLEY_UNRELIABLE),
    ]

    n_rows = len(groups)
    n_cols = max(len(scs) for _, scs in groups)  # 7
    fig = plt.figure(figsize=(3.5 * n_cols + 2.0, 3.5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.30,
                  left=0.085, right=0.985, top=0.90, bottom=0.06)

    for i, (title, scs) in enumerate(groups):
        fig.text(0.013, 1 - (i + 0.5) / n_rows, title, rotation=90,
                 va="center", ha="left", fontsize=9.5, fontweight="bold",
                 color="darkslategray")
        for j, sc in enumerate(scs):
            ax = fig.add_subplot(gs[i, j])
            rng = np.random.RandomState(sc.seed)
            eb = sc.gen(rng)
            cut = get_Ecut(eb, np.ones(len(eb)))

            lo = max(np.quantile(eb, 0.01), -1.0)
            hi = min(np.quantile(eb, 0.9), 0.0)
            nb = 80
            counts, edges = np.histogram(eb, bins=nb, range=(lo, hi))
            centers = 0.5 * (edges[:-1] + edges[1:])
            binwidth = (hi - lo) / nb
            # normalized to [0, 1]: histogram peak and total sub-distribution
            # peak are both scaled to 1 so they match in height
            counts_norm = counts / counts.max()
            ax.bar(centers, counts_norm, width=binwidth, color="steelblue",
                   alpha=0.40, edgecolor="none")

            # sub-distributions on the SAME [0, 1] scale: each curve is
            # normalized by the peak of the total distribution (their sum)
            for label, p in sc.components:
                curve = p / sc.total_peak
                ax.plot(_EGRID, curve, lw=1.5, label=label)

            # TRUE answer (analytically known)
            if sc.truth is not None:
                if isinstance(sc.truth, tuple):
                    ax.axvspan(sc.truth[0], sc.truth[1], color="green", alpha=0.18)
                    ax.axvline(sc.truth[0], color="green", ls="--", lw=1.3)
                    ax.axvline(sc.truth[1], color="green", ls="--", lw=1.3)
                else:
                    ax.axvline(sc.truth, color="green", ls="--", lw=1.5)
            if sc.rejected is not None:
                ax.axvline(sc.rejected, color="gray", ls=":", lw=1.3)

            # DETECTED answer
            if cut != 0:
                ax.axvline(cut, color="red", lw=2.0,
                           label=f"DETECTED = {cut:.3f}")
            else:
                ax.text(0.97, 0.95, "Ecut = 0", transform=ax.transAxes,
                        ha="right", va="top", color="red", fontsize=8)

            ax.legend(fontsize=6, loc="upper left", framealpha=0.7)
            ax.set_yticks([])
            ax.set_xlim(-1.0, 0.0)  # energy window fixed to [-1, 0]

    fig.suptitle(
        "get_Ecut on synthetic energy distributions\n"
        "colored curves = sub-distributions, green dashed = TRUE valley, red = DETECTED cut",
        fontsize=12, fontweight="bold",
    )
    fig.show()
    plt.show()  # block until all figures are closed (interactive mode)


if __name__ == "__main__":
    rc = pytest.main([__file__, "-v", "--tb=short"])
    visualize()
    sys.exit(rc)
