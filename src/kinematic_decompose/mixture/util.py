"""Energy-cut estimators and mixture-decomposition helpers.

Energy cuts
    get_Ecut_skewt   two-component skew-t crossing (pipeline default)
    get_Ecut         FindMin histogram-valley estimate; also the fallback used
                     when the skew-t components are unresolvable
    FindMin, RefineMin   valley locator and refinement, used by get_Ecut

Decomposition helpers
    JEHistogram                 Abadi-like eps-symmetry split
    decompose                   GMM components -> class labels on a galaxy
    decompose_mixture_model     grouped GMM parameters of a fitted model
    save_structure_properties   flat per-structure property record of a sim

Small helpers
    hist_bin_fd, separation_index, MAX_RADIUS
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq, minimize
from scipy.signal import argrelmax
from scipy.special import gamma
from scipy.stats import t as student_t

MAX_RADIUS = 10  # kpc, radial extent of the reference potential grid


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def hist_bin_fd(x):
    """Freedman-Diaconis bin width of ``x``."""
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)


def separation_index(mode1, mode2, sigma1, sigma2):
    """Ashman separation coefficient (Ashman, Bird & Zepf 1994, AJ 108:2348):
    sep = |mode2 - mode1| / sqrt(sigma1^2 + sigma2^2).

    Criterion (user-decided): sep >= 1 => the two peaks are separated by
    more than one combined sigma -> a RESOLVABLE bimodal signal; sep < 1
    unresolvable (unimodal / one component covers the other). Use as a
    reporting/screening metric for two-component decompositions.
    """
    return abs(mode2 - mode1) / np.sqrt(sigma1 * sigma1 + sigma2 * sigma2)


# ---------------------------------------------------------------------------
# Energy cut: FindMin valley estimator
# ---------------------------------------------------------------------------
def get_Ecut(eb, masses, nbins=25, M_bin=400, m_bin=80, toll=1.5, shrink=2,
             Mmin=0.05, Emin=-0.9):
    """Estimate the eoemin cut as the deepest mass-relevant valley of ``eb``.

    The valley is located by ``FindMin`` on a growing bin count and then
    narrowed by ``RefineMin``. Returns ``0`` (the "no cut found" sentinel) for
    samples with fewer than 100 particles.
    """
    if len(eb) < 100:
        return 0

    # Bin count grows with the particle number between the m_bin/M_bin limits.
    NbinMax = max(min(int(0.5 * np.sqrt(len(eb))), M_bin), m_bin)

    # Exclude the outer tail of bound particles when searching the first time.
    M_E = np.quantile(eb, 0.9)
    m_E = np.quantile(eb, 0.01)
    Ecut, E_val = FindMin(eb, m_E, M_E, nbins)

    # No minimum, or the only minimum too close to -1 (possible GC?):
    # repeat the search over the full energy range.
    if len(Ecut) == 0 or (len(Ecut) == 1 and Ecut < Emin):
        M_E = np.max(eb)
        Ecut, E_val = FindMin(eb, m_E, M_E, nbins)

    # Tolerance window around each candidate minimum, used by the refinement.
    D = (M_E - m_E) / float(nbins)
    lb = Ecut - toll * D
    rb = Ecut + toll * D
    if len(Ecut) <= 1:
        nbins = NbinMax + 1  # single candidate: the refinement loop is skipped

    while nbins < NbinMax:
        nbins = shrink * nbins
        D = D / shrink
        pos_E_refined, val_refined = FindMin(eb, m_E, M_E, nbins)
        # For each original minimum keep the deepest refined minimum inside its
        # tolerance window, and add that value to the original one so that
        # spurious shallow minima cannot win the final argmin.
        EcutTEMP = []
        E_valTEMP = []
        for i, v in enumerate(E_val):
            window = (pos_E_refined <= rb[i]) * (pos_E_refined >= lb[i])
            pTEMP = pos_E_refined[window]
            vTEMP = val_refined[window]
            if len(pTEMP) > 0:
                EcutTEMP.append(pTEMP[np.argmin(vTEMP)])
                E_valTEMP.append(v + np.min(vTEMP))

        Ecut = np.array(EcutTEMP)
        E_val = np.array(E_valTEMP)

        if len(Ecut) <= 1:
            break

        lb = Ecut - toll * D
        rb = Ecut + toll * D

    if len(Ecut) == 0:
        return 0

    # Prefer minima that are either above Emin or carry at least Mmin of the
    # total mass, then take the deepest of those.
    total_mass = np.sum(masses)
    rel_filt = [
        bool(np.sum(masses[eb < E]) / total_mass >= Mmin or E >= Emin)
        for E in Ecut
    ]
    if len(Ecut[rel_filt]) == 0:
        Ecut = Ecut[np.argmin(E_val)]
    else:
        Ecut = Ecut[rel_filt][np.argmin(E_val[rel_filt])]
    return RefineMin(eb, Ecut, D, (M_E - m_E) / NbinMax, shrink)


def FindMin(q, m_E, M_E, nbins):
    """Valley positions and counts of the ``nbins``-bin histogram of ``q``
    restricted to [m_E, M_E]."""
    # Minimum particle count for a reliable Jcirc decomposition.
    if len(q) >= 1e4:
        Npart_min = 1000
    elif 1e3 <= len(q) < 1e4:
        Npart_min = 100
    else:
        Npart_min = 10

    MinPart = max(Npart_min, 0.01 * len(q))
    arr = q[(q >= m_E) * (q <= M_E)]
    hist = np.histogram(arr, bins=np.linspace(m_E, M_E, nbins))

    # A valley is a sign change of the histogram increment on both sides.
    diff = hist[0][1:] - hist[0][:-1]
    left = diff[:-1]
    right = diff[1:]
    id_E = np.where(((left < 0) * (right >= 0)) + ((left <= 0) * (right > 0)))

    # Keep only valleys with enough particles on their right-hand side.
    rev_cumsum = np.flip(np.cumsum(np.flip(hist[0])))
    id_E = id_E[0][rev_cumsum[id_E[0] + 1] > MinPart]
    # Drop valleys that are not local: both neighbouring bins must be higher.
    local = []
    for ids in id_E:
        keep = True
        if len(hist[0]) > ids + 3:
            keep = keep and bool(hist[0][ids + 3] > hist[0][ids + 1])
        if ids > 0:
            keep = keep and bool(hist[0][ids - 1] > hist[0][ids + 1])
        local.append(keep)

    id_E = id_E[local]
    return 0.5 * (hist[1][id_E + 2] + hist[1][id_E + 1]), hist[0][id_E + 1]


def RefineMin(q, Vmin, D, Dmin, shrink):
    """Narrow the valley position ``Vmin`` of ``q`` by re-histogramming
    around it until the bin width falls below ``Dmin``."""
    arr = np.empty(0)
    if D <= Dmin:
        if len(q) >= 1e4:
            coe = 0.5
        elif 1e3 <= len(q) < 1e4:
            coe = 2
        else:
            coe = 2.5
        while len(arr) == 0:
            m_E = Vmin - coe * D
            M_E = Vmin + coe * D
            arr = q[(q >= m_E) * (q <= M_E)]
            coe = coe + 0.2
        Vmin = np.median(arr)

    while D > Dmin:
        if len(q) >= 1e3:
            coe = 1.5
        else:
            coe = 4
        m_E = max(Vmin - coe * D, q.min() + D)
        M_E = Vmin + coe * D
        D = D / shrink
        arr = q[(q >= m_E) * (q <= M_E)]
        hist = np.histogram(arr, bins=np.arange(m_E, M_E, D))
        nonzero = np.where(hist[0] != 0)[0]
        if nonzero.size == 0:
            break  # empty histogram: nothing left to refine
        hist_min = hist[0][nonzero].min()
        pid = np.where(hist[0] == hist_min)[0][0]
        arr = arr[(arr >= hist[1][pid]) * (arr <= hist[1][pid + 1])]
        # The refined position is the median energy inside the deepest bin.
        Vmin = np.median(arr)
    return Vmin


# ---------------------------------------------------------------------------
# Energy cut: two-component skew-t estimator
# ---------------------------------------------------------------------------
SMOOTH_FACTOR = 2.0
_GAMMA_LUT = None


def _t_pdf_c(nu):
    """Student-t normalisation constant, looked up on a precomputed grid
    (``nu`` is clipped to [2, 50])."""
    global _GAMMA_LUT
    if _GAMMA_LUT is None:
        nu_grid = np.arange(2.0, 50.01, 0.01)
        _GAMMA_LUT = (
            nu_grid,
            gamma(0.5 * (nu_grid + 1.0))
            / (np.sqrt(nu_grid * np.pi) * gamma(0.5 * nu_grid)),
        )
    nu_grid, constants = _GAMMA_LUT
    return float(np.interp(float(np.clip(nu, 2.0, 50.0)), nu_grid, constants))


def _skew_t_pdf(x, m, s, a, nu):
    """Azzalini skew-t density."""
    z = (x - m) / s
    t_pdf = _t_pdf_c(nu) * (1.0 + z * z / nu) ** (-0.5 * (nu + 1.0))
    argument = a * z * np.sqrt((nu + 1.0) / (nu + z * z))
    return 2.0 * t_pdf * student_t.cdf(argument, nu + 1.0) / s


def _skew_t_mode(m, s, a, nu):
    """Mode of one skew-t component, from a parabolic fit on a dense grid."""
    grid = np.linspace(m - 4.0 * s, m + 4.0 * s, 2001)
    density = _skew_t_pdf(grid, m, s, a, nu)
    index = int(np.argmax(density))
    if 0 < index < len(grid) - 1:
        x0, x1, x2 = grid[index - 1:index + 2]
        y0, y1, y2 = density[index - 1:index + 2]
        numerator = x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)
        denominator = (x2 - x1) * (y0 - y1) + (x1 - x0) * (y2 - y1)
        if abs(denominator) > 1e-30:
            return float(0.5 * numerator / denominator)
    return float(grid[index])


def _two_skew_t(x, params):
    """Density of a two-component skew-t mixture."""
    w1, m1, s1, a1, inv_nu1, m2, s2, a2, inv_nu2 = params
    nu1 = 1.0 / max(inv_nu1, 1e-6)
    nu2 = 1.0 / max(inv_nu2, 1e-6)
    return (
        w1 * _skew_t_pdf(x, m1, s1, a1, nu1)
        + (1.0 - w1) * _skew_t_pdf(x, m2, s2, a2, nu2)
    )


def _init_skew_t_params(xc, h_sm, m_E, M_E, de, eb=None, masses=None, seed=0, h0=None):
    """Initialise the two skew-t components, preferring the peaks on either
    side of the FindMin valley, then raw histogram peaks, then fixed quantile
    positions."""
    rng = np.random.RandomState(seed)
    peak_indices = argrelmax(h_sm)[0]
    peak_indices = peak_indices[h_sm[peak_indices] > 0.05 * h_sm.max()]
    if eb is not None:
        try:
            cut = get_Ecut(eb, masses)
            if cut is not None and cut != 0.0 and m_E < cut < M_E:
                left = peak_indices[xc[peak_indices] < cut]
                right = peak_indices[xc[peak_indices] > cut]
                if left.size and right.size:
                    i_left = left[np.argmax(h_sm[left])]
                    i_right = right[np.argmax(h_sm[right])]
                    m1, m2 = xc[i_left], xc[i_right]
                    if m1 < m2 and m2 - m1 > 2 * de:
                        s_guess = max(de, 0.5 * (m2 - m1))
                        return [0.5, m1, s_guess, 0.0, m2, s_guess, 0.0]
        except Exception:
            pass
    if len(peak_indices) >= 2:
        peak_indices = peak_indices[np.argsort(-h_sm[peak_indices])][:2]
        i_left, i_right = sorted(peak_indices)
        if xc[i_right] - xc[i_left] > 2 * de:
            s_guess = max(de, 0.5 * (xc[i_right] - xc[i_left]))
            return [0.5, xc[i_left], s_guess, 0.0, xc[i_right], s_guess, 0.0]
    if h0 is not None:
        top = np.argsort(h0)[::-1][:2]
        i_left, i_right = sorted(top)
        if xc[i_right] - xc[i_left] > 3 * de:
            s_guess = max(de, 0.5 * (xc[i_right] - xc[i_left]))
            return [0.5, xc[i_left], s_guess, 0.0, xc[i_right], s_guess, 0.0]
    span = M_E - m_E
    for _ in range(20):
        m1 = m_E + span * rng.uniform(0.15, 0.45)
        m2 = m_E + span * rng.uniform(0.55, 0.85)
        if m2 - m1 > 2 * de:
            s_guess = max(de, 0.5 * (m2 - m1))
            return [0.5, m1, s_guess, 0.0, m2, s_guess, 0.0]
    m_mid = 0.5 * (m_E + M_E)
    s_guess = max(de, 0.25 * span)
    return [0.5, m_mid - s_guess, s_guess, 0.0, m_mid + s_guess, s_guess, 0.0]


def _skew_t_separation_index(params):
    """Ashman separation index of the two fitted skew-t modes."""
    _, m1, s1, a1, inv_nu1, m2, s2, a2, inv_nu2 = params
    nu1 = 1.0 / max(inv_nu1, 1e-6)
    nu2 = 1.0 / max(inv_nu2, 1e-6)
    mode1 = _skew_t_mode(m1, s1, a1, nu1)
    mode2 = _skew_t_mode(m2, s2, a2, nu2)
    return float(separation_index(mode1, mode2, s1, s2))


def _order_skew_t_params(params):
    """Order the two components by increasing location."""
    w1, m1, s1, a1, inv_nu1, m2, s2, a2, inv_nu2 = params
    if m1 <= m2:
        return np.array(params, float)
    return np.array(
        [1.0 - w1, m2, s2, a2, inv_nu2, m1, s1, a1, inv_nu1], float
    )


def _fit_skew_t_with_params(eb, masses=None, d_min=1.0):
    """Fit two skew-t components and return ``(params, cut, separation)``.

    ``cut`` is the fitted component-density crossing (between the two component
    modes; the midpoint of the modes when the densities do not cross there).

    The one substitution rule is distinguishability: when the fitted components
    are unresolvable, i.e. the Ashman separation index of their modes is below
    ``d_min`` (``sep < 1`` means the peak separation does not exceed the
    combined component width), the fit carries no usable internal boundary and
    ``cut`` is the ``get_Ecut`` histogram-valley estimate for the same sample.
    ``separation`` is always the fitted separation index, so callers can tell
    the two cases apart (``separation < d_min``). Returns ``(None, None, None)``
    when the sample is too small or has no usable bin width.
    """
    e = np.asarray(eb, float)
    if e.ndim != 1:
        raise ValueError("energies must be a one-dimensional array")
    if masses is not None:
        masses = np.asarray(masses, float)
        if masses.ndim != 1 or masses.size != e.size:
            raise ValueError("masses must be one-dimensional and match energies")
    finite = np.isfinite(e)
    e = e[finite]
    if masses is not None:
        masses = masses[finite]
    if len(e) < 50:
        return None, None, None
    m_E, M_E = e.min(), e.max()
    q75, q25 = np.percentile(e, [75, 25])
    iqr = q75 - q25
    de_fd = 2.0 * iqr * len(e) ** (-1.0 / 3.0)  # Freedman-Diaconis bin width
    if de_fd <= 0 or not np.isfinite(de_fd):
        return None, None, None
    nbins = max(20, min(int(np.ceil((M_E - m_E) / de_fd)), 100))
    h0, edges = np.histogram(e, bins=nbins)
    de = edges[1] - edges[0]
    xc = edges[:-1] + 0.5 * de
    sig_hat = e.std(ddof=1)
    bandwidth_scale = min(sig_hat, iqr / 1.34)
    h_silv = 0.9 * bandwidth_scale * len(e) ** (-1.0 / 5.0)
    smoothing_bins = max(0.5, SMOOTH_FACTOR * h_silv / de)
    h_sm = gaussian_filter1d(h0.astype(float), smoothing_bins)
    h_sm = h_sm / (h_sm.sum() * de)
    sig = np.sqrt(h0 + 1.0) / (h0.sum() * de)
    sig = np.clip(sig, 1e-9, None)

    init = _init_skew_t_params(xc, h_sm, m_E, M_E, de, eb=e, masses=masses, h0=h0)
    init = np.array(list(init[:4]) + [0.1] + list(init[4:]) + [0.1], float)

    # Component locations stay on either side of the FindMin valley; the valley
    # itself is only a bound (it can no longer be returned as the cut).
    fm_split = None
    try:
        m_arr = masses if masses is not None else np.ones(len(e))
        candidate = get_Ecut(e, m_arr)
        if candidate is not None and candidate != 0.0 and m_E < candidate < M_E:
            fm_split = float(candidate)
    except Exception:
        pass

    lower = np.array([0.0, m_E, de, -5.0, 0.02, m_E, de, -5.0, 0.02])
    upper = np.array([1.0, M_E, 0.5 * (M_E - m_E), 5.0, 0.5,
                      M_E, 0.5 * (M_E - m_E), 5.0, 0.5])
    q05 = float(np.quantile(e, 0.05))
    q95 = float(np.quantile(e, 0.95))
    if fm_split is not None:
        lower[1], upper[1] = q05, fm_split
        lower[5], upper[5] = fm_split, q95
    else:
        m1_init, m2_init = float(init[1]), float(init[5])
        split = 0.5 * (m1_init + m2_init) if m2_init > m1_init else 0.5 * (m_E + M_E)
        lower[1], upper[1] = q05, split
        lower[5], upper[5] = split, q95

    # Normalisation of the fit quality: one-component skew-t reference.
    med_e = float(np.median(e))
    std_e = max(de, float(e.std(ddof=1)))
    p_single = [1.0, med_e, std_e, 0.0, 0.1, med_e, std_e, 0.0, 0.1]
    chi2_0 = float(np.sum(((h_sm - _two_skew_t(xc, p_single)) / sig) ** 2))

    def objective(params):
        chi2 = np.sum(((h_sm - _two_skew_t(xc, params)) / sig) ** 2)
        return chi2 / chi2_0

    try:
        result = minimize(
            objective, np.clip(init, lower, upper), method="Nelder-Mead",
            bounds=list(zip(lower, upper)),
            options={"maxiter": 600, "xatol": 1e-5, "fatol": 1e-5},
        )
        params = result.x
    except Exception:
        params = init
    params = _order_skew_t_params(params)

    w1, m1, s1, a1, inv_nu1, m2, s2, a2, inv_nu2 = params
    nu1 = 1.0 / max(inv_nu1, 1e-6)
    nu2 = 1.0 / max(inv_nu2, 1e-6)
    mode1 = _skew_t_mode(m1, s1, a1, nu1)
    mode2 = _skew_t_mode(m2, s2, a2, nu2)
    cut = 0.5 * (mode1 + mode2)

    def difference(x):
        return (
            w1 * _skew_t_pdf(x, m1, s1, a1, nu1)
            - (1.0 - w1) * _skew_t_pdf(x, m2, s2, a2, nu2)
        )

    # The cut is where the two component densities cross, inside the modes.
    try:
        lower_root, upper_root = min(mode1, mode2), max(mode1, mode2)
        if difference(lower_root) * difference(upper_root) <= 0:
            cut = float(brentq(difference, lower_root, upper_root))
    except Exception:
        pass

    separation = _skew_t_separation_index(params)
    if separation < d_min:
        # Unresolvable components: this fit has no internal boundary to report,
        # so use the histogram-valley estimator on the same sample.
        m_arr = masses if masses is not None else np.ones(len(e))
        fallback = get_Ecut(e, m_arr, M_bin=100, m_bin=25, Mmin=0.1)
        if fallback is not None and fallback != 0.0:
            return params, float(fallback), separation
    return params, cut, separation


def get_Ecut_skewt(eb, masses=None, d_min=1.0):
    """Return an ecut estimated from a two-component skew-t fit.

    The returned value is the fitted component-density crossing, unless the two
    fitted components are unresolvable (Ashman separation index below ``d_min``),
    in which case it is the ``get_Ecut`` histogram-valley estimate. Returns
    ``None`` only when the sample is too small/unusable for fitting.
    """
    _, cut, _ = _fit_skew_t_with_params(eb, masses=masses, d_min=d_min)
    return cut


# ---------------------------------------------------------------------------
# Decomposition helpers
# ---------------------------------------------------------------------------
def JEHistogram(E, eps, n_E=20, n_eps=30, seed=42):
    """Abadi-like decomposition with local eps symmetry.

    Assumptions:
    - All counter-rotating particles (eps < 0) belong to spheroid.
    - For each energy bin and each eps bin, the same number of
      co-rotating particles are added to spheroid to ensure
      local symmetry around eps=0.
    - Remaining co-rotating particles belong to disk.
    """
    rng = np.random.default_rng(seed)
    N = len(E)
    sph = np.zeros(N, dtype=bool)

    E_edges = np.linspace(E.min(), E.max(), n_E + 1)

    for i in range(n_E):
        upper_edge = E_edges[i + 1]
        e_mask = (E >= E_edges[i]) & (
            (E < upper_edge) | ((i == n_E - 1) & (E <= upper_edge))
        )
        idx_E = np.flatnonzero(e_mask)
        if idx_E.size == 0:
            continue

        eps_E = eps[idx_E]
        eps_max = np.max(np.abs(eps_E))
        if eps_max == 0:
            sph[idx_E] = True
            continue

        eps_edges = np.linspace(-eps_max, eps_max, n_eps + 1)
        bin_id = np.searchsorted(eps_edges, eps_E, side="right") - 1
        bin_id = np.clip(bin_id, 0, n_eps - 1)
        mid = n_eps // 2

        for k in range(mid):
            neg_bin = mid - 1 - k
            pos_bin = mid + k

            idx_neg = idx_E[bin_id == neg_bin]
            idx_pos = idx_E[bin_id == pos_bin]
            if idx_neg.size == 0:
                continue

            # All counter-rotating particles are spheroid.
            sph[idx_neg] = True
            if idx_pos.size == 0:
                continue

            # Mirror the negative bin into the positive one.
            n_sel = min(idx_neg.size, idx_pos.size)
            chosen = (
                idx_pos if idx_pos.size == n_sel
                else rng.choice(idx_pos, n_sel, replace=False)
            )
            sph[chosen] = True

        # Particles at eps ~ 0 are spheroid as well.
        sph[idx_E[bin_id == mid]] = True

    disk = ~sph
    return sph, disk


def decompose(X, galaxy, model, eoemin_cut, jzojc_cut, predict_method='soft',
              require_bulge_halo=False):
    """Assign the GMM components of ``model`` to kinematic classes.

    Classes: 0 cold disk, 1 warm disk, 2 bulge, 3 halo, 4 counter-rotating
    disk. The labels go to ``galaxy.s['label']`` and the class probabilities to
    ``galaxy.s['prob']``.
    """
    dim = model.means_.shape[1]
    Xd = X[:, :dim]
    # One full-data E-step shared by the labels and the probabilities
    # (soft_predict + predict_proba used to run one each).
    _, log_resp = model._estimate_log_prob_resp(Xd)
    resp = np.exp(log_resp)  # = predict_proba(Xd)
    prob = resp

    if predict_method == 'soft':
        rng = np.random.default_rng(42)  # matches soft_predict's fixed seed
        probs = resp / resp.sum(axis=1, keepdims=True)
        cum_probs = np.cumsum(probs, axis=1)
        rand_vals = rng.random(probs.shape[0])
        labels = (cum_probs >= rand_vals[:, np.newaxis]).argmax(axis=1)
    else:
        labels = np.argmax(log_resp, axis=1)  # argmax(log_resp) == argmax(weighted log prob)

    means = model.means_
    # Class of each component, from its (e, eps) centroid.
    component_class = np.empty(len(means), dtype=np.int64)
    for i, (ec, eta) in enumerate(means[:, 0:2]):
        if eta >= 0.85:
            component_class[i] = 0  # cold disk
        elif eta > jzojc_cut:
            component_class[i] = 1  # warm disk
        elif eta < -jzojc_cut:
            component_class[i] = 4  # counter-rotating disk
        elif eta <= jzojc_cut and ec < eoemin_cut:
            component_class[i] = 2  # bulge
        else:
            component_class[i] = 3  # halo

    galaxy.s['label'] = component_class[labels]
    new_prob = np.zeros((len(prob), 5), dtype=np.float32)
    for old_idx, new_idx in enumerate(component_class):
        new_prob[:, new_idx] += prob[:, old_idx]
    galaxy.s['prob'] = new_prob

    bulge_count = np.sum(galaxy.s['label'] == 2)
    halo_count = np.sum(galaxy.s['label'] == 3)
    if require_bulge_halo:
        original_labels = galaxy.s['label'].copy()
        # Guarantee at least one bulge and one halo particle, taking them from
        # the wrong side of the ecut (and from the JEHistogram spheroid when the
        # model produced neither).
        if bulge_count == 0 and halo_count != 0:
            halo_mask = galaxy.s['label'] == 3
            bulge_candidates = galaxy.s['eoemin'] < eoemin_cut
            to_bulge_mask = halo_mask & bulge_candidates
            if np.any(to_bulge_mask):
                galaxy.s['label'][to_bulge_mask] = 2
        elif bulge_count != 0 and halo_count == 0:
            bulge_mask = galaxy.s['label'] == 2
            halo_candidates = galaxy.s['eoemin'] > eoemin_cut
            to_halo_mask = bulge_mask & halo_candidates
            if np.any(to_halo_mask):
                galaxy.s['label'][to_halo_mask] = 3
        elif bulge_count == 0 and halo_count == 0:
            sph, _ = JEHistogram(galaxy.s['eoemin'], galaxy.s['jzojc'])
            to_bulge_mask = sph & (galaxy.s['eoemin'] < eoemin_cut)
            to_halo_mask = sph & (galaxy.s['eoemin'] > eoemin_cut)
            galaxy.s['label'][to_bulge_mask] = 2
            galaxy.s['label'][to_halo_mask] = 3
        # Reassigned particles are certain members of their new class; retain
        # the original soft posterior for every particle whose label was kept.
        reassigned = galaxy.s['label'] != original_labels
        galaxy.s['prob'][reassigned] = np.eye(5, dtype=np.float32)[
            galaxy.s['label'][reassigned]
        ]
    return galaxy


def decompose_mixture_model(model, eoemin_cut, jzojc_cut, r_jzojc_cut):
    """Group the GMM components of ``model`` into the decomposition classes.

    Returns the weights, means and covariances of the total model, of every
    class, and the three cuts that define the grouping.
    """
    COLD_DISK_THRESHOLD = 0.85

    weights, means, covariances = (
        model.weights_,
        model.means_,
        model.covariances_,
    )
    eta = means[:, 1]
    e = means[:, 0]

    disk_mask = eta > jzojc_cut
    masks = {
        "disk": disk_mask,
        "colddisk": eta >= COLD_DISK_THRESHOLD,
        "warmdisk": disk_mask & (eta < COLD_DISK_THRESHOLD),
        "spheroid": (r_jzojc_cut <= eta) & (eta <= jzojc_cut),
        "counter-rotating disk": eta <= r_jzojc_cut,
    }
    masks["bulge"] = masks["spheroid"] & (e < eoemin_cut)
    masks["halo"] = masks["spheroid"] & (e >= eoemin_cut)

    GMM_dict = {
        "total": {
            "weights": weights,
            "means": means,
            "covariances": covariances,
        }
    }
    for name, mask in masks.items():
        GMM_dict[name] = {
            "weights": weights[mask],
            "means": means[mask],
            "covariances": covariances[mask],
        }
    GMM_dict["eoemin_cut"] = eoemin_cut
    GMM_dict["jzojc_cut"] = jzojc_cut
    GMM_dict["r_jzojc_cut"] = r_jzojc_cut
    return GMM_dict


# Property groups of save_structure_properties: the group name in the returned
# record, the attribute of the sim object, and whether it carries Mass_frac.
_TOTAL_FIELDS = ("M_vir", "R_vir", "V_vir", "T_vir", "spin", "AM")
_DM_FIELDS = _TOTAL_FIELDS + ("vel_disp", "ke")
_SUB_FIELDS = ("spin", "krot", "beta", "AM", "vel_disp", "vr_disp", "vR_disp",
               "vz_disp", "v_circ", "v_rot", "ke", "Mdyn", "Mbary",
               "r50", "R50", "z50", "t50", "shape")
_STRUCTURES = (
    ("star", "s", False),
    ("disk", "disk", True),
    ("colddisk", "colddisk", True),
    ("warmdisk", "warmdisk", True),
    ("spheroid", "spheroid", True),
    ("bulge", "bulge", True),
    ("halo", "halo", True),
)


def save_structure_properties(sim):
    """Return the per-structure property record of ``sim``.

    Every structure contributes mass, kinematics, size and shape; the disk-like
    ones and the spheroid additionally contribute their mass fraction. The gas
    entries are reserved: no gas attributes are wired yet, so they stay empty.
    """
    props = {
        name: {}
        for name in ("total", "dm", "star", "disk", "colddisk", "warmdisk",
                     "spheroid", "bulge", "halo", "gas", "coldgas")
    }

    for field in _TOTAL_FIELDS:
        props["total"][field] = getattr(sim, field)
    for field in _DM_FIELDS:
        props["dm"][field] = getattr(sim.dm, field)

    for name, attr, has_mass_fraction in _STRUCTURES:
        record = getattr(sim, attr)
        target = props[name]
        target["mass"] = record.M_vir
        for field in _SUB_FIELDS:
            target[field] = getattr(record, field)
        if has_mass_fraction:
            target["Mass_frac"] = record.Mass_frac

    return props
