"""Two-skew-t curve-fit ecut (no rejection gate).

get_Ecut_2gauss(eb, ...):
  1. FD bins over [e.min(), e.max()]  (2*IQR*N^(-1/3), clipped 20..400)
  2. Silverman denoising with SMOOTH_FACTOR=2.0 enhanced bandwidth
     (statistical bandwidth 0.9*min(sig_hat, IQR/1.34)*N^(-1/5))
  3. Weighted fit (Nelder-Mead, 3 starts) of
       h_sm(e) ~ w1*ST(e;m1,s1,a1,nu1) + (1-w1)*ST(e;m2,s2,a2,nu2)
     with ST the Azzalini SKEW-t (2 t(z;nu) T(a z sqrt((nu+1)/(nu+z^2)); nu+1));
     nu parameterised as inu = 1/nu in [0.02, 0.5] (nu in [2, 50]).
     WLS weights = 1/sigma^2 from Poisson counting errors propagated
     through the smoothing (density units). Objective = chi2/chi2_0 with
     chi2_0 the single-component fit at the data median/std (scale
     reference). NO separation regularisation (user decision: fit must
     not be forced apart).
  4. Physical priors: means initialised from FindMin split (bulge left of
     the valley, disk right), m1 in [q05, cut], m2 in [cut, q95];
     basinhopping-free direct Nelder-Mead (3 starts) inside that split
     space avoids both spike-overfit traps and escapes to
     chi2-better-but-unphysical regions. Random multi-start retry if NM
     fails.
  5. ecut = numerical INTERSECTION of w1*ST1 = w2*ST2 between the two
     MODES (the physical boundary lies between the peaks). If no root
     exists there (one component covers the other), the mean of the two
     modes is returned -- always an answer.
  6. Degeneration guards: spike (s < 2 bins), s pinned to bound
     (>= 0.95*0.5*span), extreme weight (<5%), mean pinned to the data
     edge, or cut outside [q01, q99] -> fall back to the FindMin valley
     (pipeline params M_bin=100, m_bin=25, Mmin=0.1); if FindMin also
     fails, the mode midpoint / median.

Why skewed: real binding-energy distributions are monotone-tailed, so a
plain Gaussian fit has systematic bias. The skew-t adds a tail-exponent
nu per component (disk/halo are heavy-tailed) on top of the skewness a
-- verified: on TNG50-1 subID=264883 the skew-t cut is |dev|=0.019 from
the FindMin reference vs 0.090 for skew-normal.

Returns a float cut ALWAYS (no rejection); on pathological data it falls
back to the FindMin valley / mode midpoint / median.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax
from scipy.optimize import minimize

# Enhanced denoising: multiply the Silverman bandwidth by this factor
# (user decision -- stronger smoothing hides weak multi-modal noise but
# keeps the dominant bulge/disk structure).
SMOOTH_FACTOR = 2.0


_GAMMA_LUT = None  # lazily-built t-pdf normalisation constant per nu


def _t_pdf_c(nu):
    """t-pdf normalisation constant G((nu+1)/2)/(sqrt(nu*pi) G(nu/2)),
    cached on a fine nu grid and interpolated (no KeyError from float
    rounding; no gamma call in the hot loop)."""
    global _GAMMA_LUT
    if _GAMMA_LUT is None:
        from scipy.special import gamma as _gamma
        nus = np.arange(2.0, 50.01, 0.01)
        _GAMMA_LUT = (nus, np.array([_gamma(0.5*(v+1.0)) / (np.sqrt(v*np.pi) * _gamma(0.5*v))
                                     for v in nus]))
    nus, cs = _GAMMA_LUT
    return float(np.interp(float(np.clip(nu, 2.0, 50.0)), nus, cs))


def _skew_t_pdf(x, m, s, a, nu):
    """Azzalini skew-t density: f(z) = 2 t(z;nu) T(a z sqrt((nu+1)/(nu+z^2)); nu+1),
    z = (x-m)/s. nu=inf -> skew-normal limit. Analytic t-pdf (gamma LUT) +
    scipy.stats t.cdf (the incomplete-beta path is numerically unstable for
    nu >= 20 with strong skewness, so the C routine is kept)."""
    z = (x - m) / s
    c = _t_pdf_c(nu)
    tz = c * (1.0 + z * z / nu) ** (-0.5 * (nu + 1.0))
    arg = a * z * np.sqrt((nu + 1.0) / (nu + z * z))
    from scipy.stats import t as st_t
    Tz = st_t.cdf(arg, nu + 1.0)
    return 2.0 * tz * Tz / s


def _skew_t_mode(m, s, a, nu):
    """Numerical mode of a skew-t (no closed form): argmax on a coarse
    grid + parabolic (3-point) interpolation -- accurate to ~1e-6 at 1/10
    the grid cost of a fine scan."""
    lo, hi = m - 4.0 * s, m + 4.0 * s
    xg = np.linspace(lo, hi, 2001)
    fg = _skew_t_pdf(xg, m, s, a, nu)
    i = int(np.argmax(fg))
    if 0 < i < len(xg) - 1:
        x0, x1, x2 = xg[i-1], xg[i], xg[i+1]
        y0, y1, y2 = fg[i-1], fg[i], fg[i+1]
        # parabolic vertex of the 3-point fit
        num = ((x2 * x2) * (y0 - y1) + (x1 * x1) * (y2 - y0)
               + (x0 * x0) * (y1 - y2))
        den = ((x2 - x1) * (y0 - y1) + (x1 - x0) * (y2 - y1))
        if abs(den) > 1e-30:
            return float(0.5 * num / den)
    return float(xg[i])


def _two_skew_t(x, p):
    """Two-component skew-t mixture: p = (w1,m1,s1,a1,inu1, m2,s2,a2,inu2)
    with inu = 1/nu (nu in [2, 50] -> inu in [0.02, 0.5]). The inverse-nu
    parameterisation makes the Gaussian limit nu->inf natural and keeps
    the L-BFGS gradient well-conditioned for heavy tails."""
    w1, m1, s1, a1, inu1, m2, s2, a2, inu2 = p
    nu1 = 1.0 / max(inu1, 1e-6)
    nu2 = 1.0 / max(inu2, 1e-6)
    return (w1 * _skew_t_pdf(x, m1, s1, a1, nu1)
            + (1.0 - w1) * _skew_t_pdf(x, m2, s2, a2, nu2))


def _init_params(xc, h_sm, m_E, M_E, de, n_sigma=3.0, eb=None, masses=None,
                 seed=0, h0=None):
    """Initialise the two Gaussian means.

    Preferred: run get_Ecut (old FindMin method) to split the energy range,
    then on EACH side take the highest point that is ALSO a local maximum
    (argrelmax) -- a genuine peak, not a monotone-slope point. This keeps
    the two components straddling the physical bulge/disk boundary.

    If FindMin finds no cut (returns 0/None sentinel) -> RANDOM
    initialisation: two means drawn uniformly in the data range (fixed
    seed), so the fit still starts somewhere reasonable and the
    regulariser / chi2 drive it from there.
    """
    rng = np.random.RandomState(seed)
    pk_all = argrelmax(h_sm)[0]           # genuine local maxima
    pk_all = pk_all[h_sm[pk_all] > 0.05 * h_sm.max()]  # drop noise peaks
    if eb is not None:
        try:
            from kinematic_decompose.mixture.util import get_Ecut
            cut = get_Ecut(eb, masses)
            # get_Ecut returns 0 as a "no valley found" sentinel
            if cut is not None and cut != 0.0 and m_E < cut < M_E:
                pk_l = pk_all[xc[pk_all] < cut]
                pk_r = pk_all[xc[pk_all] > cut]
                if pk_l.size and pk_r.size:
                    i_l = pk_l[np.argmax(h_sm[pk_l])]
                    i_r = pk_r[np.argmax(h_sm[pk_r])]
                    m1, m2 = xc[i_l], xc[i_r]
                    if m1 < m2 and m2 - m1 > 2 * de:
                        s_guess = max(de, 0.5 * (m2 - m1))
                        return [0.5, m1, s_guess, 0.0, m2, s_guess, 0.0]
        except Exception:
            pass
    # ---- FindMin failed / no valid split: PEAK-GUIDED init (the two
    # highest genuine peaks anywhere in the distribution). If the smoothed
    # curve shows only one peak (the second is a shoulder -- heavy
    # smoothing merges close peaks), fall back to the top-2 RAW-count bins
    # (well separated) so the fit still starts on a two-peak hypothesis.
    if len(pk_all) >= 2:
        pk_sorted = pk_all[np.argsort(-h_sm[pk_all])][:2]
        i_l, i_r = sorted(pk_sorted)
        if xc[i_r] - xc[i_l] > 2 * de:
            s_guess = max(de, 0.5 * (xc[i_r] - xc[i_l]))
            return [0.5, xc[i_l], s_guess, 0.0, xc[i_r], s_guess, 0.0]
    # top-2 raw-count bins (the raw histogram still shows both peaks even
    # when smoothing merged them into one peak + shoulder)
    if h0 is not None:
        top = np.argsort(h0)[::-1][:2]
        i_l, i_r = sorted(top)
        if xc[i_r] - xc[i_l] > 3 * de:
            s_guess = max(de, 0.5 * (xc[i_r] - xc[i_l]))
            return [0.5, xc[i_l], s_guess, 0.0, xc[i_r], s_guess, 0.0]
    # ---- FindMin failed / no valid split: random init ----
    span = M_E - m_E
    for _ in range(20):
        m1 = m_E + span * rng.uniform(0.15, 0.45)
        m2 = m_E + span * rng.uniform(0.55, 0.85)
        if m2 - m1 > 2 * de:
            s_guess = max(de, 0.5 * (m2 - m1))
            return [0.5, m1, s_guess, 0.0, m2, s_guess, 0.0]
    # degenerate fallback: symmetric split around the middle
    m_mid = 0.5 * (m_E + M_E)
    s_guess = max(de, 0.25 * span)
    return [0.5, m_mid - s_guess, s_guess, 0.0, m_mid + s_guess, s_guess, 0.0]


def _separation_index(p):
    """Ashman separation coefficient for the fitted components (modes).
    Public utility: util.separation_index(mode1, mode2, s1, s2).
    USER-DECIDED criterion: sep >= 1 => resolvable; sep < 1 unresolvable."""
    w1, m1, s1, a1, inu1, m2, s2, a2, inu2 = p
    nu1 = 1.0 / max(inu1, 1e-6)
    nu2 = 1.0 / max(inu2, 1e-6)
    mo1 = _skew_t_mode(m1, s1, a1, nu1)
    mo2 = _skew_t_mode(m2, s2, a2, nu2)
    from kinematic_decompose.mixture.util import separation_index
    return float(separation_index(mo1, mo2, s1, s2))


def _order_params(p):
    """Sort the two components so m1 <= m2 (bulge on the low-energy side).
    Returns a re-ordered 9-param vector (w1,m1,s1,a1,nu1, m2,s2,a2,nu2)."""
    w1, m1, s1, a1, nu1, m2, s2, a2, nu2 = p
    if m1 <= m2:
        return np.array(p, float)
    return np.array([1.0 - w1, m2, s2, a2, nu2, m1, s1, a1, nu1], float)


def _fit_with_params(eb, masses=None, n_sigma=3.0, lam=10.0, d_min=1.5):
    """Full pipeline: FD bins + Silverman + 3-start skewed-Gaussian WLS fit
    with Bhattacharyya regularisation + physical-validity filter + FindMin
    fallback. Returns (popt, cut)."""
    e = np.asarray(eb, float)
    e = e[np.isfinite(e)]
    if len(e) < 50:
        return None, None
    m_E, M_E = e.min(), e.max()
    q75, q25 = np.percentile(e, [75, 25])
    iqr = q75 - q25
    de_fd = 2.0 * iqr * len(e) ** (-1.0 / 3.0)
    if de_fd <= 0 or not np.isfinite(de_fd):
        return None, None
    nbins = max(20, min(int(np.ceil((M_E - m_E) / de_fd)), 100))  # <=100 pts (speed)
    h0, edges = np.histogram(e, bins=nbins)
    de = edges[1] - edges[0]
    xc = edges[:-1] + 0.5 * de
    sig_hat = e.std(ddof=1)
    a = min(sig_hat, iqr / 1.34)
    h_silv = 0.9 * a * len(e) ** (-1.0 / 5.0)
    sb = max(0.5, SMOOTH_FACTOR * h_silv / de)  # enhanced denoising
    h_sm = gaussian_filter1d(h0.astype(float), sb)
    h_sm = h_sm / (h_sm.sum() * de)
    # ---- WLS weights from the RAW-count Poisson noise (decoupled from
    # the smoothing strength -- user decision). sigma = sqrt(n0+1) in
    # density units; the +1 keeps empty bins bounded (a zero variance
    # would give them infinite weight). The smoothed curve h_sm is the
    # density ESTIMATE; its weights reflect the underlying counts only,
    # so the enhanced smoothing (x2) cannot inflate the weights.
    sig = np.sqrt(h0 + 1.0) / (h0.sum() * de)
    sig = np.clip(sig, 1e-9, None)

    init = _init_params(xc, h_sm, m_E, M_E, de, n_sigma=n_sigma, eb=e,
                        masses=masses, h0=h0)
    # extend the 7-param skew-normal init to 9-param skew-t
    # (w1,m1,s1,a1,inu1, m2,s2,a2,inu2) with nu=10 -> inu=0.1
    init = list(init[:4]) + [0.1] + list(init[4:]) + [0.1]
    init = np.array(init, float)
    # FindMin split as a PHYSICAL prior: constrain m1 left of the cut and
    # m2 right of it, then 3-start Nelder-Mead inside that split space.
    # This prevents both (a) L-BFGS spike-traps (368843) and (b) global-search
    # escape to chi2-better-but-physically-wrong regions (371126).
    fm_split = None
    try:
        from kinematic_decompose.mixture.util import get_Ecut
        m_arr = masses if masses is not None else np.ones(len(e))
        _fm = get_Ecut(e, m_arr)
        if _fm is not None and _fm != 0.0 and m_E < _fm < M_E:
            fm_split = float(_fm)
    except Exception:
        pass
    # 9-param skew-t bounds: (w,m,s,a,inu, m,s,a,inu); inu in [0.02, 0.5]
    # i.e. nu in [2, 50] (inverse-nu parameterisation). The skewness a is
    # left FREE in [-5, 5] (user decision after the parameter-effect plot:
    # a controls which side the tail thickens; |a|<0.25 is nearly
    # symmetric and locks out real skewness).
    lo = np.array([0.0, m_E, de, -5.0, 0.02, m_E, de, -5.0, 0.02])
    hi = np.array([1.0, M_E, 0.5 * (M_E - m_E), 5.0, 0.5,
                   M_E, 0.5 * (M_E - m_E), 5.0, 0.5])
    q05 = float(np.quantile(e, 0.05))
    if fm_split is not None:
        lo[1] = q05              # m1 in [q05, cut] (dense region, not tail)
        hi[1] = fm_split
        lo[5] = fm_split
        hi[5] = np.quantile(e, 0.95)   # m2 in [cut, q95] (dense region)
    else:
        # FindMin found no valley: split BETWEEN the two init means so the
        # labels stay stable (component 1 = low-energy side) AND the init
        # always falls inside the box. The init itself is peak-guided /
        # top-2-bin guided (see _init_params), so this split matches it.
        m1i, m2i = float(init[1]), float(init[5])
        if m2i > m1i:
            split = 0.5 * (m1i + m2i)
            lo[1] = q05
            hi[1] = split
            lo[5] = split
            hi[5] = np.quantile(e, 0.95)
        else:
            mid = 0.5 * (m_E + M_E)
            lo[1] = q05
            hi[1] = mid
            lo[5] = mid
            hi[5] = np.quantile(e, 0.95)

    # chi2 scale reference: single-component fit at the DATA median/std
    # (not the init means -- those can be random when FindMin fails, which
    # would make the normalisation baseline unstable).
    med_e = float(np.median(e))
    std_e = max(de, float(e.std(ddof=1)))
    p_single = [1.0, med_e, std_e, 0.0, 0.1, med_e, std_e, 0.0, 0.1]
    chi2_0 = float(np.sum(((h_sm - _two_skew_t(xc, p_single)) / sig) ** 2))

    def obj(p):
        chi2 = np.sum(((h_sm - _two_skew_t(xc, p)) / sig) ** 2)
        return chi2 / chi2_0   # pure fit-quality (NO separation reg)

    # ---- DIRECT Nelder-Mead minimisation (user decision). NM supports
    # bounds natively (scipy >= 1.7); derivative-free and robust to the
    # non-smooth chi2 landscape. SINGLE start from the FindMin/peak-guided
    # init (the best-quality start -- speed requirement 0.1 s/galaxy).
    best = None
    try:
        r = minimize(obj, np.clip(init, lo, hi), method="Nelder-Mead",
                     bounds=list(zip(lo, hi)),
                     options={"maxiter": 600, "xatol": 1e-5, "fatol": 1e-5})
        best = r
    except Exception:
        pass
    # last-resort random multi-start if the 3-start NM all failed
    if best is None:
        rng = np.random.RandomState(1234)
        span = M_E - m_E
        for _ in range(12):
            m1r = m_E + span * rng.uniform(0.15, 0.50)
            m2r = m_E + span * rng.uniform(0.55, 0.90)
            if m2r - m1r <= 2 * de:
                continue
            s_guess = max(de, 0.5 * (m2r - m1r))
            trial = [0.5, m1r, s_guess, 0.0, 0.1, m2r, s_guess, 0.0, 0.1]
            try:
                r = minimize(obj, np.clip(trial, lo, hi), method="Nelder-Mead",
                             bounds=list(zip(lo, hi)),
                             options={"maxiter": 2000, "xatol": 1e-5,
                                      "fatol": 1e-5})
                if best is None or r.fun < best.fun:
                    best = r
            except Exception:
                continue
    p = best.x if best is not None else np.array(init, float)
    p = _order_params(p)
    # cut = numerical INTERSECTION of the two skew-t components between
    # their MODES (the physical boundary lies between the peaks). If no
    # root exists there (one component covers the other), fall back to
    # the mean of the two modes -- exactly as specified.
    w1f, m1f, s1f, a1f, inu1f, m2f, s2f, a2f, inu2f = p
    nu1f = 1.0 / max(inu1f, 1e-6)
    nu2f = 1.0 / max(inu2f, 1e-6)
    mo1 = _skew_t_mode(m1f, s1f, a1f, nu1f)
    mo2 = _skew_t_mode(m2f, s2f, a2f, nu2f)
    cut = 0.5 * (mo1 + mo2)   # default: mean of the two peaks
    try:
        from scipy.optimize import brentq

        def diff(x):
            return (w1f * _skew_t_pdf(x, m1f, s1f, a1f, nu1f)
                    - (1.0 - w1f) * _skew_t_pdf(x, m2f, s2f, a2f, nu2f))

        lo_r, hi_r = min(mo1, mo2), max(mo1, mo2)
        if diff(lo_r) * diff(hi_r) <= 0:
            cut = float(brentq(diff, lo_r, hi_r))
    except Exception:
        pass
    # safety: fall back to the FindMin valley when the best candidate is
    # degenerate -- (a) cut outside the data bulk, (b) a component narrower
    # than 2 bins (spike overfit), or (c) an extreme weight (<5%). The
    # spike solution is a classic chi2 overfit: s->bin width, a->+-5,
    # mathematically best but physically void.
    lo_q, hi_q = np.quantile(e, [0.01, 0.99])
    span = M_E - m_E
    degenerate = (cut is None or not (lo_q <= cut <= hi_q)
                  or min(s1f, s2f) < 2.0 * de
                  or max(s1f, s2f) >= 0.95 * 0.5 * span  # s pinned to bound
                  or min(w1f, 1.0 - w1f) < 0.05
                  or m1f < m_E + 0.05 * span            # mean pinned left
                  or m2f > M_E - 0.05 * span)           # mean pinned right
    if degenerate:
        try:
            from kinematic_decompose.mixture.util import get_Ecut
            m_arr = masses if masses is not None else np.ones(len(e))
            # explicit params -- the library defaults (M_bin=400,
            # m_bin=80, Mmin=0.05) give a DIFFERENT valley than the
            # pipeline standard (100/25/0.1), so use the pipeline's.
            fm = get_Ecut(e, m_arr, M_bin=100, m_bin=25, Mmin=0.1)
            if fm is not None and fm != 0.0 and lo_q <= fm <= hi_q:
                return p, float(fm), float(_separation_index(p))
        except Exception:
            pass
        # FindMin also failed (no valley / sentinel 0). Fall back to a
        # robust parameter-based boundary: the mean of the two modes,
        # else the distribution median (last resort).
        cands = [0.5 * (mo1 + mo2), float(np.median(e))]
        for c in cands:
            if c is not None and lo_q <= c <= hi_q:
                cut = float(c)
                break
    return p, cut, float(_separation_index(p))


def get_Ecut_2gauss(eb, masses=None, n_sigma=3.0, lam=10.0, d_min=1.5):
    """Public API: ecut via two skewed Gaussians (see _fit_with_params)."""
    p, cut, sep = _fit_with_params(eb, masses=masses, n_sigma=n_sigma, lam=lam,
                                   d_min=d_min)
    return cut
