import scipy
import numpy as np
from scipy.signal import find_peaks, argrelmax
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

MAX_RADIUS = 10


def separation_index(mode1, mode2, sigma1, sigma2):
    """Ashman separation coefficient (Ashman, Bird & Zepf 1994, AJ 108:2348):
    sep = |mode2 - mode1| / sqrt(sigma1^2 + sigma2^2).

    Criterion (user-decided): sep >= 1 => the two peaks are separated by
    more than one combined sigma -> a RESOLVABLE bimodal signal; sep < 1
    unresolvable (unimodal / one component covers the other). Use as a
    reporting/screening metric for two-component decompositions."""
    return abs(mode2 - mode1) / np.sqrt(sigma1 * sigma1 + sigma2 * sigma2)


def hist_bin_fd(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)

def create_eoemin_pdf(e):
    wid = hist_bin_fd(e)
    bins= min(int(np.ptp(e) / wid), 100)
    hist, bin_edges = np.histogram(e, bins=bins, range=[-1, 0])
    bin_widths = np.diff(bin_edges)
    pdf_values = hist / (len(e) * bin_widths)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    pdf = interp1d(bin_centers, pdf_values, 
                       kind='linear', 
                       bounds_error=False, 
                       fill_value=0)
    return pdf

def create_eoemin_profile_function(e, r, r_min, r_max=MAX_RADIUS, bins=None, range=[-1.,0.], statistic='median'):
    if bins is None:
        wid = hist_bin_fd(e)
        bins= min(int(np.ptp(e) / wid), 50)
    r_median, bin_edges, _ = scipy.stats.binned_statistic(
        e, r, statistic=statistic, bins=bins, range=range
    )
    ebin = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid_mask = np.isfinite(r_median)
    r_median, ebin = r_median[valid_mask], ebin[valid_mask]

    mask = (r_median < r_max) & (r_median > r_min)
    x, y = ebin[mask], r_median[mask]

    if len(x) < 4:
        mask = (r_median > r_min)
        x, y = ebin[mask], r_median[mask]

    y = np.maximum.accumulate(y)
    y, idx = np.unique(y, return_index=True)
    x = x[idx]
    f_r = scipy.interpolate.PchipInterpolator(y, x, extrapolate=False)
    return f_r

def create_potential_profile(pot, points, normalize=True, return_order=3):
    r = np.linalg.norm(points, axis=1)
    potential = pot.potential(points)
    if normalize:
        potential/=potential.min()
    dpot_dr = np.gradient(potential, r, edge_order=2)
    d2pot_dr2 = np.gradient(dpot_dr, r, edge_order=2)
    curvature = np.abs(d2pot_dr2) / (1 + dpot_dr**2)**1.5
    return (r, potential) if return_order == 0 else \
           (r, potential, dpot_dr) if return_order == 1 else \
           (r, potential, dpot_dr, d2pot_dr2) if return_order == 2 else \
           (r, potential, dpot_dr, d2pot_dr2, curvature) if return_order == 3 else \
           (r, potential, dpot_dr, d2pot_dr2, curvature)

def get_energy_criterion(pot, particle_radius, e, 
                         r_min=1, r_max=MAX_RADIUS, 
                         step=0.1, cut_ratio='auto', 
                         remove_boundary=True, 
                         remove_negative=True,
                         eoemin_min_cut=-0.9):
    x = y = np.arange(r_min, r_max, step)
    points = np.column_stack([x,y,np.zeros(len(x))])
    r, _, _, d2pot_dr2, curvature = create_potential_profile(pot, points, normalize=True, return_order=3)
    f_r = create_eoemin_profile_function(e, particle_radius, r_min, r_max, bins=50, range=[-1.,0.])
    
    if remove_boundary:
        d2pot_dr2 = d2pot_dr2[1:-1]
        curvature = curvature[1:-1]
        r = r[1:-1]
    if remove_negative:
        positive_indices = np.where(curvature > 0)[0]
        d2pot_dr2 = d2pot_dr2[positive_indices]
        curvature = curvature[positive_indices]
        r = r[positive_indices]
        
    max_idx = np.argmax(d2pot_dr2)
    post_curve = curvature[max_idx:]
    eoemin_cut = None
    if cut_ratio == 'auto':
        pdf = create_eoemin_pdf(e)
        cut_ratios = np.arange(0.05, 1.05, 0.01)
        cut_particle_nums = []
        eoemin_cuts = []
        for ratio in cut_ratios:
            cut_positions = np.where(post_curve <= curvature[max_idx] * ratio)[0]
            cut_idx = max_idx + cut_positions[0] if cut_positions.size > 0 else max_idx
            radius_cut = r[cut_idx]
            eoemin_cut_candidate = f_r(radius_cut)
            eoemin_cuts.append(eoemin_cut_candidate)
            cut_particle_nums.append(pdf(eoemin_cut_candidate))
        smoothed = gaussian_filter1d(cut_particle_nums, sigma=5)
        valleys, properties = find_peaks(-smoothed, prominence=0.01)
        if len(valleys) > 0:
            deepest_valley_idx = valleys[np.argmax(properties['prominences'])]
            eoemin_cut = eoemin_cuts[deepest_valley_idx]
        else:
            median_idx = len(cut_particle_nums) // 2
            eoemin_cut = eoemin_cuts[median_idx]
    elif isinstance(cut_ratio, float):
        cut_positions = np.where(post_curve <= curvature[max_idx] * cut_ratio)[0]
        cut_idx = max_idx + cut_positions[0] if cut_positions.size > 0 else max_idx
        radius_cut = r[cut_idx]
        eoemin_cut = f_r(radius_cut)
    return eoemin_cut

def get_Ecut(eb, masses, nbins = 25,M_bin=400,m_bin=80,toll=1.5,shrink=2,Mmin = 0.05,Emin = -0.9):
    
    if len(eb)<100:
        print('Ecut = ', 0)
        return 0

    #Fix the number of bin as a function of Npart
    NbinMax = max(min(int(0.5*np.sqrt(len(eb))), M_bin), m_bin)


    #This is to exclude the outer tail of bound particles
    M_E = np.quantile(eb, 0.9)
    #m_E = np.min(eb)
    m_E = np.quantile(eb, 0.01)
    Ecut, E_val = FindMin(eb, m_E, M_E, nbins)


    #If no minimum is found or the only minimum is too close to -1 (Maybe a GC?)
    if len(Ecut)==0 or (len(Ecut)==1 and Ecut<Emin):


        M_E = np.max(eb)
        Ecut, E_val = FindMin(eb, m_E, M_E, nbins)
        Ecut = Ecut
    #If one or none minima are found
    if len(Ecut)<=1:
        D = (M_E-m_E)/float(nbins)
        #Avoid the following loop
        nbins = NbinMax+1
    else:
        D = (M_E-m_E)/float(nbins)
        lb = Ecut-(toll*D)
        rb = Ecut+(toll*D)

    #-------

    while nbins < NbinMax:
        nbins = shrink*nbins
        D = D/shrink
        pos_E_refined, val_refined = FindMin(eb, m_E, M_E, nbins)
        EcutTEMP = []
        E_valTEMP = []
        for i, v in enumerate(E_val):
            pTEMP = pos_E_refined[(pos_E_refined<=rb[i])*(pos_E_refined>=lb[i])]
            vTEMP = val_refined[(pos_E_refined<=rb[i])*(pos_E_refined>=lb[i])]
            if len(pTEMP)>0:
                #A rifened position and value for each original minimum is stored. 
                #The value of the minima is summed to the original ones to avoid strange local minima 
                EcutTEMP.append(pTEMP[np.argmin(vTEMP)])
                E_valTEMP.append(v+np.min(vTEMP))

        Ecut = np.array(EcutTEMP)
        E_val = np.array(E_valTEMP)



        if len(Ecut)<=1:
            break

        lb = Ecut-(toll*D)
        rb = Ecut+(toll*D)
    
    #If no energy cut is found		
    if len(Ecut)==0:
        Ecut = 0
    else:
        #Try to avoid strange nuclear minima with low mass if there are better alternatives
        total_mass = np.sum(masses)
        rel_filt = [bool((np.sum(masses[eb<E])/total_mass>=Mmin)+(E>=Emin)) for E in Ecut]
        if len(Ecut[rel_filt])==0:
            Ecut = Ecut[np.argmin(E_val)]
        else:
            Ecut = Ecut[rel_filt][np.argmin(E_val[rel_filt])]
        Ecut = RefineMin(eb, Ecut, D, (M_E-m_E)/NbinMax, shrink)
    
    return Ecut


# ---------------------------------------------------------------------------
# Two-component skew-t ecut
# ---------------------------------------------------------------------------
# This method belongs beside get_Ecut because it is an alternative ecut
# estimator, not a separate mixture-model public API.
SMOOTH_FACTOR = 2.0
_GAMMA_LUT = None


def _t_pdf_c(nu):
    """Return the Student-t normalisation constant for ``nu``."""
    global _GAMMA_LUT
    if _GAMMA_LUT is None:
        from scipy.special import gamma

        nus = np.arange(2.0, 50.01, 0.01)
        _GAMMA_LUT = (
            nus,
            np.array([
                gamma(0.5 * (value + 1.0))
                / (np.sqrt(value * np.pi) * gamma(0.5 * value))
                for value in nus
            ]),
        )
    nus, constants = _GAMMA_LUT
    return float(np.interp(float(np.clip(nu, 2.0, 50.0)), nus, constants))


def _skew_t_pdf(x, m, s, a, nu):
    """Evaluate the Azzalini skew-t density."""
    z = (x - m) / s
    c = _t_pdf_c(nu)
    t_pdf = c * (1.0 + z * z / nu) ** (-0.5 * (nu + 1.0))
    argument = a * z * np.sqrt((nu + 1.0) / (nu + z * z))
    from scipy.stats import t as student_t

    return 2.0 * t_pdf * student_t.cdf(argument, nu + 1.0) / s


def _skew_t_mode(m, s, a, nu):
    """Numerically estimate the mode of one skew-t component."""
    lower, upper = m - 4.0 * s, m + 4.0 * s
    grid = np.linspace(lower, upper, 2001)
    density = _skew_t_pdf(grid, m, s, a, nu)
    index = int(np.argmax(density))
    if 0 < index < len(grid) - 1:
        x0, x1, x2 = grid[index - 1:index + 2]
        y0, y1, y2 = density[index - 1:index + 2]
        numerator = (
            x2 * x2 * (y0 - y1)
            + x1 * x1 * (y2 - y0)
            + x0 * x0 * (y1 - y2)
        )
        denominator = (
            (x2 - x1) * (y0 - y1)
            + (x1 - x0) * (y2 - y1)
        )
        if abs(denominator) > 1e-30:
            return float(0.5 * numerator / denominator)
    return float(grid[index])


def _two_skew_t(x, params):
    """Evaluate a two-component skew-t mixture."""
    w1, m1, s1, a1, inv_nu1, m2, s2, a2, inv_nu2 = params
    nu1 = 1.0 / max(inv_nu1, 1e-6)
    nu2 = 1.0 / max(inv_nu2, 1e-6)
    return (
        w1 * _skew_t_pdf(x, m1, s1, a1, nu1)
        + (1.0 - w1) * _skew_t_pdf(x, m2, s2, a2, nu2)
    )


def _init_skew_t_params(
    xc, h_sm, m_E, M_E, de, n_sigma=3.0, eb=None, masses=None,
    seed=0, h0=None,
):
    """Initialise the two skew-t components around an ecut valley."""
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
            return [0.5, xc[i_left], s_guess, 0.0,
                    xc[i_right], s_guess, 0.0]
    if h0 is not None:
        top = np.argsort(h0)[::-1][:2]
        i_left, i_right = sorted(top)
        if xc[i_right] - xc[i_left] > 3 * de:
            s_guess = max(de, 0.5 * (xc[i_right] - xc[i_left]))
            return [0.5, xc[i_left], s_guess, 0.0,
                    xc[i_right], s_guess, 0.0]
    span = M_E - m_E
    for _ in range(20):
        m1 = m_E + span * rng.uniform(0.15, 0.45)
        m2 = m_E + span * rng.uniform(0.55, 0.85)
        if m2 - m1 > 2 * de:
            s_guess = max(de, 0.5 * (m2 - m1))
            return [0.5, m1, s_guess, 0.0, m2, s_guess, 0.0]
    m_mid = 0.5 * (m_E + M_E)
    s_guess = max(de, 0.25 * span)
    return [0.5, m_mid - s_guess, s_guess, 0.0,
            m_mid + s_guess, s_guess, 0.0]


def _skew_t_separation_index(params):
    """Return the Ashman separation index of fitted skew-t modes."""
    _, m1, s1, a1, inv_nu1, m2, s2, a2, inv_nu2 = params
    nu1 = 1.0 / max(inv_nu1, 1e-6)
    nu2 = 1.0 / max(inv_nu2, 1e-6)
    mode1 = _skew_t_mode(m1, s1, a1, nu1)
    mode2 = _skew_t_mode(m2, s2, a2, nu2)
    return float(separation_index(mode1, mode2, s1, s2))


def _order_skew_t_params(params):
    """Order skew-t components by increasing location."""
    w1, m1, s1, a1, inv_nu1, m2, s2, a2, inv_nu2 = params
    if m1 <= m2:
        return np.array(params, float)
    return np.array([
        1.0 - w1, m2, s2, a2, inv_nu2,
        m1, s1, a1, inv_nu1,
    ], float)


def _fit_skew_t_with_params(
    eb, masses=None, n_sigma=3.0, lam=10.0, d_min=1.5,
):
    """Fit two skew-t components and return ``(params, cut, separation)``."""
    e = np.asarray(eb, float)
    finite = np.isfinite(e)
    e = e[finite]
    if len(e) < 50:
        return None, None, None
    m_E, M_E = e.min(), e.max()
    q75, q25 = np.percentile(e, [75, 25])
    iqr = q75 - q25
    de_fd = 2.0 * iqr * len(e) ** (-1.0 / 3.0)
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

    init = _init_skew_t_params(
        xc, h_sm, m_E, M_E, de, n_sigma=n_sigma, eb=e,
        masses=masses, h0=h0,
    )
    init = np.array(list(init[:4]) + [0.1] + list(init[4:]) + [0.1], float)

    fm_split = None
    try:
        m_arr = masses if masses is not None else np.ones(len(e))
        fitted_cut = get_Ecut(e, m_arr)
        if fitted_cut is not None and fitted_cut != 0.0 and m_E < fitted_cut < M_E:
            fm_split = float(fitted_cut)
    except Exception:
        pass

    lower = np.array([0.0, m_E, de, -5.0, 0.02,
                      m_E, de, -5.0, 0.02])
    upper = np.array([1.0, M_E, 0.5 * (M_E - m_E), 5.0, 0.5,
                      M_E, 0.5 * (M_E - m_E), 5.0, 0.5])
    q05 = float(np.quantile(e, 0.05))
    if fm_split is not None:
        lower[1], upper[1] = q05, fm_split
        lower[5], upper[5] = fm_split, np.quantile(e, 0.95)
    else:
        m1_init, m2_init = float(init[1]), float(init[5])
        split = 0.5 * (m1_init + m2_init) if m2_init > m1_init else 0.5 * (m_E + M_E)
        lower[1], upper[1] = q05, split
        lower[5], upper[5] = split, np.quantile(e, 0.95)

    med_e = float(np.median(e))
    std_e = max(de, float(e.std(ddof=1)))
    p_single = [1.0, med_e, std_e, 0.0, 0.1,
                med_e, std_e, 0.0, 0.1]
    chi2_0 = float(np.sum(((h_sm - _two_skew_t(xc, p_single)) / sig) ** 2))

    def objective(params):
        chi2 = np.sum(((h_sm - _two_skew_t(xc, params)) / sig) ** 2)
        return chi2 / chi2_0

    from scipy.optimize import minimize

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
    try:
        from scipy.optimize import brentq

        def difference(x):
            return (
                w1 * _skew_t_pdf(x, m1, s1, a1, nu1)
                - (1.0 - w1) * _skew_t_pdf(x, m2, s2, a2, nu2)
            )

        lower_root, upper_root = min(mode1, mode2), max(mode1, mode2)
        if difference(lower_root) * difference(upper_root) <= 0:
            cut = float(brentq(difference, lower_root, upper_root))
    except Exception:
        pass

    lo_q, hi_q = np.quantile(e, [0.01, 0.99])
    span = M_E - m_E
    degenerate = (
        cut is None or not (lo_q <= cut <= hi_q)
        or min(s1, s2) < 2.0 * de
        or max(s1, s2) >= 0.95 * 0.5 * span
        or min(w1, 1.0 - w1) < 0.05
        or m1 < m_E + 0.05 * span
        or m2 > M_E - 0.05 * span
    )
    if degenerate:
        try:
            m_arr = masses if masses is not None else np.ones(len(e))
            fallback = get_Ecut(e, m_arr, M_bin=100, m_bin=25, Mmin=0.1)
            if fallback is not None and fallback != 0.0 and lo_q <= fallback <= hi_q:
                return params, float(fallback), _skew_t_separation_index(params)
        except Exception:
            pass
        for candidate in (0.5 * (mode1 + mode2), float(np.median(e))):
            if lo_q <= candidate <= hi_q:
                cut = float(candidate)
                break
    return params, cut, _skew_t_separation_index(params)


def get_Ecut_skewt(eb, masses=None, n_sigma=3.0, lam=10.0, d_min=1.5):
    """Return an ecut estimated from a two-component skew-t fit."""
    _, cut, _ = _fit_skew_t_with_params(
        eb, masses=masses, n_sigma=n_sigma, lam=lam, d_min=d_min,
    )
    return cut


def FindMin(q, m_E, M_E, nbins):
    #Minimum number of particles to perform a reliable Jcirc decomposition
    if len(q)>=1e4:
        Npart_min=1000
    elif 1e3<=len(q)<1e4:
        Npart_min=100
    else:
        Npart_min=10
    
    MinPart = max(Npart_min, 0.01*len(q))
    arr = q[(q>=m_E)*(q<=M_E)]
    #Build the histogram
    hist = np.histogram(arr, bins=np.linspace(m_E, M_E, nbins))

    #Evaluate the increment on both sides A
    diff = hist[0][1:]-hist[0][:-1]
    left = diff[:-1]
    right = diff[1:]
    #Find the minima
    id_E = np.where(((left<0)*(right>=0))+((left<=0)*(right>0)))

    #C — right-of-valley particle count, vectorized via reverse cumsum
    rev_cumsum = np.flip(np.cumsum(np.flip(hist[0])))
    id_E = id_E[0][rev_cumsum[id_E[0] + 1] > MinPart]
    id_E_flag = [True]*len(id_E)
    
    
    #B
    for i, ids in enumerate(id_E):
        if len(hist[0])>ids+3:
            id_E_flag[i] *= hist[0][ids+3]>hist[0][ids+1]
        if ids > 0:
            id_E_flag[i] *= hist[0][ids-1]>hist[0][ids+1]

    id_E = id_E[id_E_flag]
    #Return the central position of the bins
    return 0.5*(hist[1][id_E+2]+hist[1][id_E+1]), hist[0][id_E+1]


def RefineMin(q, Vmin, D, Dmin, shrink):    
    arr=[]
    if D<=Dmin:
        if len(q)>=1e4:
            coe=0.5
        elif 1e3<=len(q)<1e4:
            coe=2
        else:
            coe=2.5
        while len(arr)==0:
            m_E = Vmin-coe*D 
            M_E = Vmin+coe*D
            arr = q[(q>=m_E)*(q<=M_E)]
            coe=coe+0.2
        Vmin = np.median(arr)

    while D > Dmin:
        if len(q)>=1e3:
            coe=1.5
        else:
            coe=4
        m_E = max(Vmin-coe*D, q.min()+D) 
        M_E = Vmin+coe*D
        D = D/shrink
        arr = q[(q>=m_E)*(q<=M_E)]
        hist = np.histogram(arr, bins=np.arange(m_E, M_E, D))
        nonzero = np.where(hist[0] != 0)[0]
        if nonzero.size == 0:
            break  # empty histogram: nothing to refine
        hist_min = hist[0][nonzero].min()
        pid = np.where(hist[0] == hist_min)[0][0]
        arr=arr[(arr>=hist[1][pid])*(arr<=(hist[1][pid+1]))]
        #Get the energy as the median within the selected bin
        Vmin = np.median(arr)

    return Vmin

def JEHistogram(E, eps, n_E=20, n_eps=30, seed=42):
    """
    Abadi-like decomposition with local eps symmetry.

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

    # Energy bins
    E_edges = np.linspace(E.min(), E.max(), n_E + 1)

    for i in range(n_E):
        e_mask = (E >= E_edges[i]) & (E < E_edges[i + 1])
        idx_E = np.flatnonzero(e_mask)
        if idx_E.size == 0:
            continue

        eps_E = eps[idx_E]

        # Symmetric eps bins
        eps_max = np.max(np.abs(eps_E))
        if eps_max == 0:
            sph[idx_E] = True
            continue

        eps_edges = np.linspace(-eps_max, eps_max, n_eps + 1)
        bin_id = np.searchsorted(eps_edges, eps_E, side="right") - 1
        bin_id = np.clip(bin_id, 0, n_eps - 1)

        mid = n_eps // 2

        # Loop over symmetric eps bins
        for k in range(mid):
            neg_bin = mid - 1 - k
            pos_bin = mid + k

            idx_neg = idx_E[bin_id == neg_bin]
            idx_pos = idx_E[bin_id == pos_bin]

            if idx_neg.size == 0:
                continue

            # All counter-rotating are spheroid
            sph[idx_neg] = True

            if idx_pos.size == 0:
                continue

            n_sel = min(idx_neg.size, idx_pos.size)
            chosen = (
                idx_pos if idx_pos.size == n_sel
                else rng.choice(idx_pos, n_sel, replace=False)
            )
            sph[chosen] = True

        # Optional: eps ~ 0 bin → spheroid
        sph[idx_E[bin_id == mid]] = True

    disk = ~sph
    return sph, disk

def decompose(X, galaxy, model, eoemin_cut, jzojc_cut, predict_method='soft', require_bulge_halo=False):
    dim = model.means_.shape[1]
    Xd = X[:, :dim]
    # Single full-data E-step shared by BOTH the labels and the
    # probabilities (soft_predict + predict_proba used to each run one).
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
    
    weights, means, covariances = (
        model.weights_,
        model.means_,
        model.covariances_
    )
    
    label_map = {}
    for i, (ec, eta) in enumerate(means[:,0:2]):
        if eta >= 0.85:
            label_map[i] = 0  # cold disk
        elif eta > jzojc_cut:
            label_map[i] = 1  # warm disk
        elif eta < -jzojc_cut:
            label_map[i] = 4  # counter-rotating disk
        elif eta <= jzojc_cut and ec < eoemin_cut:
            label_map[i] = 2  # bulge
        else:
            label_map[i] = 3  # halo
    
    new_labels = np.vectorize(label_map.get)(labels)
    new_prob = np.zeros((len(prob), 5), dtype=np.float32)
    for old_idx, new_idx in label_map.items():
        new_prob[:, new_idx] += prob[:, old_idx]
    
    # Store results in galaxy object
    galaxy.s['label'] = new_labels
    galaxy.s['prob'] = new_prob
    del new_labels, labels, new_prob, prob
    bulge_count = np.sum(galaxy.s['label'] == 2)
    halo_count = np.sum(galaxy.s['label'] == 3)
    if require_bulge_halo:
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
            sph, disk = JEHistogram(galaxy.s['eoemin'], galaxy.s['jzojc'])
            bulge_candidates = galaxy.s['eoemin'] < eoemin_cut
            halo_candidates = galaxy.s['eoemin'] > eoemin_cut
            to_bulge_mask = sph & bulge_candidates
            to_halo_mask  = sph & halo_candidates
            galaxy.s['label'][to_bulge_mask] = 2
            galaxy.s['label'][to_halo_mask] = 3
        for i in range(len(galaxy.s['label'])):
            old_label = galaxy.s['label'][i]
            galaxy.s['prob'][i] = np.zeros(5, dtype=np.float32)
            galaxy.s['prob'][i][old_label] = 1.0
    return galaxy

def decompose_mixture_model(model, eoemin_cut, jzojc_cut, r_jzojc_cut):
    COLD_DISK_THRESHOLD = 0.85
    
    weights, means, covariances = (
            model.weights_,
            model.means_,
            model.covariances_
        )

    eta = means[:, 1]
    e   = means[:, 0]

    disk_mask = eta > jzojc_cut
    cold_disk_mask = eta >= COLD_DISK_THRESHOLD
    warm_disk_mask = disk_mask & (eta < COLD_DISK_THRESHOLD)
    spheroid_mask = (r_jzojc_cut<=eta)&(eta<= jzojc_cut)
    bulge_mask = spheroid_mask & (e < eoemin_cut)
    halo_mask = spheroid_mask & (e >= eoemin_cut)
    counter_rotating_disk_mask = (eta<=r_jzojc_cut) 

    GMM_dict = {
        "total": {
            "weights": weights,
            "means": means,
            "covariances": covariances
        },
        "disk": {
            "weights": weights[disk_mask],
            "means": means[disk_mask],
            "covariances": covariances[disk_mask]
        },
        "colddisk": {
            "weights": weights[cold_disk_mask],
            "means": means[cold_disk_mask],
            "covariances": covariances[cold_disk_mask]
        },
        "warmdisk": {
            "weights": weights[warm_disk_mask],
            "means": means[warm_disk_mask],
            "covariances": covariances[warm_disk_mask]
        },
        "spheroid": {
            "weights": weights[spheroid_mask],
            "means": means[spheroid_mask],
            "covariances": covariances[spheroid_mask]
        },
        "bulge": {
            "weights": weights[bulge_mask],
            "means": means[bulge_mask],
            "covariances": covariances[bulge_mask]
        },
        "halo": {
            "weights": weights[halo_mask],
            "means": means[halo_mask],
            "covariances": covariances[halo_mask]
        },
        "counter-rotating disk": {
            "weights": weights[counter_rotating_disk_mask],
            "means": means[counter_rotating_disk_mask],
            "covariances": covariances[counter_rotating_disk_mask]
        },
        "eoemin_cut": eoemin_cut,
        "jzojc_cut": jzojc_cut,
        "r_jzojc_cut": r_jzojc_cut,
    }
    return GMM_dict

# TODO: save_structure_properties
def save_structure_properties(sim):
    dict = {"total":{}, "dm":{}, "star":{}, 
            "disk":{}, "colddisk":{}, "warmdisk":{}, "spheroid":{}, "bulge":{}, "halo":{},
            "gas":{}, "coldgas":{}}
    """
    total:
    M_vir, R_vir, V_vir, T_vir, 
    spin, krot, beta, AM, 
    vel_disp, vr_disp, vR_disp, vz_disp,
    v_circ, v_rot, ke,
    Mcold, Mbary,
    r50, R50, z50, 
    shape
    """
    dict['total']['M_vir']   = sim.M_vir
    dict['total']['R_vir']   = sim.R_vir
    dict['total']['V_vir']   = sim.V_vir
    dict['total']['T_vir']   = sim.T_vir
    dict['total']['spin']    = sim.spin
    #dict['total']['krot']    = sim.krot
    #dict['total']['beta']    = sim.beta
    dict['total']['AM']      = sim.AM
    #dict['total']['vel_disp']= sim.vel_disp
    #dict['total']['vr_disp'] = sim.vr_disp
    #dict['total']['vR_disp'] = sim.vR_disp
    #dict['total']['vz_disp'] = sim.vz_disp
    #dict['total']['v_circ']  = sim.v_circ
    #dict['total']['v_rot']   = sim.v_rot
    #dict['total']['ke']      = sim.ke
    #dict['total']['Mcold']   = sim.Mcold
    #dict['total']['Mbary']   = sim.Mbary
    #dict['total']['r50']     = sim.r50
    #dict['total']['R50']     = sim.R50
    #dict['total']['z50']     = sim.z50
    #dict['total']['shape']   = sim.shape
    """
    dm:
    M_vir, R_vir, V_vir, T_vir, 
    spin, krot, beta, AM, 
    vel_disp, vr_disp, vR_disp, vz_disp,
    v_circ, v_rot, ke,
    Mdyn, Mcold, Mbary,
    r50, R50, z50, 
    shape
    """
    dict['dm']['M_vir']   = sim.dm.M_vir
    dict['dm']['R_vir']   = sim.dm.R_vir
    dict['dm']['V_vir']   = sim.dm.V_vir
    dict['dm']['T_vir']   = sim.dm.T_vir
    dict['dm']['spin']    = sim.dm.spin
    #dict['dm']['krot']    = sim.dm.krot
    #dict['dm']['beta']    = sim.dm.beta
    dict['dm']['AM']      = sim.dm.AM
    dict['dm']['vel_disp']= sim.dm.vel_disp
    #dict['dm']['vr_disp'] = sim.dm.vr_disp
    #dict['dm']['vR_disp'] = sim.dm.vR_disp
    #dict['dm']['vz_disp'] = sim.dm.vz_disp
    #dict['dm']['v_circ']  = sim.dm.v_circ
    #dict['dm']['v_rot']   = sim.dm.v_rot
    dict['dm']['ke']      = sim.dm.ke
    #dict['dm']['Mcold']   = sim.dm.Mcold
    #dict['dm']['Mbary']   = sim.dm.Mbary
    #dict['dm']['r50']     = sim.dm.r50
    #dict['dm']['R50']     = sim.dm.R50
    #dict['dm']['z50']     = sim.dm.z50
    #dict['dm']['shape']   = sim.dm.shape

    """
    star:
    spin, krot, beta, AM, 
    vel_disp, vr_disp, vR_disp, vz_disp,
    v_circ, v_rot, ke,
    Mdyn, Mcold, Mbary,
    r50, R50, z50, 
    shape
    """
    dict['star']['mass']    = sim.s.M_vir
    dict['star']['spin']    = sim.s.spin
    dict['star']['krot']    = sim.s.krot
    dict['star']['beta']    = sim.s.beta
    dict['star']['AM']      = sim.s.AM
    dict['star']['vel_disp']= sim.s.vel_disp
    dict['star']['vr_disp'] = sim.s.vr_disp
    dict['star']['vR_disp'] = sim.s.vR_disp
    dict['star']['vz_disp'] = sim.s.vz_disp
    dict['star']['v_circ']  = sim.s.v_circ
    dict['star']['v_rot']   = sim.s.v_rot
    dict['star']['ke']      = sim.s.ke
    dict['star']['Mdyn']    = sim.s.Mdyn
    #dict['star']['Mcold']   = sim.s.Mcold
    dict['star']['Mbary']   = sim.s.Mbary
    dict['star']['r50']     = sim.s.r50
    dict['star']['R50']     = sim.s.R50
    dict['star']['z50']     = sim.s.z50
    dict['star']['t50']     = sim.s.t50
    dict['star']['shape']   = sim.s.shape
    
    dict['disk']['mass']    = sim.disk.M_vir
    dict['disk']['spin']    = sim.disk.spin
    dict['disk']['krot']    = sim.disk.krot
    dict['disk']['beta']    = sim.disk.beta
    dict['disk']['AM']      = sim.disk.AM
    dict['disk']['vel_disp']= sim.disk.vel_disp
    dict['disk']['vr_disp'] = sim.disk.vr_disp
    dict['disk']['vR_disp'] = sim.disk.vR_disp
    dict['disk']['vz_disp'] = sim.disk.vz_disp
    dict['disk']['v_circ']  = sim.disk.v_circ
    dict['disk']['v_rot']   = sim.disk.v_rot
    dict['disk']['ke']      = sim.disk.ke
    dict['disk']['Mdyn']    = sim.disk.Mdyn
    #dict['disk']['Mcold']   = sim.disk.Mcold
    dict['disk']['Mbary']   = sim.disk.Mbary
    dict['disk']['r50']     = sim.disk.r50
    dict['disk']['R50']     = sim.disk.R50
    dict['disk']['z50']     = sim.disk.z50
    dict['disk']['t50']     = sim.disk.t50
    dict['disk']['shape']   = sim.disk.shape
    dict['disk']['Mass_frac'] = sim.disk.Mass_frac
    
    dict['colddisk']['mass']    = sim.colddisk.M_vir
    dict['colddisk']['spin']    = sim.colddisk.spin
    dict['colddisk']['krot']    = sim.colddisk.krot
    dict['colddisk']['beta']    = sim.colddisk.beta
    dict['colddisk']['AM']      = sim.colddisk.AM
    dict['colddisk']['vel_disp']= sim.colddisk.vel_disp
    dict['colddisk']['vr_disp'] = sim.colddisk.vr_disp
    dict['colddisk']['vR_disp'] = sim.colddisk.vR_disp
    dict['colddisk']['vz_disp'] = sim.colddisk.vz_disp
    dict['colddisk']['v_circ']  = sim.colddisk.v_circ
    dict['colddisk']['v_rot']   = sim.colddisk.v_rot
    dict['colddisk']['ke']      = sim.colddisk.ke
    dict['colddisk']['Mdyn']    = sim.colddisk.Mdyn
    #dict['colddisk']['Mcold']   = sim.colddisk.Mcold
    dict['colddisk']['Mbary']   = sim.colddisk.Mbary
    dict['colddisk']['r50']     = sim.colddisk.r50
    dict['colddisk']['R50']     = sim.colddisk.R50
    dict['colddisk']['z50']     = sim.colddisk.z50
    dict['colddisk']['t50']     = sim.colddisk.t50
    dict['colddisk']['shape']   = sim.colddisk.shape
    dict['colddisk']['Mass_frac'] = sim.colddisk.Mass_frac

    dict['warmdisk']['mass']    = sim.warmdisk.M_vir
    dict['warmdisk']['spin']    = sim.warmdisk.spin
    dict['warmdisk']['krot']    = sim.warmdisk.krot
    dict['warmdisk']['beta']    = sim.warmdisk.beta
    dict['warmdisk']['AM']      = sim.warmdisk.AM
    dict['warmdisk']['vel_disp']= sim.warmdisk.vel_disp
    dict['warmdisk']['vr_disp'] = sim.warmdisk.vr_disp
    dict['warmdisk']['vR_disp'] = sim.warmdisk.vR_disp
    dict['warmdisk']['vz_disp'] = sim.warmdisk.vz_disp
    dict['warmdisk']['v_circ']  = sim.warmdisk.v_circ
    dict['warmdisk']['v_rot']   = sim.warmdisk.v_rot
    dict['warmdisk']['ke']      = sim.warmdisk.ke
    dict['warmdisk']['Mdyn']    = sim.warmdisk.Mdyn
    #dict['warmdisk']['Mcold']   = sim.warmdisk.Mcold
    dict['warmdisk']['Mbary']   = sim.warmdisk.Mbary
    dict['warmdisk']['r50']     = sim.warmdisk.r50
    dict['warmdisk']['R50']     = sim.warmdisk.R50
    dict['warmdisk']['z50']     = sim.warmdisk.z50
    dict['warmdisk']['t50']     = sim.warmdisk.t50
    dict['warmdisk']['shape']   = sim.warmdisk.shape
    dict['warmdisk']['Mass_frac'] = sim.warmdisk.Mass_frac

    dict['spheroid']['mass']    = sim.spheroid.M_vir
    dict['spheroid']['spin']    = sim.spheroid.spin
    dict['spheroid']['krot']    = sim.spheroid.krot
    dict['spheroid']['beta']    = sim.spheroid.beta
    dict['spheroid']['AM']      = sim.spheroid.AM
    dict['spheroid']['vel_disp']= sim.spheroid.vel_disp
    dict['spheroid']['vr_disp'] = sim.spheroid.vr_disp
    dict['spheroid']['vR_disp'] = sim.spheroid.vR_disp
    dict['spheroid']['vz_disp'] = sim.spheroid.vz_disp
    dict['spheroid']['v_circ']  = sim.spheroid.v_circ
    dict['spheroid']['v_rot']   = sim.spheroid.v_rot
    dict['spheroid']['ke']      = sim.spheroid.ke
    dict['spheroid']['Mdyn']    = sim.spheroid.Mdyn
    #dict['spheroid']['Mcold']   = sim.spheroid.Mcold
    dict['spheroid']['Mbary']   = sim.spheroid.Mbary
    dict['spheroid']['r50']     = sim.spheroid.r50
    dict['spheroid']['R50']     = sim.spheroid.R50
    dict['spheroid']['z50']     = sim.spheroid.z50
    dict['spheroid']['t50']     = sim.spheroid.t50
    dict['spheroid']['shape']   = sim.spheroid.shape
    dict['spheroid']['Mass_frac'] = sim.spheroid.Mass_frac

    dict['bulge']['mass']    = sim.bulge.M_vir
    dict['bulge']['spin']    = sim.bulge.spin
    dict['bulge']['krot']    = sim.bulge.krot
    dict['bulge']['beta']    = sim.bulge.beta
    dict['bulge']['AM']      = sim.bulge.AM
    dict['bulge']['vel_disp']= sim.bulge.vel_disp
    dict['bulge']['vr_disp'] = sim.bulge.vr_disp
    dict['bulge']['vR_disp'] = sim.bulge.vR_disp
    dict['bulge']['vz_disp'] = sim.bulge.vz_disp
    dict['bulge']['v_circ']  = sim.bulge.v_circ
    dict['bulge']['v_rot']   = sim.bulge.v_rot
    dict['bulge']['ke']      = sim.bulge.ke
    dict['bulge']['Mdyn']    = sim.bulge.Mdyn
    #dict['bulge']['Mcold']   = sim.bulge.Mcold
    dict['bulge']['Mbary']   = sim.bulge.Mbary
    dict['bulge']['r50']     = sim.bulge.r50
    dict['bulge']['R50']     = sim.bulge.R50
    dict['bulge']['z50']     = sim.bulge.z50
    dict['bulge']['t50']     = sim.bulge.t50
    dict['bulge']['shape']   = sim.bulge.shape
    dict['bulge']['Mass_frac'] = sim.bulge.Mass_frac

    dict['halo']['mass']    = sim.halo.M_vir
    dict['halo']['spin']    = sim.halo.spin
    dict['halo']['krot']    = sim.halo.krot
    dict['halo']['beta']    = sim.halo.beta
    dict['halo']['AM']      = sim.halo.AM
    dict['halo']['vel_disp']= sim.halo.vel_disp
    dict['halo']['vr_disp'] = sim.halo.vr_disp
    dict['halo']['vR_disp'] = sim.halo.vR_disp
    dict['halo']['vz_disp'] = sim.halo.vz_disp
    dict['halo']['v_circ']  = sim.halo.v_circ
    dict['halo']['v_rot']   = sim.halo.v_rot
    dict['halo']['ke']      = sim.halo.ke
    dict['halo']['Mdyn']    = sim.halo.Mdyn
    #dict['halo']['Mcold']   = sim.halo.Mcold
    dict['halo']['Mbary']   = sim.halo.Mbary
    dict['halo']['r50']     = sim.halo.r50
    dict['halo']['R50']     = sim.halo.R50
    dict['halo']['z50']     = sim.halo.z50
    dict['halo']['t50']     = sim.halo.t50
    dict['halo']['shape']   = sim.halo.shape
    dict['halo']['Mass_frac'] = sim.halo.Mass_frac

    '''
    dict['gas']['spin']    = sim.gas.spin
    dict['gas']['krot']    = sim.gas.krot
    dict['gas']['beta']    = sim.gas.beta
    dict['gas']['AM']      = sim.gas.AM
    dict['gas']['vel_disp']= sim.gas.vel_disp
    dict['gas']['vr_disp'] = sim.gas.vr_disp
    dict['gas']['vR_disp'] = sim.gas.vR_disp
    dict['gas']['vz_disp'] = sim.gas.vz_disp
    dict['gas']['v_circ']  = sim.gas.v_circ
    dict['gas']['v_rot']   = sim.gas.v_rot
    dict['gas']['ke']      = sim.gas.ke
    dict['gas']['Mdyn']    = sim.gas.Mdyn
    dict['gas']['Mcold']   = sim.gas.Mcold
    dict['gas']['Mbary']   = sim.gas.Mbary
    dict['gas']['r50']     = sim.gas.r50
    dict['gas']['R50']     = sim.gas.R50
    dict['gas']['z50']     = sim.gas.z50
    dict['gas']['shape']   = sim.gas.shape
    
    # TODO: Re-face on and use z<z50 & R<R50
    dict['coldgas']['spin']    = sim.coldgas.spin
    dict['coldgas']['krot']    = sim.coldgas.krot
    dict['coldgas']['beta']    = sim.coldgas.beta
    dict['coldgas']['AM']      = sim.coldgas.AM
    dict['coldgas']['vel_disp']= sim.coldgas.vel_disp
    dict['coldgas']['vr_disp'] = sim.coldgas.vr_disp
    dict['coldgas']['vR_disp'] = sim.coldgas.vR_disp
    dict['coldgas']['vz_disp'] = sim.coldgas.vz_disp
    dict['coldgas']['v_circ']  = sim.coldgas.v_circ
    dict['coldgas']['v_rot']   = sim.coldgas.v_rot
    dict['coldgas']['ke']      = sim.coldgas.ke
    dict['coldgas']['Mdyn']    = sim.coldgas.Mdyn
    dict['coldgas']['Mcold']   = sim.coldgas.Mcold
    dict['coldgas']['Mbary']   = sim.coldgas.Mbary
    dict['coldgas']['r50']     = sim.coldgas.r50
    dict['coldgas']['R50']     = sim.coldgas.R50
    dict['coldgas']['z50']     = sim.coldgas.z50
    dict['coldgas']['shape']   = sim.coldgas.shape
    '''
    return dict
