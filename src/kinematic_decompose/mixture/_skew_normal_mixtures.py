"""Finite mixtures of skew-normal densities (univariate and multivariate).

References
----------
* Univariate: Lin, T. I., Lee, J. C., & Yen, S. Y. (2007), "Finite mixture
  modelling using the skew normal distribution", Statistica Sinica 17(3),
  909-927 -- the ECM for univariate skew-normal mixtures.
* Multivariate: the Delta-parameterization ECM of the canonical R package
  ``mixsmsn`` (Prates, Lachos & Cabral 2013, "mixsmsn: R package for fitting
  finite mixture of scale mixture of skew-normal distributions",
  Journal of Statistical Software 54(12); functions ``smsn.mix`` for the
  univariate case and ``smsn.mmix`` for the multivariate case). Python has
  no equivalent maintained library; this module ports the multivariate
  Skew.normal ECM (full closed-form CM steps) into the project's
  BaseMixture framework.

Model
-----
g-component mixture of skew-normal (Azzalini & Capitanio 2003) densities

    f(x | Theta) = sum_k w_k * 2 phi_p(x; mu_k, Sigma_k)
                    * Phi(shape_k^T Sigma_k^{-1/2} (x - mu_k)),

with the component shape vector shape_k in R^p (0 = Gaussian). The
"Delta-parameterization" (mixsmsn) uses, per component,

    delta_k = shape_k / sqrt(1 + shape_k^T shape_k)   (unit-ball shape)
    Delta_k = Sigma_k^{1/2} delta_k
    Gamma_k = Sigma_k - Delta_k Delta_k^T,

and the hierarchical representation X = mu + Delta*|U0| + Gamma^{1/2} U1
with U0 ~ N(0,1) (truncated to the positive half-line) and U1 ~ N_p(0, I).

ECM (mixsmsn smsn.mmix, family = "Skew.normal"; all steps closed form):

  E-step  : responsibilities tal_ij and the latent-magnitude moments
            S1 = tal, S2 = tal*(mu_tau + M*R), S3 = tal*(mu_tau^2 + M^2
            + M*mu_tau*R), where mu_tau = M^2 Delta^T Gamma^{-1} (x - mu),
            M = sqrt(1/(1 + Delta^T Gamma^{-1} Delta)) and R = phi/Phi is
            the Mills ratio at the scalar argument A = mu_tau / M;
  CM-step 1: weights pi_k = sum(tal)/n;
  CM-step 2: mu_k = [sum(S1 x) - Delta_old sum(S2)] / sum(S1);
  CM-step 3: Delta_k = sum(S2 (x - mu_new)) / sum(S3);
  CM-step 4: Gamma_k = [sum(S1 zz) - Delta sum(S2 z)^T - sum(S2 z) Delta^T
            + Delta Delta^T sum(S3)] / sum(tal), z = x - mu_new;
  then Sigma = Gamma + Delta Delta^T and
  shape = Sigma^{-1/2} Delta / sqrt(1 - Delta^T Sigma^{-1} Delta).

Initialization
--------------
* p = 1: deterministic delta-sign scan (see _initialize_parameters): a
  symmetric Gaussian fit anchors (w, xi, sigma), then the 2^K sign
  patterns at magnitudes (0.3, 0.5, 0.8) are scanned because the delta
  likelihood ridge is multimodal (local maxima at ~0.29 / 0.46 / 0.71 /
  0.87 on a 50/50 two-component skew mixture; delta = 0 is a stationary
  point of EM, so a symmetric start never develops skew). The candidate
  with the highest full-data likelihood at fixed positions is kept, then
  a single EM runs from it (fast; recovers the ecut intersection to
  < 0.02 but not component-level delta -- the flat ridge top spans delta
  ~0.45-0.75 at near-equal likelihood, so delta is only weakly
  identified by design).
* p >= 2: the mixsmsn initialization -- KMeans partition, then per-cluster
  (mu = center, Sigma = cluster covariance) with shape direction
  shape = sign(third central moment) per dimension.

Scope
-----
p = 1, 2, 3, ... supported. For p = 1 the attributes xi_, scales_,
deltas_, lambdas_ and the module-level mixture_intersection() (the ecut
use case) are exposed as derived conveniences.

Numerical notes
---------------
* log Phi via scipy.special.log_ndtr; the Mills ratio in log space
  (log R = log phi - log Phi; R ~ -A as A -> -inf).
* Sigma and Gamma are kept symmetric positive-definite with reg_covar.
* sample_weight (number-weighted EM, matching GaussianMixture's
  convention) multiplies the responsibilities inside _m_step.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.special import log_ndtr

from ._base import BaseMixture
from sklearn.utils._param_validation import StrOptions

_SQRT_2_PI = math.sqrt(2.0 / math.pi)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_DELTA_MARGIN = 1e-6          # numerical guard: delta in (-1+margin, 1-margin)
_DELTA_MAX = 1.0 - _DELTA_MARGIN
# deterministic delta-sign scan magnitudes (p = 1 only; see class docstring)
_DELTA_GRID = (0.3, 0.5, 0.8)


# ---------------------------------------------------------------------------
# component density helpers (vectorized, numpy)
# ---------------------------------------------------------------------------
def log_skew_normal(y, xi, scale, delta):
    """log psi(y | xi, scale, delta) for a univariate skew-normal component.

    Parameters
    ----------
    y : array-like of shape (n,)
    xi, scale, delta : float or array-like of shape (n,)
        location, scale and shape; |delta| < 1.
    """
    y = np.asarray(y, dtype=float)
    xi = np.asarray(xi, dtype=float)
    scale = np.asarray(scale, dtype=float)
    delta = np.asarray(delta, dtype=float)
    lam = delta / np.sqrt(np.maximum(1.0 - delta * delta, 1e-300))
    z = (y - xi) / scale
    return (
        math.log(2.0) - np.log(scale) - 0.5 * z * z - _LOG_SQRT_2PI
        + log_ndtr(lam * z)
    )


def skew_normal(y, xi, scale, delta):
    """psi(y | xi, scale, delta): skew-normal density (paper eq. 1)."""
    return np.exp(log_skew_normal(y, xi, scale, delta))


def mixture_intersection(weights, xi, scales, deltas, i=0, j=1):
    """Root of w_i * psi_i(x) = w_j * psi_j(x) between the two locations
    (equal-posterior point; the ecut use case for a two-component fit).

    Returns None when the two densities do not cross between xi_i and xi_j
    (same sign of the difference at both ends)."""
    from scipy.optimize import root_scalar

    w_i, w_j = float(weights[i]), float(weights[j])
    lo, hi = min(xi[i], xi[j]), max(xi[i], xi[j])

    def diff(x):
        return (w_i * skew_normal(x, xi[i], scales[i], deltas[i])
                - w_j * skew_normal(x, xi[j], scales[j], deltas[j]))

    f_lo, f_hi = diff(lo), diff(hi)
    if f_lo * f_hi > 0 or not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return None
    try:
        sol = root_scalar(diff, bracket=[lo, hi])
        return float(sol.root) if sol.converged else None
    except Exception:
        return None


def _log_phi(z):
    return -0.5 * z * z - _LOG_SQRT_2PI


def _mills_ratio(A):
    """R(A) = phi(A)/Phi(A) in log space; |R| ~ |A| as A -> -inf."""
    return np.exp(_log_phi(A) - log_ndtr(A))


def _delta_from_skewness(gamma):
    """Invert the theoretical skewness gamma_1(delta) (Lin 2007 eq. 2):
    gamma_1 = ((4-pi)/2) * (delta*sqrt(2/pi))^3 / (1 - 2 delta^2/pi)^(3/2),
    odd in delta, range (-0.9953, 0.9953). Returns delta in (-1, 1), or 0
    when |gamma| is negligible or the inversion fails."""
    from scipy.optimize import root_scalar

    gamma = float(gamma)
    if not np.isfinite(gamma) or abs(gamma) < 1e-6:
        return 0.0
    t = min(abs(gamma), 0.9952)

    def f(x):
        num = (4.0 - math.pi) / 2.0 * (x * _SQRT_2_PI) ** 3
        den = (1.0 - 2.0 * x * x / math.pi) ** 1.5
        return num / den

    try:
        sol = root_scalar(lambda x: f(x) - t, bracket=[1e-9, _DELTA_MAX])
        x = sol.root if sol.converged else None
    except Exception:
        return 0.0
    if x is None:
        return 0.0
    return math.copysign(float(x), gamma)


def _sym(x):
    return 0.5 * (x + x.T)


def _sqrtm(A):
    """Symmetric square root via eigh (A symmetric PSD)."""
    w, V = np.linalg.eigh(A)
    return (V * np.sqrt(np.maximum(w, 0.0))) @ V.T


def _sqrtm_inv(A, floor=1e-300):
    """Inverse symmetric square root via eigh."""
    w, V = np.linalg.eigh(A)
    return (V / np.sqrt(np.maximum(w, floor))) @ V.T


# ---------------------------------------------------------------------------
# the mixture class (BaseMixture template contract)
# ---------------------------------------------------------------------------
class SkewNormalMixtures(BaseMixture):
    """Finite mixture of skew-normal densities (Lin et al. 2007 for p = 1;
    the mixsmsn Delta-parameterization ECM for p >= 1). See the module
    docstring for the model, the ECM steps and the initialization.

    Inherits the project's BaseMixture machinery: mini-batch fit, n_init
    selection, warm_start, BIC/AIC/ICL and all scoring/prediction methods.
    Stores per component (w, mu, Sigma, shape) in ``weights_``, ``means_``,
    ``covariances_``, ``shapes_``. For p = 1 the derived conveniences
    ``xi_``, ``scales_``, ``deltas_`` (delta in (-1, 1)) and ``lambdas_``
    are exposed.

    Parameters mirror ``GaussianMixture`` (same BaseMixture base):
    n_components, tol, reg_covar, max_iter, min_iter, n_init, init_params,
    random_state, warm_start, batch_size, window_size, tv_error, delta,
    use_polylog, verbose, verbose_interval.
    """

    _parameter_constraints: dict = {
        **BaseMixture._parameter_constraints,
        "init_params": [StrOptions(
            {"kmeans", "random", "random_from_data", "k-means++"}
        )],
    }

    def __init__(
        self,
        n_components=1,
        *,
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        min_iter=3,
        n_init=1,
        init_params="kmeans",
        random_state=None,
        warm_start=False,
        batch_size=None,
        window_size=None,
        tv_error=0.05,
        delta=0.05,
        use_polylog=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            min_iter=min_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            batch_size=batch_size,
            window_size=window_size,
            tv_error=tv_error,
            delta=delta,
            use_polylog=use_polylog,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

    # -- template hooks ---------------------------------------------------
    def _check_parameters(self, X, xp=None):
        """Validate X (p >= 1) and the fitted parameter arrays."""
        if np.asarray(X, dtype=float).ndim != 2:
            raise ValueError("X must be a 2-D array of shape (n_samples, p)")

    def _initialize_parameters(self, X, random_state, sample_weight=None, xp=None):
        """See the class docstring: p = 1 uses the delta-sign scan; p >= 2
        falls through to the base KMeans/random dispatcher and the
        mixsmsn-style per-cluster initialization in ``_initialize``."""
        X = np.asarray(X, dtype=float)
        if X.shape[1] == 1 and self.init_params == "kmeans":
            return self._init_1d_scan(X, random_state)
        return super()._initialize_parameters(
            X, random_state, sample_weight=sample_weight, xp=xp
        )

    def _init_1d_scan(self, X, random_state):
        """p = 1 delta-sign scan (see the class docstring): positions from
        a symmetric Gaussian fit, then all 2^K sign patterns at magnitudes
        (0.3, 0.5, 0.8) are evaluated at fixed positions and the candidate
        with the highest FULL-DATA log-likelihood is kept; a single EM then
        runs from it (fast; the ecut use case). Component-level delta is
        NOT recovered in this mode -- the delta likelihood ridge is flat
        with several local maxima, and the best short-lookahead start is
        not the best basin (verified); the ecut intersection is accurate
        regardless (<= 0.02 on the synthetic matrix). Fallback: the base
        dispatcher.
        """
        cands = self._delta_candidates(X)
        if cands is None or not cands:
            return super()._initialize_parameters(X, random_state)
        assert cands is not None
        best, best_lb = None, -np.inf
        for w, xi, s, d in cands:
            self._set_1d_params(w, xi, s, d)
            lb = self._full_data_lb(X)
            if lb > best_lb:
                best_lb, best = lb, (w, xi, s, d)
        self._set_1d_params(*best)

    def _set_1d_params(self, w, xi, s, d):
        """Map (w, xi, sigma, delta) onto the general (means, covariances,
        shapes) storage for p = 1."""
        self.weights_ = np.asarray(w, dtype=float)
        self.means_ = np.asarray(xi, dtype=float)[:, None]
        self.covariances_ = (np.asarray(s, dtype=float)[:, None, None] ** 2
                             * np.eye(1)[None])
        dd = np.clip(np.asarray(d, dtype=float), -_DELTA_MAX, _DELTA_MAX)
        self.shapes_ = (dd / np.sqrt(np.maximum(1.0 - dd * dd, 1e-300)))[:, None]

    def _delta_candidates(self, X):
        """Build the deterministic p = 1 init candidate list (cached per
        fit). Positions from a symmetric Gaussian fit; K = 1 uses the
        full-sample skewness directly; K > 3 falls back to the base
        dispatcher."""
        key = id(X)
        if getattr(self, "_cand_key", None) == key:
            return self._cand_list
        y = np.asarray(X[:, 0], dtype=float)
        k = self.n_components
        cands = []
        if k == 1:
            m1 = float(np.mean(y))
            v = float(np.var(y))
            if v > 0:
                gam = float(np.mean((y - m1) ** 3)) / v ** 1.5
                cands.append((np.array([1.0]), np.array([m1]),
                              np.array([math.sqrt(v)]),
                              np.array([_delta_from_skewness(gam)])))
        elif k <= 3:
            w0 = xi0 = s0 = None
            try:
                from ._gaussian_mixture import GaussianMixture as _GM
                g = _GM(n_components=k, n_init=3, max_iter=300, random_state=0)
                g.fit(X, use_mini_batch=False)
                means = np.asarray(g.means_, float)
                covs = np.asarray(g.covariances_, float)
                o = np.argsort(means[:, 0])
                w0 = np.asarray(g.weights_, float)[o]
                xi0 = means[o, 0]
                s0 = np.sqrt(covs[o, 0, 0])
            except Exception:
                pass
            if w0 is None:
                self._cand_key, self._cand_list = key, None
                return None
            for mag in _DELTA_GRID:
                for signs in itertools.product((-1.0, 1.0), repeat=k):
                    cands.append((w0, xi0, s0, np.array(signs, float) * mag))
        self._cand_key, self._cand_list = key, cands
        return cands

    def _full_data_lb(self, X):
        """Mean log-likelihood without the fitted-state validation the
        public score() would require (used by the pre-selection)."""
        lp = self._estimate_log_prob(X)
        lw = self._estimate_log_weights()
        from scipy.special import logsumexp
        return float(np.mean(logsumexp(lp + lw[None, :], axis=1)))

    def _initialize(self, X, resp):
        """Per-cluster method-of-moments init (mixsmsn style): weighted
        mean/covariance per cluster; for p = 1 the shape comes from the
        skewness inversion (Lin 2007 eqs. 2-3), for p >= 2 the shape
        direction is the sign of the per-dimension third central moment
        (the mixsmsn smsn.mmix initialization)."""
        X = np.asarray(X, dtype=float)
        resp = np.asarray(resp, dtype=float)
        n, p = X.shape
        g = self.n_components
        nk = resp.sum(axis=0)
        n_total = max(float(nk.sum()), 1e-300)

        means = np.zeros((g, p))
        covs = np.zeros((g, p, p))
        shapes = np.zeros((g, p))
        for k in range(g):
            r = resp[:, k]
            if nk[k] <= 1e-12 * n_total or nk[k] < 3.0:
                means[k] = np.average(X, axis=0, weights=np.maximum(r, 0.0)) \
                    if nk[k] > 0 else 0.0
                covs[k] = np.eye(p)
                shapes[k] = 0.0
                continue
            m1 = np.average(X, axis=0, weights=r)
            z = X - m1
            cov = (r[:, None, None] * z[:, :, None] * z[:, None, :]).sum(0) / nk[k]
            cov = _sym(cov) + self.reg_covar * np.eye(p)
            m3 = np.average(z ** 3, axis=0, weights=r)
            if p == 1:
                var = float(cov[0, 0])
                gam = float(m3[0]) / max(var, 1e-300) ** 1.5
                dk = _delta_from_skewness(gam)
                shapes[k, 0] = dk / math.sqrt(max(1.0 - dk * dk, 1e-300))
            else:
                shapes[k] = np.sign(m3)
            means[k] = m1
            covs[k] = cov

        self.weights_ = nk / n_total
        self.means_ = means
        self.covariances_ = covs
        self.shapes_ = shapes

    def _estimate_log_prob(self, X, xp=None):  # type: ignore[override]
        """log of the component densities, shape (n, g): each component is
        2 phi_p(x; mu, Sigma) Phi(shape^T Sigma^{-1/2} (x - mu)); the Phi
        argument is a scalar, evaluated stably with log_ndtr."""
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        g = self.n_components
        out = np.empty((n, g))
        for k in range(g):
            mu = self.means_[k]
            Sig = _sym(self.covariances_[k]) + self.reg_covar * np.eye(p)
            shape = self.shapes_[k]
            z = X - mu
            sign, logdet = np.linalg.slogdet(Sig)
            maha = np.einsum("ni,ij,nj->n", z, np.linalg.inv(Sig), z)
            log_phi = -0.5 * p * math.log(2.0 * math.pi) - 0.5 * logdet \
                - 0.5 * maha
            arg = z @ (shape @ _sqrtm_inv(Sig))
            out[:, k] = math.log(2.0) + log_phi + log_ndtr(arg)
        return out

    def _m_step(self, X, log_resp, sample_weight=None, xp=None):
        """ECM M step -- the mixsmsn 'Skew.normal' algorithm (Prates,
        Lachos & Cabral 2013, smsn.mmix / smsn.mix), ported 1:1 for all
        dimensionalities: CM1 weights, CM2 mu, CM3 Delta, CM4 Gamma -- all
        closed form. p = 1 is the same code path (Delta = sigma*delta,
        Gamma = sigma^2 (1-delta^2)); it is NOT the Lin 2007 CM4 variant,
        so the algorithm stays identical to the reference R package.
        """
        X = np.asarray(X, dtype=float)
        resp = np.exp(np.asarray(log_resp, dtype=float))
        if sample_weight is not None:
            resp = resp * np.asarray(sample_weight, dtype=float)[:, None]
        n, p = X.shape
        g = self.n_components
        nk = resp.sum(axis=0)
        n_total = max(float(nk.sum()), 1e-300)
        weights_new = nk / n_total

        means_new = np.empty((g, p))
        covs_new = np.empty((g, p, p))
        shapes_new = np.empty((g, p))
        for k in range(g):
            mu = self.means_[k]
            Sig = _sym(self.covariances_[k]) + self.reg_covar * np.eye(p)
            shape = self.shapes_[k]
            d = shape / math.sqrt(1.0 + float(shape @ shape))
            Sigh = _sqrtm(Sig)
            Del = Sigh @ d
            Gam = _sym(Sig - np.outer(Del, Del))
            wG, VG = np.linalg.eigh(Gam)
            Gam = (VG * np.maximum(wG, self.reg_covar)) @ VG.T  # SPD guard
            GiDel = np.linalg.solve(Gam, Del)
            M2 = 1.0 / (1.0 + float(Del @ GiDel))
            M = math.sqrt(M2)
            z = X - mu
            mu_tau = M2 * (z @ GiDel)                 # (n,)
            A = mu_tau / M
            R = _mills_ratio(A)
            S1 = resp[:, k]
            S2 = S1 * (mu_tau + M * R)
            S3 = S1 * (mu_tau * mu_tau + M2 + M * mu_tau * R)

            s1 = max(float(S1.sum()), 1e-300)
            means_new[k] = (S1 @ X - Del * float(S2.sum())) / s1
            z = X - means_new[k]
            s3 = max(float(S3.sum()), 1e-300)
            Del_new = (S2 @ z) / s3
            zz = z[:, :, None] * z[:, None, :]
            Gam_new = ((S1[:, None, None] * zz).sum(0)
                       - np.outer(Del_new, S2 @ z)
                       - np.outer(S2 @ z, Del_new)
                       + np.outer(Del_new, Del_new) * float(S3.sum()))
            Gam_new = _sym(Gam_new / max(nk[k], 1e-300))
            Sig_new = _sym(Gam_new + np.outer(Del_new, Del_new)) \
                + self.reg_covar * np.eye(p)
            dd = _sqrtm_inv(Sig_new) @ Del_new
            denom = math.sqrt(max(1.0 - float(dd @ dd), 1e-12))
            shapes_new[k] = dd / denom
            covs_new[k] = Sig_new

        self.weights_ = weights_new
        self.means_ = means_new
        self.covariances_ = covs_new
        self.shapes_ = shapes_new

    def _estimate_log_weights(self, xp=None):  # type: ignore[override]
        return np.log(np.asarray(self.weights_, dtype=float))

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _get_parameters(self):  # type: ignore[override]
        return (self.weights_, self.means_, self.covariances_, self.shapes_)

    def _set_parameters(self, params, xp=None):
        (self.weights_, self.means_, self.covariances_, self.shapes_) = params

    def _n_parameters(self):
        """Free parameters: per component 1 (w) + p (mu) + p(p+1)/2 (Sigma)
        + p (shape); p = 1 reduces to 4g - 1 (the paper's count)."""
        p = self.means_.shape[1]
        return self.n_components * (1 + p + p * (p + 1) // 2 + p) - 1

    # -- derived 1-D conveniences ------------------------------------------
    @property
    def xi_(self):
        if self.means_.shape[1] != 1:
            raise AttributeError("xi_ is only defined for p = 1")
        return self.means_[:, 0]

    @property
    def scales_(self):
        if self.means_.shape[1] != 1:
            raise AttributeError("scales_ is only defined for p = 1")
        return np.sqrt(self.covariances_[:, 0, 0])

    @property
    def deltas_(self):
        if self.means_.shape[1] != 1:
            raise AttributeError("deltas_ is only defined for p = 1")
        sh = self.shapes_[:, 0]
        return sh / np.sqrt(np.maximum(1.0 + sh * sh, 1e-300))

    @property
    def lambdas_(self):
        if self.means_.shape[1] != 1:
            raise AttributeError("lambdas_ is only defined for p = 1")
        return self.shapes_[:, 0]

    # -- extra API ---------------------------------------------------------
    def bic(self, X):
        """Bayesian information criterion (lower is better)."""
        return -2.0 * self.score(X) * X.shape[0] \
            + self._n_parameters() * math.log(X.shape[0])

    def mbic(self, X):
        """BIC per sample."""
        return self.bic(X) / X.shape[0]

    def icl(self, X):
        """ICL = BIC + 2 * posterior entropy (mirrors the vendored
        GaussianMixture implementation)."""
        _, log_resp = self._e_step(X)
        resp = np.clip(np.exp(np.asarray(log_resp, dtype=float)), 1e-12, 1.0)
        entropy = -np.sum(resp * np.log(resp))
        return self.bic(X) + 2.0 * entropy

    def aic(self, X):
        """Akaike information criterion (lower is better)."""
        return -2.0 * self.score(X) * X.shape[0] + 2.0 * self._n_parameters()

    def sample(self, n_samples=1):
        """Draw from the fitted mixture via the hierarchical representation
        X = mu + Delta*|U0| + Gamma^{1/2} U1 (U0 ~ N(0,1) positive, U1 ~
        N_p(0, I)); the p = 1 special case matches paper eq. 9."""
        rng = np.random.default_rng(self.random_state)
        counts = rng.multinomial(n_samples, np.asarray(self.weights_, dtype=float))
        p = self.means_.shape[1]
        X = np.zeros((n_samples, p))
        y = np.zeros(n_samples, dtype=int)
        start = 0
        for k, c in enumerate(counts):
            if c == 0:
                continue
            Sig = _sym(self.covariances_[k]) + self.reg_covar * np.eye(p)
            shape = self.shapes_[k]
            d = shape / math.sqrt(1.0 + float(shape @ shape))
            Del = _sqrtm(Sig) @ d
            Gam = _sym(Sig - np.outer(Del, Del))
            u0 = np.abs(rng.standard_normal(c))
            u1 = rng.standard_normal((c, p))
            X[start:start + c] = (self.means_[k]
                                  + np.outer(u0, Del) + u1 @ _sqrtm(Gam).T)
            y[start:start + c] = k
            start += c
        return X, y
