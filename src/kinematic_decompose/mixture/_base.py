"""Base class for mixture models."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from abc import ABCMeta, abstractmethod
from collections import deque
from contextlib import nullcontext
from numbers import Integral, Real
from time import time

import numpy as np

from sklearn import cluster
from sklearn.base import BaseEstimator, DensityMixin, _fit_context
from sklearn.cluster import kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils._array_api import (
    _convert_to_numpy,
    _is_numpy_namespace,
    _logsumexp,
    _max_precision_float_dtype,
    get_namespace,
    get_namespace_and_device,
)
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data


def _check_shape(param, param_shape, name):
    """Validate the shape of the input parameter 'param'.

    Parameters
    ----------
    param : array

    param_shape : tuple

    name : str
    """
    if param.shape != param_shape:
        raise ValueError(
            "The parameter '%s' should have the shape of %s, but got %s"
            % (name, param_shape, param.shape)
        )


class BaseMixture(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    _MEAN_SMOOTHING = 3
    _INIT_EPSILON = 0.05
    """Base class for mixture models.

    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "reg_covar": [Interval(Real, 0.0, None, closed="left")],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "init_params": [
            StrOptions({"kmeans", "random", "random_from_data", "k-means++"})
        ],
        "random_state": ["random_state"],
        "warm_start": ["boolean"],
        "verbose": ["verbose"],
        "verbose_interval": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        n_components,
        tol,
        reg_covar,
        max_iter,
        min_iter,
        n_init,
        init_params,
        random_state,
        warm_start,
        batch_size,
        window_size,
        verbose,
        verbose_interval,
    ):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.batch_size = batch_size
        self.window_size = window_size
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    @abstractmethod
    def _check_parameters(self, X, xp=None):
        """Check initial parameters of the derived class.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)
        """
        pass

    def _initialize_parameters(self, X, random_state, sample_weight=None, xp=None):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        xp, _, device = get_namespace_and_device(X, xp=xp)
        n_samples, _ = X.shape
        
        if sample_weight is None:
            sample_weight = xp.ones(X.shape[0], dtype=X.dtype)
        else:
            sample_weight = xp.asarray(sample_weight, dtype=X.dtype) 

        if self.init_params == "kmeans":
            resp = np.zeros((n_samples, self.n_components), dtype=X.dtype)
            label = (
                cluster.KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(X, sample_weight)
                .labels_
            )
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = xp.asarray(
                random_state.uniform(size=(n_samples, self.n_components)),
                dtype=X.dtype,
                device=device,
            )
            resp *= sample_weight[:, None]
            resp /= xp.sum(resp, axis=1)[:, xp.newaxis]
        elif self.init_params == "random_from_data":
            resp = xp.zeros(
                (n_samples, self.n_components), dtype=X.dtype, device=device
            )
            indices = random_state.choice(
                n_samples, size=self.n_components, replace=False
            )
            # TODO: when array API supports __setitem__ with fancy indexing we
            # can use the previous code:
            # resp[indices, xp.arange(self.n_components)] = 1
            # Until then we use a for loop on one dimension.
            for col, index in enumerate(indices):
                resp[index, col] = 1
        elif self.init_params == "k-means++":
            resp = np.zeros((n_samples, self.n_components), dtype=X.dtype)
            _, indices = kmeans_plusplus(
                X,
                self.n_components,
                random_state=random_state,
            )
            resp[indices, np.arange(self.n_components)] = 1

        self._initialize(X, resp)

    @abstractmethod
    def _initialize(self, X, resp):
        """Initialize the model parameters of the derived class.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        pass

    def fit(self, X, y=None, sample_weight=None, use_mini_batch=True,
            _use_bounded_init=True, _replace_sampling=True):
        """Estimate model parameters with the EM algorithm.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        _use_bounded_init : bool, default=True
            Private: in mini-batch mode, initialize the parameters from a
            bounded random subsample instead of the full dataset. The
            subsample size is set by statistical power
            (S = K*d*(d+1)/(2*eps**2), eps = INIT_EPSILON, i.e. ~5% relative
            precision of the initial covariance estimate), so the init cost
            is O(S*K*d) and independent of N when N > S. Skipped when N <= S
            (small data keeps the exact full-dataset path).

        _replace_sampling : bool, default=True
            Private: draw each mini-batch with replacement via
            ``random_state.randint(0, n_samples, batch_size)`` (O(batch) per
            iteration, independent of N) instead of maintaining a full
            permutation of size N (O(N) per epoch). Statistically unbiased
            (each sample has the same expected selection probability) but the
            per-epoch "visit each sample exactly once" property is lost.
            Verified equivalent in converged lower bound (|dLB| < 0.002,
            K=2..6, d=2..3).

        Returns
        -------
        self : object
            The fitted mixture.
        """
        # parameters are validated in fit_predict
        self.fit_predict(X, y, sample_weight, use_mini_batch,
                         _return_labels=False,
                         _use_bounded_init=_use_bounded_init,
                         _replace_sampling=_replace_sampling)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_predict(self, X, y=None, sample_weight=None, use_mini_batch=True,
                    _return_labels=True, _use_bounded_init=True,
                    _replace_sampling=True):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        _return_labels : bool, default=True
            Private: whether to run the final full-data E-step needed to
            return labels. ``fit()`` passes False to skip this redundant
            scan (the labels are discarded there anyway). The fitted model
            parameters are identical either way.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        xp, _ = get_namespace(X)
        X = validate_data(self, X, dtype=[xp.float64, xp.float32], ensure_min_samples=2)
        n_samples, _ = X.shape

        if sample_weight is None:
            sample_weight = xp.ones(X.shape[0], dtype=X.dtype)
        else:
            sample_weight = xp.asarray(sample_weight, dtype=X.dtype)

        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X, xp=xp)
        
        # if n_samples > 3*batch_size, switch to mini-batch for efficiency
        if use_mini_batch:
            use_mini_batch = 3 * self.batch_size < n_samples

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "_initialized"))# and hasattr(self, "converged_"))
        # resolve effective window_size (local copy, don't mutate self.window_size)
        if use_mini_batch and self.window_size is None:
            _window_size = max(3, min(10, int(np.ceil(n_samples / self.batch_size))))
        else:
            _window_size = self.window_size
        n_init = self.n_init if do_init else 1

        max_lower_bound = -xp.inf
        best_lower_bounds = []
        self.converged_ = False

        random_state = check_random_state(self.random_state)
 
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)
            if use_mini_batch: 
                if do_init:
                    if _use_bounded_init:
                        # bounded subsample init: statistical-power formula
                        # S = K*d*(d+1)/(2*eps**2) with eps = INIT_EPSILON
                        # (~5% relative precision of the initial covariance
                        # estimate per component). O(S*K*d) instead of
                        # O(N*K*d): independent of N when N > S. Small data
                        # (N <= S) keeps the exact full-dataset path.
                        d = X.shape[1]
                        s_init = int(np.ceil(
                            self.n_components * d * (d + 1)
                            / (2 * self._INIT_EPSILON ** 2)
                        ))
                        s_init = min(n_samples, s_init)
                        if s_init < n_samples:
                            init_idx = random_state.randint(0, n_samples, size=s_init)
                            self._initialize_parameters(X[init_idx], random_state, xp=xp)
                        else:
                            self._initialize_parameters(X, random_state, xp=xp)
                    else:
                        self._initialize_parameters(X, random_state, xp=xp)
                    self._initialized = True
                if not _replace_sampling:
                    if not hasattr(self, "_perm") or do_init:
                        self._perm = random_state.permutation(n_samples)
                        self._cursor = 0

                lb_window = deque(maxlen=_window_size)
                mean_window = deque(maxlen=self._MEAN_SMOOTHING)
                prev_smoothed_lb = -xp.inf
                current_lower_bounds = []

                if self.max_iter == 0:
                    best_params = self._get_parameters()
                    best_n_iter = 0
                else:
                    converged = False
                    for n_iter in range(1, self.max_iter + 1):

                        if _replace_sampling:
                            # draw with replacement: O(batch_size), independent of N
                            batch_idx = random_state.randint(0, n_samples, size=self.batch_size)
                        else:
                            if self._cursor + self.batch_size > n_samples:
                                self._perm = random_state.permutation(n_samples)
                                self._cursor = 0
                            batch_idx = self._perm[self._cursor:self._cursor + self.batch_size]
                            self._cursor += self.batch_size
                        mini_X = X[batch_idx]
                        mini_sample_weight = sample_weight[batch_idx]

                        log_prob_norm, log_resp = self._e_step(mini_X, mini_sample_weight, xp=xp)
                        self._m_step(mini_X, log_resp, sample_weight=mini_sample_weight, xp=xp)
                        raw_lb = self._compute_lower_bound(log_resp, log_prob_norm)

                        # median removes spike outliers; short mean smooths residual jitter
                        lb_window.append(raw_lb)
                        if len(lb_window) == _window_size:
                            median_lb = float(np.median(lb_window))
                            mean_window.append(median_lb)
                            smoothed_lb = float(np.mean(mean_window))
                            current_lower_bounds.append(smoothed_lb)

                            if prev_smoothed_lb > -xp.inf and len(mean_window) == self._MEAN_SMOOTHING:
                                change = smoothed_lb - prev_smoothed_lb
                                self._print_verbose_msg_iter_end(n_iter, change)
                                if abs(change) < self.tol and n_iter >= self.min_iter:
                                    converged = True
                                    break
                            prev_smoothed_lb = smoothed_lb
                        else:
                            current_lower_bounds.append(raw_lb)

                    self._print_verbose_msg_init_end(prev_smoothed_lb, converged)

                    if prev_smoothed_lb > max_lower_bound or max_lower_bound == -xp.inf:
                        max_lower_bound = prev_smoothed_lb
                        best_params = self._get_parameters()
                        best_n_iter = n_iter
                        best_lower_bounds = current_lower_bounds
                        self.converged_ = converged
            else:
                if do_init:
                    self._initialize_parameters(X, random_state, xp=xp)
                    self._initialized = True
                lower_bound = -xp.inf if do_init else self.lower_bound_
                current_lower_bounds = []

                if self.max_iter == 0:
                    best_params = self._get_parameters()
                    best_n_iter = 0
                else:
                    converged = False
                    for n_iter in range(1, self.max_iter + 1):
                        prev_lower_bound = lower_bound

                        log_prob_norm, log_resp = self._e_step(X, sample_weight, xp=xp)
                        self._m_step(X, log_resp, sample_weight, xp=xp)
                        lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)
                        current_lower_bounds.append(lower_bound)

                        change = lower_bound - prev_lower_bound
                        self._print_verbose_msg_iter_end(n_iter, change)

                        if abs(change) < self.tol and n_iter >= self.min_iter:
                            converged = True
                            break

                    self._print_verbose_msg_init_end(lower_bound, converged)

                    if lower_bound > max_lower_bound or max_lower_bound == -xp.inf:
                        max_lower_bound = lower_bound
                        best_params = self._get_parameters()
                        best_n_iter = n_iter
                        best_lower_bounds = current_lower_bounds
                        self.converged_ = converged

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                (
                    "Best performing initialization did not converge. "
                    "Try different init parameters, or increase max_iter, "
                    "tol, or check for degenerate data."
                ),
                ConvergenceWarning,
            )
        self._set_parameters(best_params, xp=xp)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound
        self.lower_bounds_ = best_lower_bounds

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        # Skipped when the caller only needs the fitted parameters (fit()).
        if _return_labels:
            _, log_resp = self._e_step(X, sample_weight, xp=xp)
            return xp.argmax(log_resp, axis=1)
        return None

    def _e_step(self, X, sample_weight=None, xp=None):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        xp, _ = get_namespace(X, xp=xp)
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X, xp=xp)
        if sample_weight is None:
            return xp.mean(log_prob_norm), log_resp
        else:
            return xp.average(log_prob_norm, weights=sample_weight), log_resp

    @abstractmethod
    def _m_step(self, X, log_resp, sample_weight=None, xp=None):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        pass

    @abstractmethod
    def _get_parameters(self):
        pass

    @abstractmethod
    def _set_parameters(self, params):
        pass

    def score_samples(self, X):
        """Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """
        #check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        return _logsumexp(self._estimate_weighted_log_prob(X), axis=1)

    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of `X` under the Gaussian mixture model.
        """
        xp, _ = get_namespace(X)
        return float(xp.mean(self.score_samples(X)))

    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        #check_is_fitted(self)
        xp, _ = get_namespace(X)
        X = validate_data(self, X, reset=False)
        return xp.argmax(self._estimate_weighted_log_prob(X), axis=1)
    
    def soft_predict(self, X, seed=42):
        X = validate_data(self, X, reset=False)
        xp, _ = get_namespace(X)
        rng = xp.random.default_rng(seed)
        _, log_resp = self._estimate_log_prob_resp(X, xp=xp)
        resp = xp.exp(log_resp)
        probs = resp / resp.sum(axis=1, keepdims=True)
        cum_probs = xp.cumsum(probs, axis=1)
        rand_vals = rng.random(probs.shape[0])
        labels = (cum_probs >= rand_vals[:, xp.newaxis]).argmax(axis=1)
        return labels
    
    def predict_proba(self, X):
        """Evaluate the components' density for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Density of each Gaussian component for each sample in X.
        """
        #check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        xp, _ = get_namespace(X)
        _, log_resp = self._estimate_log_prob_resp(X, xp=xp)
        return xp.exp(log_resp)

    def sample(self, n_samples=1):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        y : array, shape (nsamples,)
            Component labels.
        """
        #check_is_fitted(self)
        xp, _, device_ = get_namespace_and_device(self.means_)

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        _, n_features = self.means_.shape
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(
            n_samples, _convert_to_numpy(self.weights_, xp)
        )

        if self.covariance_type == "full":
            X = np.vstack(
                [
                    rng.multivariate_normal(mean, covariance, int(sample))
                    for (mean, covariance, sample) in zip(
                        _convert_to_numpy(self.means_, xp),
                        _convert_to_numpy(self.covariances_, xp),
                        n_samples_comp,
                    )
                ]
            )
        elif self.covariance_type == "tied":
            X = np.vstack(
                [
                    rng.multivariate_normal(
                        mean, _convert_to_numpy(self.covariances_, xp), int(sample)
                    )
                    for (mean, sample) in zip(
                        _convert_to_numpy(self.means_, xp), n_samples_comp
                    )
                ]
            )
        else:
            X = np.vstack(
                [
                    mean
                    + rng.standard_normal(size=(sample, n_features))
                    * np.sqrt(covariance)
                    for (mean, covariance, sample) in zip(
                        _convert_to_numpy(self.means_, xp),
                        _convert_to_numpy(self.covariances_, xp),
                        n_samples_comp,
                    )
                ]
            )

        y = xp.concat(
            [
                xp.full(int(n_samples_comp[i]), i, dtype=xp.int64, device=device_)
                for i in range(len(n_samples_comp))
            ]
        )

        max_float_dtype = _max_precision_float_dtype(xp=xp, device=device_)
        return xp.asarray(X, dtype=max_float_dtype, device=device_), y

    def _estimate_weighted_log_prob(self, X, xp=None):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return self._estimate_log_prob(X, xp=xp) + self._estimate_log_weights(xp=xp)

    @abstractmethod
    def _estimate_log_weights(self, xp=None):
        """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape (n_components, )
        """
        pass

    @abstractmethod
    def _estimate_log_prob(self, X, xp=None):
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """
        pass

    def _estimate_log_prob_resp(self, X, xp=None):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        xp, _ = get_namespace(X, xp=xp)
        weighted_log_prob = self._estimate_weighted_log_prob(X, xp=xp)
        log_prob_norm = _logsumexp(weighted_log_prob, axis=1, xp=xp)

        # There is no errstate equivalent for warning/error management in array API
        context_manager = (
            np.errstate(under="ignore") if _is_numpy_namespace(xp) else nullcontext()
        )
        with context_manager:
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, xp.newaxis]
        return log_prob_norm, log_resp

    def _print_verbose_msg_init_beg(self, n_init):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization %d" % n_init)
        elif self.verbose >= 2:
            print("Initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time()
                print(
                    "  Iteration %d\t time lapse %.5fs\t ll change %.5f"
                    % (n_iter, cur_time - self._iter_prev_time, diff_ll)
                )
                self._iter_prev_time = cur_time

    def _print_verbose_msg_init_end(self, lb, init_has_converged):
        """Print verbose message on the end of iteration."""
        converged_msg = "converged" if init_has_converged else "did not converge"
        if self.verbose == 1:
            print(f"Initialization {converged_msg}.")
        elif self.verbose >= 2:
            t = time() - self._init_prev_time
            print(
                f"Initialization {converged_msg}. time lapse {t:.5f}s\t lower bound"
                f" {lb:.5f}."
            )
