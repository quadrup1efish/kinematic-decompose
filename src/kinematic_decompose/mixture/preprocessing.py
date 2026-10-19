import numpy as np
from copy import deepcopy
from ._gaussian_mixture import _compute_precision_cholesky

class RobustScaler():
    def __init__(self, quantile_range=[25, 75]):
        self.quantile_range = quantile_range
        
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if np.any(np.isinf(X)):
            raise ValueError("X must not contain infinite values")
        if np.any(np.all(np.isnan(X), axis=0)):
            raise ValueError("cannot fit RobustScaler on an all-NaN feature")
        self.center_ = np.nanmedian(X, axis=0)
        quantiles = np.nanpercentile(X, self.quantile_range, axis=0)
        self.scale_ = np.abs(np.subtract(*quantiles))
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ == 0)] = 1.0
        return self
    
    def transform(self, X, columns=None):
        if columns is None:
            return (X - self.center_) / self.scale_
        else:
            return (X - self.center_[columns]) / self.scale_[columns]
    
    def fit_transform(self, X):
        return self.fit(X).transform(np.asarray(X, dtype=float))
    
    def inverse_transform(self, X, columns=None):
        if columns is None:
            return X * self.scale_ + self.center_
        else:
            return X * self.scale_[columns] + self.center_[columns]
    
    def inverse_transform_GMM(self, gmm):
        """Return an independent full-covariance GMM in original coordinates.

        The input model and its training statistics remain unchanged. Rebuild
        both precision representations from the transformed covariance before
        using the copy for inference. Copied lower bounds remain historical
        statistics of the standardized-space fit, not original-space scores.
        """
        if gmm.covariance_type != "full":
            raise ValueError("inverse_transform_GMM supports only full covariance matrices")
        transformed_gmm = deepcopy(gmm)
        n_features = gmm.means_.shape[1]
        if n_features == len(self.scale_):
            means = gmm.means_.copy() * self.scale_ + self.center_
            transformed_gmm.means_ = means
            scale_matrix = np.outer(self.scale_, self.scale_)
            covariances = gmm.covariances_.copy() * scale_matrix[np.newaxis, :, :]
            transformed_gmm.covariances_ = covariances
        else:
            # fallback: only transform the first 2 dimensions
            means = gmm.means_.copy() * self.scale_[:2] + self.center_[:2]
            transformed_gmm.means_ = means
            scale_matrix = np.outer(self.scale_[:2], self.scale_[:2])
            covariances = gmm.covariances_.copy() * scale_matrix[np.newaxis, :, :]
            transformed_gmm.covariances_ = covariances
        precisions_cholesky = _compute_precision_cholesky(
            transformed_gmm.covariances_, transformed_gmm.covariance_type)
        transformed_gmm._set_parameters((
            transformed_gmm.weights_, transformed_gmm.means_,
            transformed_gmm.covariances_, precisions_cholesky))
        return transformed_gmm
