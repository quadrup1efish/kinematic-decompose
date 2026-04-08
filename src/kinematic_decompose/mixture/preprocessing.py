import numpy as np
from copy import deepcopy

class RobustScaler():
    def __init__(self, quantile_range=[25, 75]):
        self.quantile_range = quantile_range
        
    def fit(self, X):
        self.center_ = np.nanmedian(X, axis=0)
        self.scale_  = np.abs(np.subtract(*np.percentile(X, self.quantile_range, axis=0)))
        return self
    
    def transform(self, X):        
        X -= self.center_
        X /= self.scale_
        return X
    
    def fit_transform(self, X):
        self.center_ = np.nanmedian(X, axis=0)
        self.scale_  = np.abs(np.subtract(*np.percentile(X, self.quantile_range, axis=0)))
        X -= self.center_
        X /= self.scale_
        return X
    
    def inverse_transform(self, X):
        X *= self.scale_
        X += self.center_
        return X
    
    def inverse_transform_GMM(self, gmm):
        transformed_gmm = deepcopy(gmm)
        try:
            means=gmm.means_.copy()
            means=means*self.scale_+self.center_
            transformed_gmm.means_ = means

            covariances=gmm.covariances_.copy()
            scale_matrix = np.outer(self.scale_, self.scale_)
            covariances = covariances * scale_matrix[np.newaxis, :, :]
            transformed_gmm.covariances_ = covariances
        except:
            means=gmm.means_.copy()
            means=means*self.scale_[:2]+self.center_[:2]
            transformed_gmm.means_ = means

            covariances=gmm.covariances_.copy()
            scale_matrix = np.outer(self.scale_[:2], self.scale_[:2])
            covariances = covariances * scale_matrix[np.newaxis, :, :]
            transformed_gmm.covariances_ = covariances
        return transformed_gmm
