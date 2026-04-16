import numpy as np
from scipy.ndimage import label

from .util import *
from ._gaussian_mixture import GaussianMixture as GM

def hist_bin_fd(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)

MIN_MAHAL = 0.5
MIN_WEIGHT = 0.01
MAX_N_COMPONENTS = 15
MIN_POINTS = 10
CBIC = -0.1
BF = 0.95

class AutoGaussianMixtureModel:
    def __init__(self, n_components=None, model=None, dim=None, morphology_type=None):
        self.n_components = n_components
        self.morphology_type = morphology_type

    def _morphology_class(self, X, jzojc_cut):
        self.morphology_model = GM(n_components=3, n_init=1, init_params='kmeans', max_iter=200, min_iter=50, tol=1e-4).fit(X)
        if np.max(self.morphology_model.means_[:,1]) > jzojc_cut:
            self.morphology_type='disk'
        else:
            self.morphology_type='spheroid'
            self.morphology_model = GM(n_components=2, n_init=1, init_params='kmeans', max_iter=200, min_iter=10, tol=1e-3).fit(X)
        return self.morphology_model
    
    def _initialize(self, X, model, eoemin_cut, jzojc_cut, r_jzojc_cut,  
                    eoemin_index=0, jzojc_index=1, jpojc_index=2, disable_zero_rotation='halo', scaler=None):
        if model is None and getattr('morphology_model'):
            model = self.morphology_model
            
        old_weights = model.weights_
        old_means = model.means_
        old_covariances = model.covariances_
    
        labels = model.soft_predict(X)
        spheroid_labels = np.where((model.means_[:,jzojc_index] < jzojc_cut) & ((model.means_[:,jzojc_index] > r_jzojc_cut)))[0]
        halo_labels = np.where((model.means_[:,jzojc_index] < jzojc_cut) & 
                            (model.means_[:,jzojc_index] > r_jzojc_cut) & 
                            (model.means_[:,eoemin_index] > eoemin_cut))[0]
        bulge_labels = np.where((model.means_[:,jzojc_index] < jzojc_cut) & 
                            (model.means_[:,jzojc_index] > r_jzojc_cut) & 
                            (model.means_[:,eoemin_index] < eoemin_cut))[0]
        
        disk_labels = np.setdiff1d(np.unique(labels), spheroid_labels)
        spheroid_X = X[np.isin(labels,spheroid_labels)]
        bulge_X = spheroid_X[spheroid_X[:,eoemin_index]< eoemin_cut]
        halo_X  = spheroid_X[(spheroid_X[:,eoemin_index]>=eoemin_cut) &
                             ((spheroid_X[:,jzojc_index]<=jzojc_cut)) &
                             ((spheroid_X[:,jzojc_index]>=r_jzojc_cut))]
        
        if disable_zero_rotation == 'halo' and scaler is not None:
            halo_X_raw = scaler.inverse_transform(halo_X, columns=[eoemin_index, jzojc_index])
            sph, _ = JEHistogram(halo_X_raw[:, eoemin_index], halo_X_raw[:, jzojc_index])
            halo_X = halo_X[sph]
            del halo_X_raw, sph
                    
        if (len(halo_labels) != 0 & len(bulge_labels) != 0):
            halo_weight = model.weights_[halo_labels][0]
            halo_mean   = model.means_[halo_labels][0]
            halo_covariance  = model.covariances_[halo_labels][0]
            
            bulge_weight = model.weights_[bulge_labels][0]
            bulge_mean   = model.means_[bulge_labels][0]
            bulge_covariance = model.covariances_[bulge_labels][0]
            #mahal_between = abs(halo_mean[0] - bulge_mean[0]) / np.sqrt((halo_covariance[0,0]**2 + bulge_covariance[0,0]**2)/2)
            if bulge_weight > len(bulge_X)/len(X):
                bulge_weight = len(bulge_X)/len(X)
                bulge_mean   = np.mean(bulge_X, axis=0)
                bulge_covariance = np.cov(bulge_X, rowvar=False)
            
            if halo_weight > len(halo_X)/len(X):
                halo_weight  = len(halo_X)/len(X)
                halo_mean    = np.mean(halo_X, axis=0)
                halo_covariance  = np.cov(halo_X, rowvar=False)
        else:
            bulge_weight = len(bulge_X)/len(X)
            bulge_mean   = np.mean(bulge_X, axis=0)
            bulge_covariance = np.cov(bulge_X, rowvar=False)
            
            halo_weight  = len(halo_X)/len(X)
            halo_mean    = np.mean(halo_X, axis=0)
            halo_covariance  = np.cov(halo_X, rowvar=False)
        
        new_weights = old_weights[disk_labels].tolist()
        new_means = old_means[disk_labels].tolist()
        new_covariances = old_covariances[disk_labels].tolist()

        new_weights.extend([bulge_weight, halo_weight])
        new_means.extend([bulge_mean, halo_mean])
        new_covariances.extend([bulge_covariance, halo_covariance])
        
        weights_init = np.array(new_weights)
        weights_init/=weights_init.sum()
        means_init = np.array(new_means)
        covariances_init=np.array(new_covariances)
        precisions_init  = np.linalg.pinv(covariances_init)

        self.initial_model = GM(len(new_weights), 
                                weights_init=weights_init, 
                                means_init=means_init, 
                                precisions_init=precisions_init, 
                                max_iter=0,
                                min_iter=0).fit(X)
        
        return self.initial_model
    
    def _find_residual_component(self, X, model: GM, 
                                 eoemin_index=0, 
                                 jzojc_index=1, 
                                 jpojc_index=2):
        base_nc = model.n_components
        dim = X.shape[1]
        N = len(X)

        # ---------- 1. histogram ----------
        eoemin = X[:, eoemin_index]
        jzojc = X[:, jzojc_index]

        wid0 = hist_bin_fd(eoemin)
        wid1 = hist_bin_fd(jzojc)

        bin_number0 = min(int(np.ptp(eoemin) / wid0), 150)
        bin_number1 = min(int(np.ptp(jzojc) / wid1), 300)

        x_range = [np.percentile(eoemin, 0.1), np.percentile(eoemin, 99.9)]
        y_range = [np.percentile(jzojc, 0.1), np.percentile(jzojc, 99.9)]

        true_prob, x_edges, y_edges = np.histogram2d(
            eoemin, 
            jzojc,
            bins=[bin_number0, bin_number1],
            density=True,
            range=[x_range, y_range]
        )
        # ---------- 2. grid & model ----------
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')

        grid_points = np.column_stack([xx.ravel(order='C'), yy.ravel(order='C')])

        model_prob = np.exp(model.score_samples(grid_points)).reshape(xx.shape)
        
        dx = x_edges[1] - x_edges[0]
        dy = y_edges[1] - y_edges[0]
        bin_area = dx * dy

        true_count  = true_prob  * N * bin_area
        model_count = model_prob * N * bin_area

        # ---------- 3. ΔL ----------
        epsilon = 1
        ratio = (true_count + epsilon) / (model_count + epsilon)
        delta_L = true_count * np.log(ratio) - (true_count - model_count)
        delta_L = np.clip(delta_L, 0, None)
        delta_L = np.nan_to_num(delta_L)

        # ---------- 4. threshold ----------
        delta_L_positive = delta_L[delta_L > 0]
        q1, q3 = np.percentile(delta_L_positive, [25, 75])
        iqr = q3 - q1
        mask = delta_L > (q3 + 1.5 * iqr)
        labels, _ = label(mask)

        # ---------- 5. map data → region ----------
        ix = np.digitize(eoemin, x_edges) - 1
        iy = np.digitize(jzojc, y_edges) - 1

        valid = (
            (ix >= 0) & (ix < len(x_edges) - 1) &
            (iy >= 0) & (iy < len(y_edges) - 1)
        )

        point_labels = np.zeros(N, dtype=int)
        point_labels[valid] = labels[ix[valid], iy[valid]]
        
        # ---------- 6. select valid regions ----------
        k = 0.5*dim*(dim+1) + dim + 1
        penalty = k * np.log(len(X))

        region_ids = np.unique(labels)
        region_ids = region_ids[region_ids != 0]

        gains = [
            (2*delta_L[labels == lbl].sum() - penalty)/np.sum(point_labels>0)
            for lbl in region_ids
        ]

        gains = np.asarray(gains)
        sorted_idx = np.argsort(gains)[::-1]
        sorted_gains = gains[sorted_idx]
        sorted_labels = region_ids[sorted_idx]

        positive_mask = sorted_gains > 0
        positive_gains = sorted_gains[positive_mask]
        positive_labels = sorted_labels[positive_mask]

        positive_gains = positive_gains[:MAX_N_COMPONENTS-base_nc]
        positive_labels = positive_labels[:MAX_N_COMPONENTS-base_nc]
        if len(positive_gains) > 0:
            cum_ratio = np.exp(np.cumsum(positive_gains) - np.sum(positive_gains))
            n_selected = np.searchsorted(cum_ratio, BF, side='right')
            n_selected = min(n_selected, len(positive_labels))
            q1, q3 = np.percentile(positive_gains, [25, 75])
            iqr = q3 - q1
            if n_selected==0 and positive_gains[0] > q3 + 1.5 * iqr:
                n_selected = 1
            selected_labels = positive_labels[:n_selected]
        else:
            selected_labels = np.array([], dtype=int)

        model_3d = self._dimensional_ascension(X, model)
        
        weights     = list(model_3d[0])
        means       = list(model_3d[1])
        covariances = list(model_3d[2])
        precisions  = list(model_3d[3])
        
        for lbl in selected_labels:
            pts = X[point_labels == lbl]
            weight = len(pts) / N

            if (len(pts) < MIN_POINTS or weight < MIN_WEIGHT):
                continue

            mean = pts.mean(axis=0)
            
            too_close = False
            for mu, prec in zip(means, precisions):
                diff = mean - mu
                mahal = np.sqrt(diff @ prec @ diff)
                if mahal < MIN_MAHAL:
                    too_close = True
                    break
            if too_close:
                continue
            
            cov = np.cov(pts, rowvar=False)
            cov += 1e-6 * np.eye(dim)

            prec = np.linalg.inv(cov)

            weights.append(weight)
            means.append(mean)
            covariances.append(cov)
            precisions.append(prec)

        weights = np.asarray(weights)
        weights /= weights.sum()

        means       = np.asarray(means)
        covariances = np.asarray(covariances)
        precisions  = np.asarray(precisions)

        self.initial_model = GM(len(weights), 
                                weights_init=weights, 
                                means_init=means, 
                                precisions_init=precisions, 
                                max_iter=0,
                                min_iter=0).fit(X)
        return self.initial_model, delta_L
    
    def fit(self, X, eoemin_cut, jzojc_cut, r_jzojc_cut, initial_use=[0,1], sample_weight=None,  disable_zero_rotation='halo', scaler=None, **kwargs): 
        self.morphology_model = self._morphology_class(X=X[:,initial_use], jzojc_cut=jzojc_cut)
        self.initial_model = self._initialize(X[:,initial_use], self.morphology_model, 
                                              eoemin_cut=eoemin_cut,
                                              jzojc_cut=jzojc_cut,
                                              r_jzojc_cut=r_jzojc_cut,
                                              disable_zero_rotation=disable_zero_rotation,
                                              scaler=scaler)
        data_to_train = X[:,initial_use] if self.morphology_type == 'spheroid' else X
        self.initial_model, delta_L = self._find_residual_component(data_to_train, model=self.initial_model)
        params = {
            'n_components': self.initial_model.n_components,
            'weights_init': self.initial_model.weights_,
            'means_init': self.initial_model.means_,
            'precisions_init': self.initial_model.precisions_,
            'max_iter': 200,
            'min_iter': 50
        }
        params.update(kwargs)
        self.best_model = GM(**params).fit(data_to_train, sample_weight)

        return self
    
    def _dimensional_ascension(self, X, model):
        N, dim = X.shape
        if dim == model.means_.shape[1]:
            return model.weights_, model.means_, model.covariances_, model.precisions_
        
        weights = model.weights_
        K = len(weights)

        means_3d = np.zeros((K, dim))
        covariances_3d  = np.zeros((K, dim, dim))
        labels = model.soft_predict(X[:, :2])
        
        for k in range(K):
            mask = (labels == k)
            pts = X[mask]
            means_3d[k] = pts.mean(axis=0)
            covariances_3d[k] = np.cov(pts, rowvar=False) + 1e-6 * np.eye(dim)
        means_3d[:,[0,1]] = model.means_
        precisions_3d = np.linalg.inv(covariances_3d)
        return weights, means_3d, covariances_3d, precisions_3d
