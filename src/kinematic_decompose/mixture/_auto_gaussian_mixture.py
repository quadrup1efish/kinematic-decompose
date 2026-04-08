import numpy as np
from scipy.ndimage import label

from .util import *
from ._gaussian_mixture import GaussianMixture as GM
from .preprocessing import RobustScaler


MIN_WEIGHT = 0.01
MAX_N_COMPONENTS = 15
MIN_POINTS = 10
CBIC = -0.1
BF = 0.95

class AutoGMM():
    def __init__(self, galaxy, n_components=None, model=None, morphology_type=None, dim=None):
        self.galaxy = galaxy
        self.n_components = n_components
        self.morphology_type = morphology_type
        self.ecut = None
        self.etacut=0.5
        self.X_full = np.column_stack((galaxy.s['eoemin'],galaxy.s['jzojc'],galaxy.s['jpojc'])).astype(np.float32, copy=False)
        del galaxy.ancestor['eoemin'], galaxy.ancestor['jzojc'], galaxy.ancestor['jpojc']
        if dim == 2:
            self.X = self.X_full[:, :2]
        else:
            self.X = self.X_full
        self.remove_particles = (self.X_full[:,0]<0)&(np.abs(self.X_full[:,1])<1.5)&(self.X_full[:,2]<1.5)
        self.X_clean= self.X[self.remove_particles]
        self.scaler = RobustScaler().fit(self.X_clean)
        self.X_train= self.scaler.transform(self.X_clean)

    def _morphology_class(self):
        model = GM(n_components=3, n_init=1, init_params='kmeans', max_iter=200, min_iter=50, tol=1e-3).fit(self.X_train[:,[0,1]])
        if np.max(model.means_[:,1]) > self._scale(0.5, self.X_clean[:,1]):
            self.morphology_type='disk'
        else:
            self.morphology_type='spheroid'
            model = GM(n_components=2, n_init=1, init_params='kmeans', max_iter=200, min_iter=10, tol=1e-3).fit(self.X_train[:,[0,1]])
            if self.X.shape[1] == 3:
                self.X = self.X_full[:, :2]
                self.X_train = self.X_train[:,[0,1]]
        self.morphology_model = model
        return model
    
    def _scale(self, x, X):
        x_scale = (x-np.nanmedian(X))/(np.nanpercentile(X,75)-np.nanpercentile(X,25))
        return x_scale
    
    def _find_residual_component(self, X, model: GM):
        base_nc = model.n_components
        dim = X.shape[1]
        N = len(X)

        # ---------- 1. histogram ----------
        wid0 = hist_bin_fd(X[:, 0])
        wid1 = hist_bin_fd(X[:, 1])

        bin_number0 = min(int(np.ptp(X[:, 0]) / wid0), 150)
        bin_number1 = min(int(np.ptp(X[:, 1]) / wid1), 300)

        true_prob, x_edges, y_edges = np.histogram2d(
            X[:, 0], X[:, 1],
            bins=[bin_number0, bin_number1],
            density=True, range=[[np.percentile(X[:, 0],0.1), np.percentile(X[:, 0],99.9)], [np.percentile(X[:, 1],0.1), np.percentile(X[:, 1],99.9)]]
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
        delta_L = true_count * np.log(true_count / model_count) - (true_count - model_count)
        delta_L = np.nan_to_num(delta_L)

        # ---------- 4. threshold ----------
        q1, q3 = np.percentile(delta_L, [25, 75])
        iqr = q3 - q1
        mask = delta_L > (q3 + 1.5 * iqr)

        labels, _ = label(mask)

        # ---------- 5. map data → region ----------
        ix = np.digitize(X[:, 0], x_edges) - 1
        iy = np.digitize(X[:, 1], y_edges) - 1

        valid = (
            (ix >= 0) & (ix < len(x_edges) - 1) &
            (iy >= 0) & (iy < len(y_edges) - 1)
        )

        point_labels = np.zeros(N, dtype=int)
        point_labels[valid] = labels[ix[valid], iy[valid]]
        
        # ---------- 6. select valid regions ----------
        total_delta_L = delta_L[mask].sum()
        k = 0.5*dim*(dim+1) + dim + 1
        penalty = k * np.log(len(X))

        region_ids = np.unique(labels)
        region_ids = region_ids[region_ids != 0]

        gains = [
            (2*delta_L[labels == lbl].sum() - penalty)/np.sum(point_labels>0)
            for lbl in region_ids
        ]

        gains = np.asarray(gains)
        sorted_idx = np.argsort(gains)[::-1][:MAX_N_COMPONENTS-base_nc]
        sorted_gains = gains[sorted_idx]
        sorted_labels = region_ids[sorted_idx]
        valid_gains = sorted_gains[sorted_gains>0]
        valid_labels= sorted_labels[sorted_gains>0]
        cum_ratio = np.exp(np.cumsum(valid_gains) - np.sum(valid_gains))
        n_selected = np.searchsorted(cum_ratio, BF, side='left')
        n_selected = min(n_selected, len(valid_labels))
        valid_labels = sorted_labels[:n_selected]

        model_3d = self._dimensional_ascension(X, model)
        
        weights     = list(model_3d[0])
        means       = list(model_3d[1])
        covariances = list(model_3d[2])
        precisions  = list(model_3d[3])

        for lbl in valid_labels:
            pts = X[point_labels == lbl]
            weight = len(pts) / N

            if (len(pts) < MIN_POINTS or weight < MIN_WEIGHT):
                continue

            mean = pts.mean(axis=0)

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

        return delta_L, len(weights), weights, means, covariances, precisions
    
    def fit(self, **kwargs): 
        self.morphology_model = self._morphology_class()
        _, nc, weights, means, covariances, precisions = self._find_residual_component(self.X_train, model=self.morphology_model)
        params = {
            'n_components': nc,
            'weights_init': weights,
            'means_init': means,
            'precisions_init': precisions,
            'max_iter': 200,
            'min_iter': 50
        }
        params.update(kwargs)
        self.best_train_model = GM(**params).fit(self.X_train)
        self.best_model = self.scaler.inverse_transform_GMM(self.best_train_model)
        return self
    
    def _dimensional_ascension(self, X, model):
        N, dim = X.shape
        if dim == model.means_.shape[1]:
            return model.weights_, model.means_, model.covariances_, model.precisions_
        
        weights = model.weights_
        K = len(weights)

        means_3d = np.zeros((K, dim))
        covariances_3d  = np.zeros((K, dim, dim))

        for k in range(K):
            resp = model.predict_proba(X[:, :2])[:, k]
            mask = resp > 0.5
            pts = X[mask]
            means_3d[k] = pts.mean(axis=0)
            covariances_3d[k] = np.cov(pts, rowvar=False) + 1e-6 * np.eye(dim)

        precisions_3d = np.linalg.inv(covariances_3d)
        return weights, means_3d, covariances_3d, precisions_3d
    
    def decompose(self, pot=None):
        # Predict labels and probabilities
        labels = self.best_model.soft_predict(self.X)
        prob = self.best_model.predict_proba(self.X)
        
        # Extract GMM parameters
        weights, means, covariances = (
            self.best_model.weights_,
            self.best_model.means_,
            self.best_model.covariances_
        )
        
        # Set morphology threshold
        self.etacut = 0.7 if self.morphology_type == 'spheroid' else self.etacut
        self.ecut = get_energy_criterion(pot, self.galaxy)
        # Define classification mapping
        label_map = {}
        for i, (ec, eta) in enumerate(means[:,0:2]):
            if eta >= 0.85:
                label_map[i] = 0  # cold disk
            elif eta > self.etacut:
                label_map[i] = 1  # warm disk
            elif eta < -self.etacut:
                label_map[i] = 4  # counter-rotating disk
            elif eta <= self.etacut and ec < self.ecut:
                label_map[i] = 2  # bulge
            else:
                label_map[i] = 3  # halo
        
        new_labels = np.vectorize(label_map.get)(labels)
        new_prob = np.zeros((len(prob), 5), dtype=np.float32)
        for old_idx, new_idx in label_map.items():
            new_prob[:, new_idx] += prob[:, old_idx]
        
        # Store results in galaxy object
        self.galaxy.s['label'] = new_labels
        self.galaxy.s['prob'] = new_prob
        
        # Clean up and prepare GMM storage
        self.X_full = self.X = None
        del new_labels, labels, new_prob, prob
        
        # Build GMM dictionary with classifications
        mask = means[:, 1] > self.etacut
        GMM_dict = {
            "total": {"weights": weights, "means": means, "covariances": covariances},
            "disk": {"weights": weights[mask], 
                     "means": means[mask], 
                     "covariances": covariances[mask]},
            "colddisk": {"weights": weights[means[:, 1] >= 0.85], 
                         "means": means[means[:, 1] >= 0.85], 
                        "covariances": covariances[means[:, 1] >= 0.85]},
            "warmdisk": {"weights": weights[(means[:, 1] > self.etacut) & (means[:, 1] < 0.85)], 
                        "means": means[(means[:, 1] > self.etacut) & (means[:, 1] < 0.85)], 
                        "covariances": covariances[(means[:, 1] > self.etacut) & (means[:, 1] < 0.85)]},
            "counter-rotate": {"weights": weights[means[:, 1] < -self.etacut], 
                            "means": means[means[:, 1] < -self.etacut], 
                            "covariances": covariances[means[:, 1] < -self.etacut]},
            "spheroid": {"weights": weights[np.abs(means[:, 1]) <= self.etacut], 
                        "means": means[np.abs(means[:, 1]) <= self.etacut], 
                        "covariances": covariances[np.abs(means[:, 1]) <= self.etacut]},
            "bulge": {"weights": weights[(np.abs(means[:, 1]) <= self.etacut) & (means[:, 0] < self.ecut)], 
                    "means": means[(np.abs(means[:, 1]) <= self.etacut) & (means[:, 0] < self.ecut)], 
                    "covariances": covariances[(np.abs(means[:, 1]) <= self.etacut) & (means[:, 0] < self.ecut)]},
            "halo": {"weights": weights[(np.abs(means[:, 1]) <= self.etacut) & (means[:, 0] >= self.ecut)], 
                    "means": means[(np.abs(means[:, 1]) <= self.etacut) & (means[:, 0] >= self.ecut)], 
                    "covariances": covariances[(np.abs(means[:, 1]) <= self.etacut) & (means[:, 0] >= self.ecut)]},
            "ecut": self.ecut,
        }
        return self.galaxy, GMM_dict
