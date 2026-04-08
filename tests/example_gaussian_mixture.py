from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#import sys
#sys.path.insert(0, '../..')
from kinematic_decompose.mixture import GaussianMixture

def generate_data(n_samples, n_features, weights, means, precisions, covariance_type, dtype=np.float64):
    rng = np.random.RandomState(0)

    X = []
    if covariance_type == "spherical":
        for _, (w, m, c) in enumerate(zip(weights, means, precisions["spherical"])):
            X.append(
                rng.multivariate_normal(
                    m, c * np.eye(n_features), int(np.round(w * n_samples))
                ).astype(dtype)
            )
    if covariance_type == "diag":
        for _, (w, m, c) in enumerate(zip(weights, means, precisions["diag"])):
            X.append(
                rng.multivariate_normal(
                    m, np.diag(c), int(np.round(w * n_samples))
                ).astype(dtype)
            )
    if covariance_type == "tied":
        for _, (w, m) in enumerate(zip(weights, means)):
            X.append(
                rng.multivariate_normal(
                    m, precisions["tied"], int(np.round(w * n_samples))
                ).astype(dtype)
            )
    if covariance_type == "full":
        for _, (w, m, c) in enumerate(zip(weights, means, precisions["full"])):
            X.append(
                rng.multivariate_normal(m, c, int(np.round(w * n_samples))).astype(
                    dtype
                )
            )

    X = np.vstack(X)
    return X

def test_mini_batch():
    n_samples = 50000
    n_features = 2
    weights = [0.6, 0.4]
    means = [[-3, -3], [3, 3]]
    covariance_type = "full"
    precisions = {
        covariance_type: [
            [[0.5, 0], [0, 0.5]],
            [[1, 0.3], [0.3, 0.8]]
        ]
    }
    X = generate_data(n_samples, n_features, weights, means, precisions, covariance_type)
    
    start = time()
    GMM = GaussianMixture(n_components=2, init_params='random', batch_size=10240, min_iter=1000).fit(X, use_mini_batch=False)
    end = time()
    labels = GMM.predict(X)
    fig,axes = plt.subplots(2,2)
    for label in np.unique(labels):
        axes[0,0].scatter(X[labels==label,0], X[labels==label,1])
    axes[0,1].plot(GMM.lower_bounds_, label=f'full batch time={end-start:.2f}')
    axes[0,1].legend()

    start = time()
    GMM = GaussianMixture(n_components=2, init_params='random', batch_size=10240, min_iter=1000).fit(X, use_mini_batch=True)
    end = time()
    labels = GMM.predict(X) 
    for label in np.unique(labels):
        axes[1,0].scatter(X[labels==label,0], X[labels==label,1])
    axes[1,1].plot(GMM.lower_bounds_, label=f'mini batch time={end-start:.2f}')
    axes[1,1].legend()
    plt.show()

def test_sample_weight():
    n_samples = 10000
    n_features = 2
    weights = [0.6, 0.4]
    means = [[-2, -2], [2, 2]]
    covariance_type = "full"
    precisions = {
        covariance_type: [
            [[0.5, 0], [0, 0.5]],
            [[1, 0.3], [0.3, 0.8]]
        ]
    }
    X = generate_data(n_samples, n_features, weights, means, precisions, covariance_type)
    
    n_quantiles_x = 50
    n_quantiles_y = 50

    x_sorted = np.sort(X[:, 0])
    y_sorted = np.sort(X[:, 1])

    x_edges = x_sorted[np.linspace(0, len(x_sorted)-1, n_quantiles_x+1).astype(int)]
    y_edges = y_sorted[np.linspace(0, len(y_sorted)-1, n_quantiles_y+1).astype(int)]

    hist, x_edges, y_edges = np.histogram2d(
        X[:, 0], X[:, 1], 
        bins=[x_edges, y_edges]
    )
    sample_weight = hist.T.ravel()
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    fig,axes = plt.subplots(2,2)

    GMM = GaussianMixture(n_components=2, init_params='random', batch_size=10240, max_iter=1000, min_iter=1000).fit(X, use_mini_batch=True)
    labels = GMM.predict(X)
    
    for label in np.unique(labels):
        axes[0,0].scatter(X[labels==label,0], X[labels==label,1])
        mean = GMM.means_[label]
        cov  = GMM.covariances_[label]

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0)

        widths = 2 * np.sqrt(2) * np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        ellipse = patches.Ellipse(
            xy=mean,
            width=widths[0],
            height=widths[1],
            angle=angle,
            edgecolor='k',
            facecolor='none',
            linewidth=3
        )

        axes[0,0].add_patch(ellipse)
    axes[0,1].plot(GMM.lower_bounds_[1:], label=f'particle')
    axes[0,1].set_xscale('log')
    axes[0,1].legend()
    GMM = GaussianMixture(n_components=2, init_params='random', batch_size=10240, max_iter=1000, min_iter=1000).fit(points, use_mini_batch=True, sample_weight=sample_weight)
    labels = GMM.predict(X) 
    for label in np.unique(labels):
        axes[1,0].scatter(X[labels==label,0], X[labels==label,1])
        mean = GMM.means_[label]
        cov  = GMM.covariances_[label]

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0)

        widths = 2 * np.sqrt(2) * np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        ellipse = patches.Ellipse(
            xy=mean,
            width=widths[0],
            height=widths[1],
            angle=angle,
            edgecolor='k',
            facecolor='none',
            linewidth=3
        )

        axes[1,0].add_patch(ellipse)
    axes[1,1].plot(GMM.lower_bounds_[1:], label=f'density')
    axes[1,1].legend()
    axes[1,1].set_xscale('log')
    plt.show()

def test_initialize():
    n_samples = 1000
    n_features = 2
    weights = [0.6, 0.4]
    means = [[-2, -2], [2, 2]]
    covariance_type = "full"
    precisions = {
        covariance_type: [
            [[0.5, 0], [0, 0.5]],
            [[1, 0.3], [0.3, 0.8]]
        ]
    }
    X = generate_data(n_samples, n_features, weights, means, precisions, covariance_type)
    GMM = GaussianMixture(n_components=2, weights_init=weights, means_init=means, precisions_init=precisions['full'], max_iter=100, min_iter=100)
    GMM.fit(X)
    labels = GMM.predict(X) 
    fig,axes = plt.subplots(1,2)
    for label in np.unique(labels):
        axes[0].scatter(X[labels==label,0], X[labels==label,1])
        mean = GMM.means_[label]
        cov  = GMM.covariances_[label]

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0)

        widths = 2 * np.sqrt(2) * np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        ellipse = patches.Ellipse(
            xy=mean,
            width=widths[0],
            height=widths[1],
            angle=angle,
            edgecolor='k',
            facecolor='none',
            linewidth=3
        )

        axes[0].add_patch(ellipse)
    axes[1].plot(GMM.lower_bounds_[1:], label=f'density')
    axes[1].legend()
    axes[1].set_xscale('log')
    plt.show()

def test_warm_start():
    fig,axes = plt.subplots(1,2)
    n_samples = 1000
    n_features = 2
    weights = [0.6, 0.4]
    means = [[-2, -2], [2, 2]]
    covariance_type = "full"
    precisions = {
        covariance_type: [
            [[0.5, 0], [0, 0.5]],
            [[1, 0.3], [0.3, 0.8]]
        ]
    }
    X = generate_data(n_samples, n_features, weights, means, precisions, covariance_type)
    axes[0].scatter(X[:, 0], X[:,1])
    iter = 25
    ini_iter = 0
    GMM = GaussianMixture(n_components=2, init_params="random", max_iter=iter, min_iter=iter, warm_start=True)
    for _ in range(2): 
        GMM.fit(X)
        labels = GMM.predict(X) 
        for label in np.unique(labels):
            mean = GMM.means_[label]
            cov  = GMM.covariances_[label]

            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 0)

            widths = 1 * np.sqrt(2) * np.sqrt(eigvals)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

            ellipse = patches.Ellipse(
                xy=mean,
                width=widths[0],
                height=widths[1],
                angle=angle,
                edgecolor='k',
                facecolor='none',
                linewidth=1.5
            )

            axes[0].add_patch(ellipse)
        axes[1].plot(np.arange(ini_iter+1,ini_iter+iter+1,1), GMM.lower_bounds_) 
        ini_iter += iter
    plt.show()

def test_add_component():
    fig,axes = plt.subplots(1,2)
    n_samples = 1000
    n_features = 2
    weights = [0.6, 0.4]
    means = [[-2, -2], [2, 2]]
    covariance_type = "full"
    precisions = {
        covariance_type: [
            [[0.5, 0], [0, 0.5]],
            [[1, 0.3], [0.3, 0.8]]
        ]
    }
    X = generate_data(n_samples, n_features, weights, means, precisions, covariance_type)
    axes[0].scatter(X[:, 0], X[:,1])
    iter = 25
    ini_iter = 0
    GMM = GaussianMixture(n_components=1, init_params="random", max_iter=iter, min_iter=iter, warm_start=True)
    for _ in range(2): 
        GMM.fit(X)
        labels = GMM.predict(X) 
        for label in np.unique(labels):
            mean = GMM.means_[label]
            cov  = GMM.covariances_[label]

            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 0)

            widths = 1 * np.sqrt(2) * np.sqrt(eigvals)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

            ellipse = patches.Ellipse(
                xy=mean,
                width=widths[0],
                height=widths[1],
                angle=angle,
                edgecolor='k',
                facecolor='none',
                linewidth=1.5
            )

            axes[0].add_patch(ellipse)
        axes[1].plot(np.arange(ini_iter+1,ini_iter+iter+1,1), GMM.lower_bounds_) 
        ini_iter += iter
        GMM.means_ = np.vstack([GMM.means_, means[1]])
        new_cov = np.linalg.inv(precisions['full'][1])
        GMM.covariances_ = np.concatenate(
            [GMM.covariances_, new_cov[None, :, :]],
            axis=0
        )
        GMM.weights_ = np.append(GMM.weights_, weights[1])
        GMM.weights_ /= GMM.weights_.sum()
        GMM.n_components += 1
        from kinematic_decompose.mixture._gaussian_mixture import _compute_precision_cholesky
        GMM.precisions_cholesky_ = _compute_precision_cholesky(
            GMM.covariances_, GMM.covariance_type
        )
    plt.show()

def test_initialize_sample_weight():
    n_samples = 100000
    n_features = 2
    weights = [0.6, 0.4]
    means = [[-2, -2], [2, 2]]
    covariance_type = "full"
    precisions = {
        covariance_type: [
            [[0.5, 0], [0, 0.5]],
            [[1, 0.3], [0.3, 0.1]]
        ]
    }
    X = generate_data(n_samples, n_features, weights, means, precisions, covariance_type)
    
    bins=100 

    from scipy.stats import binned_statistic_2d
    hist, x_edges, y_edges, binnumber = binned_statistic_2d(
        X[:, 0], X[:, 1], 
        values=None,
        statistic='count',
        bins=bins,
        expand_binnumbers=False
    )

    sample_weight = hist.ravel(order='C')
    
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
    points = np.column_stack([xx.ravel(order='C'), yy.ravel(order='C')])
    points = points[sample_weight > 0]
    sample_weight = sample_weight[sample_weight > 0]
    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=2).fit(X)
    plt.pcolormesh(xx, yy, hist, cmap='hot', shading='auto')
    plt.colorbar()
    #plt.scatter(X[:,0], X[:,1], s=1, alpha=1000/n_samples) 
    plt.scatter(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1])
    kmeans_model = KMeans(n_clusters=2).fit(points, sample_weight)
    plt.scatter(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1])
    plt.show()

if __name__ == "__main__":
    test_mini_batch()
    test_initialize_sample_weight()
    test_sample_weight()
    test_initialize()
    test_warm_start()
    test_add_component()
