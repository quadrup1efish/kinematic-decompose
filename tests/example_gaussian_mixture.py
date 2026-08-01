from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#import sys
#sys.path.insert(0, '../..')
from kinematic_decompose.mixture import GaussianMixture
from kinematic_decompose.visualize import NATURE_STYLE

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

def test_speed():
    n_samples = 100000
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
    GMM_full = GaussianMixture(n_components=2, init_params='random', batch_size=10240, min_iter=100).fit(X, use_mini_batch=False)
    end = time()
    full_time = end - start
    
    start = time()
    GMM_mini = GaussianMixture(n_components=2, init_params='random', batch_size=10240, min_iter=100).fit(X, use_mini_batch=True)
    end = time()
    mini_time = end - start
    
    n_iter_full = len(GMM_full.lower_bounds_)
    time_per_iter_full = full_time / n_iter_full
    time_axis_full = np.arange(n_iter_full) * time_per_iter_full
    
    n_iter_mini = len(GMM_mini.lower_bounds_)
    time_per_iter_mini = mini_time / n_iter_mini
    time_axis_mini = np.arange(n_iter_mini) * time_per_iter_mini

    plt.figure(figsize=(10, 6))
    plt.plot(time_axis_full, GMM_full.lower_bounds_, 'b-', linewidth=2, label=f'Full Batch ({full_time:.2f}s)')
    plt.plot(time_axis_mini, GMM_mini.lower_bounds_, 'r-', linewidth=2, label=f'Mini Batch ({mini_time:.2f}s)')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Lower Bound', fontsize=12)
    plt.title('Convergence Comparison: Full Batch vs Mini Batch', fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


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

def test_scaling_with_n_samples():
    """Average per-iteration fit time vs n_samples for full vs mini batch.

    Fixed 10 EM iterations (min_iter = max_iter = 10), n_samples on a
    log-space grid of 10 points from 1e4 to 1e6 (K=2, d=2). Reports the
    median time of ONE iteration (total / 10) over 10 repeated mini-batch
    fits per N, with 16%-84% (1-sigma) spread bands for stability, plus
    the old pre-optimisation mini path (full-dataset init + full
    permutation, O(N) per fit) as a reference. The right y-axis shows the
    relative lower-bound error of mini vs full batch (|dLB|/|LB_full|);
    the full/mini curves share the same synthetic data per N.
    batch_size=1024 so every point really uses the mini-batch path
    (3*batch_size = 3072 < 1e4). The figure is saved to
    image/scaling_performance.png (used in the README).
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning

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

    n_iters = 10  # fixed iteration count: min_iter = max_iter = n_iters
    n_repeats = 10  # mini-batch repeats per N: median + 16-84% spread
    n_list = np.logspace(4, 6, 10).astype(int)  # 1e4 .. 1e6, 10 points
    per_iter_full, per_iter_mini_old = [], []
    per_iter_mini_all = []  # (n_points, n_repeats) after collection
    lb_full, lb_mini_all = [], []  # lb_mini_all: (n_points, n_repeats)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for n in n_list:
            X = generate_data(n, n_features, weights, means, precisions, covariance_type)

            t0 = time()
            m_full = GaussianMixture(n_components=2, init_params='random', batch_size=1024,
                                     min_iter=n_iters, max_iter=n_iters, tol=1e-3).fit(X, use_mini_batch=False)
            per_iter_full.append((time() - t0) / n_iters)
            lb_full.append(m_full.lower_bound_)

            # repeated mini-batch fits: median + 16-84% spread per N
            mini_times, mini_lb = [], []
            for _ in range(n_repeats):
                t0 = time()
                m_mini = GaussianMixture(n_components=2, init_params='random', batch_size=1024,
                                         min_iter=n_iters, max_iter=n_iters, tol=1e-3).fit(X, use_mini_batch=True)
                mini_times.append((time() - t0) / n_iters)
                mini_lb.append(m_mini.lower_bound_)
            per_iter_mini_all.append(mini_times)
            lb_mini_all.append(mini_lb)

            # old path: full-dataset init + no-replacement permutation
            # (disabling both optimizations reproduces the pre-optimization cost)
            t0 = time()
            GaussianMixture(n_components=2, init_params='random', batch_size=1024,
                            min_iter=n_iters, max_iter=n_iters, tol=1e-3).fit(
                X, use_mini_batch=True, _use_bounded_init=False,
                _replace_sampling=False)
            per_iter_mini_old.append((time() - t0) / n_iters)

            print(f"  N={n:>8d}: full {per_iter_full[-1] * 1e3:7.1f} ms/iter | "
                  f"mini(med) {np.median(mini_times) * 1e3:7.2f} ms/iter "
                  f"[{np.percentile(mini_times, 16) * 1e3:.2f}-{np.percentile(mini_times, 84) * 1e3:.2f}] | "
                  f"mini(old) {per_iter_mini_old[-1] * 1e3:7.1f} ms/iter")

    per_iter_full = np.array(per_iter_full)
    per_iter_mini = np.array(per_iter_mini_all)  # (n_points, n_repeats)
    per_iter_mini_med = np.median(per_iter_mini, axis=1)
    per_iter_mini_lo = np.percentile(per_iter_mini, 16, axis=1)
    per_iter_mini_hi = np.percentile(per_iter_mini, 84, axis=1)
    per_iter_mini_old = np.array(per_iter_mini_old)
    lb_full = np.array(lb_full)
    lb_mini_all = np.array(lb_mini_all)  # (n_points, n_repeats)

    # relative lower-bound error per repeat, then median + 16-84% spread
    lb_err_all = 100.0 * np.abs(lb_full[:, None] - lb_mini_all) / np.abs(lb_full[:, None])
    lb_err_med = np.median(lb_err_all, axis=1)
    lb_err_lo = np.percentile(lb_err_all, 16, axis=1)
    lb_err_hi = np.percentile(lb_err_all, 84, axis=1)

    # per-iteration cost should scale up with N for full batch
    assert per_iter_full[-1] > per_iter_full[0], "full per-iter should grow with N"
    # at large N, one mini-batch iteration must cost less than one full one
    assert per_iter_mini_med[-1] < per_iter_full[-1], "mini per-iter should beat full at largest N"
    # the bounded-subsample init must not be slower than the full-dataset init
    # at the largest N (it removes the O(N) init scan)
    assert per_iter_mini_med[-1] <= per_iter_mini_old[-1], "bounded init should not be slower"

    # ---- figure: unified Nature-journal style (shared with visualize.py) ----
    # Font sizes are bumped up relative to NATURE_STYLE so they match the
    # large 9x6 canvas (labels stand out instead of being dwarfed).
    with plt.rc_context({
        **NATURE_STYLE,
        'font.size': 16,
        'axes.labelsize': 19,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 14,
    }):
        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax1.loglog(n_list, per_iter_full, 'b-o', linewidth=2, label='Full batch')
        ax1.loglog(n_list, per_iter_mini_med, 'r-o', linewidth=2, label='Mini batch')
        ax1.fill_between(n_list, per_iter_mini_lo, per_iter_mini_hi,
                         color='r', alpha=0.2, label='Mini batch (16–84%)')
        ax1.loglog(n_list, per_iter_mini_old, color='r', linestyle='--', marker='s', linewidth=1.5, alpha=0.7,
                   label='Mini batch (old)')
        ax1.set_xlabel('N')
        ax1.set_ylabel('Time per iteration (s)')
        ax1.legend(loc='upper left', framealpha=0.9)

        # right axis: relative lower-bound error of mini batch vs full batch
        # (mini batch sees fewer data per iteration, so its converged LB may
        # deviate from the full-batch one; |dLB|/|LB_full| quantifies it).
        # Shown in percent with a fixed [0, 1%] ylim so the (tiny) errors
        # are read at their true magnitude instead of looking inflated.
        ax2 = ax1.twinx()
        ax2.semilogx(n_list, lb_err_med, 'g-^', linewidth=1.5, alpha=0.8, label='LB error (%)')
        ax2.fill_between(n_list, lb_err_lo, lb_err_hi, color='g', alpha=0.15,
                         label='LB error (16–84%)')
        ax2.set_ylabel('Relative LB error (%)', color='g')
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.legend(loc='upper right', framealpha=0.9)
        ax2.grid(False)

        fig.tight_layout()
        fig.savefig("image/scaling_performance.png", dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    test_speed()
    test_scaling_with_n_samples()
    #test_mini_batch()
    #test_initialize_sample_weight()
    #test_sample_weight()
    #test_initialize()
    #test_warm_start()
    #test_add_component()
