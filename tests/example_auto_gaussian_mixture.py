import agama
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from kinematic_decompose.mixture import *
from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.gravity.kinematic_solver import calculate_kinematic_param
from kinematic_decompose.visualize import visualize_phase_space, gaussian_ell

run = 'TNG50-1' 
basePath = f"/Users/yuwa/sims.TNG/{run}/output"
subID = 307486
snapNum = 99
snapshot = Snapshot(basePath, snapNum)
load_particle_fields = {"star": ['Coordinates', 'Velocities', 'Masses'],
                        "gas":  ['Coordinates', 'Masses'],
                        "dm":   ['Coordinates'],}
snapshot.load_particle(ID = subID, load_particle_fields=load_particle_fields)
snapshot.physical_units()
snapshot.load_group_catalog(ID = subID)
snapshot.GC_physical_units()
snapshot.center(snapshot.group_catalog['SubhaloPos'])
snapshot.faceon(align_with='star', as_context=False)

data_dir = Path("data")
filename = data_dir / f"{subID}.ini"
pot = agama.Potential(str(filename))
galaxy = calculate_kinematic_param(snapshot, None, partType='star')
X = np.column_stack([galaxy.s['eoemin'], galaxy.s['jzojc'], galaxy.s['jpojc']])

eoemin_index = 0
jzojc_index = 1
jpojc_index = 2

eps = galaxy.properties.get('eps', 0.5)
r_min = 2*eps
r_max = min(1.1*galaxy.s.r90, 10)
step = 0.5*eps
keep_particle = (galaxy.s['eoemin']<0)&(np.abs(galaxy.s['jzojc'])<1.5)&(galaxy.s['jpojc']<1.5)
sph, _ = util.JEHistogram(galaxy.s['eoemin'][keep_particle], galaxy.s['jzojc'][keep_particle], n_E=25, n_eps=50)
sph = (sph) & (np.abs(galaxy.s['jzojc'][keep_particle])<=0.5)
eoemin_cut= util.get_Ecut(galaxy.s['eoemin'][keep_particle][sph], galaxy.s['mass'][keep_particle][sph], M_bin=100, m_bin=25, Mmin=0.1)
jzojc_cut=0.5
scaler = preprocessing.RobustScaler()
X_train= scaler.fit_transform(X)

eoemin_cut_train = scaler.transform(eoemin_cut, columns=eoemin_index)
jzojc_cut_train = scaler.transform(jzojc_cut, columns=jzojc_index)
r_jzojc_cut_train = scaler.transform(-jzojc_cut, columns=jzojc_index)

auto_gmm = AutoGaussianMixtureModel()
auto_gmm = auto_gmm.fit(X_train, 
                        eoemin_cut=eoemin_cut_train, 
                        jzojc_cut=jzojc_cut_train,
                        r_jzojc_cut = r_jzojc_cut_train, 
                        sample_weight=None,
                        max_iter=100, 
                        min_iter=25,
                        scaler=scaler)

fig = plt.figure(figsize=(14, 6))

gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.6], hspace=0.3, wspace=0.35)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
ax_line = fig.add_subplot(gs[1, :])

bins = [min(int(np.ptp(X[:, 0]) / util.hist_bin_fd(X[:, 0])), 200), 
        min(int(np.ptp(X[:, 1]) / util.hist_bin_fd(X[:, 1])), 200)]
hist_params = {
    'bins': bins,
    'cmap': 'Spectral',
    'cmin': 1,
    'norm': LogNorm(),
}
dims = 2
proj = [1, 0]
percentile_low, percentile_high = 0.5, 99.5  
buffer_factor = 0.05
ranges = []
for i in range(dims):
    low = np.percentile(X[:, i], percentile_low)
    high = np.percentile(X[:, i], percentile_high)
    span = high - low
    low -= span * buffer_factor
    high += span * buffer_factor
    ranges.append([low, high])

titles = ['Morphology', 'Initial', 'Final']
models = [auto_gmm.morphology_model, auto_gmm.initial_model, auto_gmm.best_model]
for ax, model, title in zip(axes, models, titles):
    model = scaler.inverse_transform_GMM(model)
    im = ax.hist2d(X[:, proj[0]], X[:, proj[1]], 
                   range=[ranges[proj[0]], ranges[proj[1]]], **hist_params)
    for j, (mean, covariance) in enumerate(zip(model.means_, model.covariances_)):
        gaussian_ell(ax, mean[proj], covariance[np.ix_(proj, proj)], color='k')
    ax.set_title(f'{title}', color='k', fontsize=12)

ax_line.plot(auto_gmm._n_components_candidates, auto_gmm._cum_ratio)
ax_line.set_xlabel('Number of components')
ax_line.set_ylabel('BF')
ax_line.set_title('Model Selection Criterion')

plt.tight_layout()
plt.show()
