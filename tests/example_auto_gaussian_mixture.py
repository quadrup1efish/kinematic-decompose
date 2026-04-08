import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from kinematic_decompose.mixture import AutoGMM
from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.gravity.kinematic_solver import calculate_kinematic_param
from kinematic_decompose.visualize import visualize_phase_space, visualize_residual

run = 'TNG50-1' 
basePath = f"/Users/yuwa/sims.TNG/{run}/output"
subID = 307486
snapNum = 99
snapshot = Snapshot(basePath, snapNum)
load_particle_fields = 'default'
snapshot.load_particle(ID = subID, load_particle_fields=load_particle_fields)
snapshot.physical_units()
snapshot.center()
snapshot.faceon(align_with='star', as_context=False)

galaxy = calculate_kinematic_param(snapshot)
auto_gmm = AutoGMM(galaxy)

model = auto_gmm._morphology_class()
residual, ncs, weights, means, covariances, precisions = auto_gmm._find_residual_component(auto_gmm.X_train, model)
if ncs > 3:
    visualize_residual(residual, means[3:], covariances[3:], extent=[auto_gmm.X_train[:,1].min(), auto_gmm.X_train[:,1].max(), auto_gmm.X_train[:,0].min(), auto_gmm.X_train[:,0].max()])
auto_gmm = auto_gmm.fit(max_iter=100)
model = auto_gmm.best_model
visualize_phase_space(auto_gmm.X, means=model.means_, covariances=model.covariances_)
plt.show()
