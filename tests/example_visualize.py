import matplotlib.pyplot as plt
from kinematic_decompose.mixture import AutoGMM
from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.gravity.kinematic_solver import create_multipole_potential, calculate_kinematic_param
from kinematic_decompose.visualize import visualize_decomposition

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
galaxy = snapshot.container

pot = create_multipole_potential(galaxy['pos'], galaxy['mass'])
galaxy = calculate_kinematic_param(galaxy, pot)
auto_gmm = AutoGMM(galaxy)
auto_gmm = auto_gmm.fit(max_iter=100)
X = auto_gmm.X
model = auto_gmm.best_model
galaxy, gmm_dict = auto_gmm.decompose(pot)
visualize_decomposition(X, auto_gmm, galaxy, ranges=None,threshold_line=True)
plt.show()
