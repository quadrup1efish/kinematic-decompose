import agama
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from kinematic_decompose.mixture import AutoGMM
from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.gravity.kinematic_solver import create_multipole_potential, calculate_kinematic_param
from kinematic_decompose.visualize import visualize_decomposition

run = 'TNG50-1' 
basePath = f"/Users/yuwa/sims.TNG/{run}/output"
subIDs = [307486]
snapNum = 99

data_dir = Path("data")
image_dir= Path("image")

for subID in subIDs:
    filename = data_dir / f"{subID}.ini"
    if not filename.exists():
        print("create multipole potential ...")
        snapshot = Snapshot(basePath, snapNum)
        load_particle_fields = 'potential'
        snapshot.load_particle(ID = subID, load_particle_fields=load_particle_fields)
        snapshot.physical_units()
        snapshot.load_group_catalog(ID=subID)
        snapshot.GC_physical_units()
        snapshot.center(cen=snapshot.group_catalog['SubhaloPos'])
        snapshot.faceon(align_with='star', range=[3*snapshot.properties['eps'], 5*snapshot.s.r50], as_context=False)
        galaxy = snapshot.container
        pot = create_multipole_potential(galaxy['pos'], galaxy['mass'])
        pot.export(str(filename))
    pot = agama.Potential(str(filename))
    snapshot = Snapshot(basePath, snapNum)
    load_particle_fields = {"star": ['Coordinates', 'Velocities', 'Masses']}
    snapshot.load_particle(ID = subID, load_particle_fields=load_particle_fields)
    snapshot.physical_units()
    snapshot.load_group_catalog(ID=subID)
    snapshot.GC_physical_units()
    snapshot.center(cen=snapshot.group_catalog['SubhaloPos'])
    snapshot.faceon(align_with='star', range=[3*snapshot.properties['eps'], 5*snapshot.s.r50], as_context=False)
    galaxy = snapshot.container
    galaxy = calculate_kinematic_param(galaxy, pot)
    auto_gmm = AutoGMM(galaxy)
    auto_gmm = auto_gmm.fit(max_iter=100)
    model = auto_gmm.best_model
    X = auto_gmm.X
    galaxy, _ = auto_gmm.decompose(pot)
    visualize_decomposition(X, auto_gmm, galaxy, threshold_line=True, ranges=None)
    plt.savefig(image_dir/f"{subID}.pdf")
    plt.show()
