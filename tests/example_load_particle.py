import sys
import pynbody
import numpy as np
import matplotlib.pyplot as plt

from kinematic_decompose.mixture import AutoGMM
from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.gravity.kinematic_solver import calculate_kinematic_param

run = 'TNG100-3' 
basePath = f"/Users/yuwa/sims.TNG/{run}/output"
subID = 1#307486
snapNum = 99
snapshot = Snapshot(basePath, snapNum)
load_particle_fields = 'default'
snapshot.load_particle(ID = subID, load_particle_fields=load_particle_fields)
snapshot.physical_units()
snapshot.center()
snapshot.faceon(align_with='star', as_context=False)

fig, axes = plt.subplots(1,3, figsize=(12,3))
pynbody.plot.image(snapshot.dm, axes=axes[0], width="1.5 Mpc", units="Msol kpc^-2", cmap="twilight", colorbar_label=r"$\Sigma_{\mathrm{gas}} / M_{\odot}\,\mathrm{kpc}^{-2}$")
pynbody.plot.image(snapshot.star, axes=axes[1], width="500 kpc", units="Msol kpc^-2", cmap="bone", colorbar_label=r"$\Sigma_{\mathrm{gas}} / M_{\odot}\,\mathrm{kpc}^{-2}$")
pynbody.plot.image(snapshot.gas, axes=axes[2], width="500 kpc", units="Msol kpc^-2", cmap="bone", colorbar_label=r"$\Sigma_{\mathrm{gas}} / M_{\odot}\,\mathrm{kpc}^{-2}$")
plt.tight_layout()
plt.show()
