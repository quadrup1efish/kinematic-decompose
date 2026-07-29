import sys
import pynbody
import numpy as np
import matplotlib.pyplot as plt

from kinematic_decompose.PyTNG.snapshot_loader import Snapshot

run = 'TNG100-3' 
basePath = f"/Users/yuwa/sims.TNG/{run}/output"
subID = 0#307486
snapNum = 99
snapshot = Snapshot(basePath, snapNum)
load_particle_fields = 'default'
snapshot.load_particle(ID = subID, groupType='Group', load_particle_fields=load_particle_fields)
snapshot.physical_units()
snapshot.center()
snapshot.faceon(align_with='star', as_context=False)

fig, axes = plt.subplots(1,3, figsize=(15,3))
pynbody.plot.image(snapshot.dm, axes=axes[0], width="10 Mpc", units="Msol kpc^-2", cmap="twilight", colorbar_label=r"$\Sigma_{\mathrm{gas}} / M_{\odot}\,\mathrm{kpc}^{-2}$")
pynbody.plot.image(snapshot.star, axes=axes[1], width="10 Mpc", units="Msol kpc^-2", cmap="bone", colorbar_label=r"$\Sigma_{\mathrm{gas}} / M_{\odot}\,\mathrm{kpc}^{-2}$")
pynbody.plot.image(snapshot.gas, axes=axes[2], width="10 Mpc", units="Msol kpc^-2", cmap="bone", colorbar_label=r"$\Sigma_{\mathrm{gas}} / M_{\odot}\,\mathrm{kpc}^{-2}$")
plt.tight_layout()
plt.show()
