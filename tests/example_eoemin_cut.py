import agama
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from kinematic_decompose.mixture import *
from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.gravity.kinematic_solver import create_multipole_potential, calculate_kinematic_param
from kinematic_decompose.visualize import visualize_decomposition

MAX_RADIUS = 10


run = 'TNG50-1' 
basePath = f"/Users/yuwa/sims.TNG/{run}/output"
subID = 307486
snapNum = 99
snapshot = Snapshot(basePath, snapNum)
load_particle_fields = 'default'
snapshot.load_particle(ID = subID, load_particle_fields=load_particle_fields)
snapshot.physical_units()
snapshot.load_group_catalog(ID=subID)
snapshot.GC_physical_units()
snapshot.center(cen=snapshot.group_catalog['SubhaloPos'])
snapshot.faceon(align_with='star', range=[3*snapshot.properties['eps'], 5*snapshot.s.r50], as_context=False)
galaxy = snapshot.container
data_dir = Path("data")
filename = data_dir / f"{subID}.ini"
pot = agama.Potential(str(filename))
galaxy = calculate_kinematic_param(snapshot, pot)

r_min = 2*galaxy.properties['eps']
r_max = 10
step  = 0.5*galaxy.properties['eps']
x = y = np.arange(r_min, r_max, step)
points = np.column_stack([x,y,np.zeros(len(x))])
r, potential, dpot_dr, d2pot_dr2, curvature = util.create_potential_profile(pot, points)
f_r = util.create_eoemin_profile_function(galaxy.s['eoemin'], galaxy.s['r'], r_min, r_max, bins=50, range=[-1.,0.], statistic='median')
ecut = util.get_energy_criterion(pot, galaxy.s['r'], galaxy.s['eoemin'], r_min, r_max, step, cut_ratio='auto')

fig, axes = plt.subplots(2, 1, gridspec_kw={'hspace': 0}, figsize=(6,5))
axes[0].plot(f_r(r), dpot_dr/dpot_dr.min(), label=r'$\partial_r\phi$', color='red', lw=1.5)
axes[0].plot(f_r(r), d2pot_dr2/d2pot_dr2.max(), label=r'$T_{rr}$', color='b', lw=1.5)
axes[0].plot(f_r(r), curvature/curvature.max(), label=r'$\kappa_r$', color='k', lw=1.5)
axes[0].set_xlim(-1,0)
axes[0].set_xticks([])
axes[0].set_ylabel('Normalized')
axes[0].axvline(ecut, linestyle=':', color='gray', lw=1.5)
ax2 = axes[0].twiny()
r_ticks = [0, 1, 3, 5, 10]  
ax2.set_xticks(r_ticks)
ax2.set_xlabel('r (kpc)')
ax2.set_xticklabels([f'{tick:.0f}' for tick in r_ticks])
axes[0].legend()

axes[1].hist(galaxy.s['eoemin'], bins=20, density=False, 
             weights=np.ones_like(galaxy.s['eoemin'])/len(galaxy.s['eoemin']),
             histtype='step', color='k', lw=2)
axes[1].axvline(ecut, linestyle=':', label='Energy threshold', color='gray',lw=1.5)
axes[1].set_xlim(-1,0)
axes[1].set_xlabel(r'$e/|e|_\mathrm{max}$')
axes[1].set_ylabel('Probability')
axes[1].legend()
plt.tight_layout()
plt.show()
