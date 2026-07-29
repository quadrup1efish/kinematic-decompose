import agama
from time import time
import numpy as np
import matplotlib.pyplot as plt
from pynbody import units, gravity
from kinematic_decompose.mixture import *
from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.gravity.kinematic_solver import construct_galaxy_potential_model

run = 'TNG100-3'
basePath = f"/Users/yuwa/sims.TNG/{run}/output"
subID = 5
snapNum = 99

snapshot = Snapshot(basePath, snapNum)
snapshot.load_particle(ID=subID, load_particle_fields="potential")
snapshot.physical_units()
snapshot.load_group_catalog(ID=subID)
snapshot.GC_physical_units()
snapshot.center(cen=snapshot.group_catalog['SubhaloPos'])
galaxy = snapshot.container

# Method 1: Multipole expansion
start = time()
pot = construct_galaxy_potential_model(galaxy)
multipole_potential = pot.potential(galaxy['pos'])
t1 = time() - start
print(f"Multipole time = {t1:.3f}s")

# Method 2: Direct N-body summation
units.G = 4.30091e-6 * units.Unit('kpc Msol**-1 km**2 s**-2')
start = time()
eps = np.repeat(galaxy.properties['eps'], len(galaxy['mass']))
phi, accel = gravity.direct(galaxy, galaxy['pos'].view(np.ndarray), eps)
phi = phi.in_units('km**2 s**-2')
t2 = time() - start
print(f"Direct N-body time = {t2:.3f}s")

# Ensure consistent units
if not hasattr(multipole_potential, 'units'):
    multipole_potential = multipole_potential * units.Unit('km**2 s**-2')
else:
    multipole_potential = multipole_potential.in_units('km**2 s**-2')

phi_comp = -phi
multipole_comp = -multipole_potential

# Plot
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

ax = axes[0]
ax.loglog(phi_comp, multipole_comp, '.', alpha=0.5, markersize=2)
ax.plot([phi_comp.min(), phi_comp.max()], 
        [phi_comp.min(), phi_comp.max()], 'r--', label='y=x')
ax.set_xlabel('-phi (direct) [km²/s²]')
ax.set_ylabel('-phi (multipole) [km²/s²]')
ax.set_title(f'Potential Comparison\nMultipole: {t1:.2f}s, Direct: {t2:.2f}s')
ax.legend()

ax = axes[1]
relative_error = (multipole_comp - phi_comp) / phi_comp
ax.semilogx(phi_comp, relative_error, '.', alpha=0.5, markersize=2)
ax.axhline(y=0, color='r', linestyle='--')
ax.axhline(y=0.1, color='gray', linestyle=':', alpha=0.75)
ax.axhline(y=-0.1, color='gray', linestyle=':', alpha=0.75)
ax.set_xlabel('-phi (direct) [km²/s²]')
ax.set_ylabel('Relative error')
ax.set_title('Relative Error (multipole - direct) / direct')
plt.tight_layout()
plt.show()
