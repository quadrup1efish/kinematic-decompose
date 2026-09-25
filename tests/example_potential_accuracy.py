import os
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
# pynbody.gravity.direct requires its per-particle softening buffer as float32.
eps = np.full(len(galaxy['mass']), galaxy.properties['eps'], dtype=np.float32)
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
relative_error_values = np.asarray(relative_error, dtype=float)
finite_error = relative_error_values[np.isfinite(relative_error_values)]
print(
    "Relative-error summary: "
    f"median={np.median(finite_error):+.4f}, "
    f"p95(|error|)={np.percentile(np.abs(finite_error), 95):.4f}, "
    f"|error|<10%={np.mean(np.abs(finite_error) < 0.1):.3%}"
)
ax.semilogx(phi_comp, relative_error_values, '.', alpha=0.5, markersize=2)
ax.axhline(y=0, color='r', linestyle='--')
ax.axhline(y=0.1, color='gray', linestyle=':', alpha=0.75)
ax.axhline(y=-0.1, color='gray', linestyle=':', alpha=0.75)
ax.set_xlabel('-phi (direct) [km²/s²]')
ax.set_ylabel('Relative error')
ax.set_title('Relative Error (multipole - direct) / direct')
plt.tight_layout()
figure_path = os.environ.get('POTENTIAL_ACCURACY_FIGURE')
if figure_path:
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {figure_path}")
plt.show()
