import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from kinematic_decompose.mixture import AutoGMM
from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.gravity.kinematic_solver import create_multipole_potential, calculate_kinematic_param
from kinematic_decompose.visualize import visualize_decomposition

MAX_RADIUS = 10

def get_ecut(pot, galaxy):
    eps = galaxy.properties.get('eps', 1)
    R_max = min(0.1*galaxy.R_vir/1.414, MAX_RADIUS)
    x = y = np.arange(eps, R_max, eps)
    points = np.column_stack([x,y,np.zeros(len(x))])
    R = np.linalg.norm(points, axis=1)
    potential = pot.potential(points) 
    dpot_dR = np.gradient(potential/pot.potential([0,0,0]), R, edge_order=2)
    d2pot_dR2 = np.gradient(dpot_dR, R, edge_order=2)
    curvature = np.abs(d2pot_dR2) / (1 + dpot_dR**2)**1.5
    rcut = R[np.argmax(curvature)]
    
    e = galaxy.s['eoemin']
    r = galaxy.s['R'] 

    r_median, bin_edges, _ = scipy.stats.binned_statistic(
        e, r, statistic='median', bins=50, range=[-1, -0.]
    )
    ebin = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid_mask = np.isfinite(rmax)
    r_median, ebin = r_median[valid_mask], ebin[valid_mask]

    mask = (r_median < MAX_RADIUS) & (r_median > eps)
    if np.sum(mask) < MAX_RADIUS: mask = (rmax > 0)
    x, y = ebin[mask], r_median[mask]
    y = np.maximum.accumulate(y)
    y, idx = np.unique(y, return_index=True)
    x = x[idx]

    f_r = PchipInterpolator(y, x, extrapolate=True)
    #f_e = PchipInterpolator(x, y, extrapolate=True)

    ecut = f_r(rcut)
    return ecut 

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
print(snapshot.group_catalog['SubhaloPos'], snapshot.group_catalog['SubhaloPos'].units)
snapshot.center(cen=snapshot.group_catalog['SubhaloPos'])
snapshot.faceon(align_with='star', range=[3*snapshot.properties['eps'], 5*snapshot.s.r50], as_context=False)
galaxy = snapshot.container
pot = create_multipole_potential(galaxy['pos'], galaxy['mass'])
x = y = np.arange(2*galaxy.properties['eps'], min(5*galaxy.s.r50/1.414, MAX_RADIUS), galaxy.properties['eps'])
points = np.column_stack([x,y,np.zeros(len(x))])
R = np.linalg.norm(points, axis=1)
potential = pot.potential(points) 
density = pot.density(points)

fig, axes = plt.subplots(1,5, figsize=(20,4))
axes[0].semilogy(R, density)
axes[1].semilogy(R, -potential)
dpot_dR = np.gradient(potential/pot.potential([0.1,0.1,0.1]), R, edge_order=2)
d2pot_dR2 = np.gradient(dpot_dR, R, edge_order=2)
curvature = np.abs(d2pot_dR2) / (1 + dpot_dR**2)**1.5
axes[2].plot(R, dpot_dR)
axes[2].plot(R, d2pot_dR2)
axes[2].plot(R, curvature)
axes[3].plot(R, potential/pot.potential([0.1,0.1,0.1]))
axes[3].axvline(R[np.argmax(curvature)], linestyle=':')

import scipy
from scipy.interpolate import PchipInterpolator
galaxy = calculate_kinematic_param(galaxy)
e = galaxy.s['eoemin']
r = galaxy.s['R'] 
m = galaxy.s['mass']

rmax, bin_edges, _ = scipy.stats.binned_statistic(
    e, r, statistic='median', bins=50, range=[-1, -0.]
)
ebin = 0.5 * (bin_edges[:-1] + bin_edges[1:])
valid_mask = np.isfinite(rmax)
rmax, ebin = rmax[valid_mask], ebin[valid_mask]

mask = (rmax < 10) & (rmax > 0)
if np.sum(mask) < 10:
    mask = (rmax > 0)
x, y = ebin[mask], rmax[mask]

# monotone + deduplicate
y = np.maximum.accumulate(y)
y, idx = np.unique(y, return_index=True)
x = x[idx]

# interpolators
f_r = PchipInterpolator(y, x, extrapolate=True)
f_e = PchipInterpolator(x, y, extrapolate=True)

axes[3].plot(np.arange(0.5, 10), -f_r(np.arange(0.5, 10)))
axes[3].scatter(R[np.argmax(curvature)], -f_r(R[np.argmax(curvature)]))
axes[4].hist(galaxy.s['eoemin'], bins=100, histtype='step')
axes[4].axvline(f_r(R[np.argmax(curvature)]), linestyle=':')
plt.show()
