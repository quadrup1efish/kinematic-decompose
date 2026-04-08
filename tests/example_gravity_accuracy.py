import numpy as np
import matplotlib.pyplot as plt
from kinematic_decompose.gravity.kinematic_solver import create_multipole_potential
from scipy.interpolate import interp1d

rho_s, r_s = 1.0, 1.0
G = 4.30092e-06
r = np.logspace(np.log10(0.1), np.log10(100), 1000)

density_theory = rho_s / (r/r_s) / (1 + r/r_s)**2
potential_theory = -4*np.pi*G*rho_s*r_s**2 * np.log(1 + r/r_s) / (r/r_s)

n_particles = 10000
r_max = 100.0

r_cdf_grid = np.logspace(-6, np.log10(r_max), 50000)
cdf = (np.log(1+r_cdf_grid/r_s) - r_cdf_grid/(r_cdf_grid+r_s))
cdf /= cdf[-1]

inv_cdf = interp1d(cdf, r_cdf_grid, kind='linear')
u = np.random.rand(n_particles)
r_particles = inv_cdf(u)

theta = np.arccos(2*np.random.rand(n_particles)-1)
phi = 2*np.pi*np.random.rand(n_particles)

particle_positions = np.column_stack([
    r_particles*np.sin(theta)*np.cos(phi),
    r_particles*np.sin(theta)*np.sin(phi),
    r_particles*np.cos(theta)
])

m_tot = 4*np.pi*rho_s*r_s**3 * (np.log(1+r_max/r_s) - r_max/(r_max+r_s))
mass = np.ones(n_particles) * m_tot / n_particles 
pot_multipole = create_multipole_potential(particle_positions, mass)
eval_positions = np.column_stack([r, np.zeros((len(r),2))])
density_multipole = pot_multipole.density(eval_positions)

radii = np.linalg.norm(particle_positions, axis=1)
bins = np.logspace(np.log10(0.1), np.log10(100), 50)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
counts, _ = np.histogram(radii, bins=bins, weights=mass)
shell_volumes = 4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3)
density_particles = counts / shell_volumes

plt.figure(figsize=(7,3))
plt.subplot(1,2,1)
plt.loglog(r, density_theory, 'k--', label='Theory')
plt.loglog(r, density_multipole, 'b-', label='Multipole')
plt.loglog(bin_centers, density_particles, 'r-', label='Particles')
plt.xlabel('r')
plt.ylabel('Density')
plt.legend()

pot_multipole = create_multipole_potential(particle_positions, mass)

eval_positions = np.column_stack([r, np.zeros((len(r), 2))])
potential_numerical = pot_multipole.potential(eval_positions)

plt.subplot(1, 2, 2)
plt.loglog(r, (-potential_numerical), 'k-', label='Multipole')
plt.loglog(r, (-potential_theory), 'k--', label='Theory')
plt.xlabel('r')
plt.ylabel('Potential')
plt.legend()
plt.tight_layout()
plt.show()
