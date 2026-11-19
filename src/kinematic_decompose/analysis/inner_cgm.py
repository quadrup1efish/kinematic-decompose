"""Stern-style inner-CGM diagnostic using TNG subhalo members."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..PyTNG import illustris_python as il

LOCAL_COOLING_TABLE_DIR = Path(__file__).resolve().parent / 'data' / 'wiersma'


@dataclass(frozen=True)
class InnerCGM:
    host_id: int
    is_central: bool
    r200c_pkpc: float
    enclosed_mass_fraction: float
    temperature_K: float
    hydrogen_density_cm3: float
    density_cap_cm3: float
    metallicity: float
    tff_Myr: float
    tcool_Myr: float

    @property
    def ratio(self):
        return self.tcool_Myr / self.tff_Myr

    @property
    def density_capped(self):
        return self.hydrogen_density_cm3 > self.density_cap_cm3


def wiersma_cooling(table_dir, temperature, density, metallicity, redshift):
    """Return net cooling in erg cm^3/s from local Wiersma HDF5 tables."""
    table_dir = LOCAL_COOLING_TABLE_DIR if table_dir is None else Path(table_dir)
    tables = sorted((float(p.stem[2:]), p) for p in Path(table_dir).glob('z_*.hdf5')
                    if p.stem[2:].replace('.', '', 1).isdigit())
    zs = np.array([z for z, _ in tables])
    if not len(zs):
        raise FileNotFoundError(f'No Wiersma tables in {table_dir}; set cooling_table_dir to override')
    if not zs[0] <= redshift <= zs[-1]:
        raise ValueError('Cooling tables do not cover the snapshot redshift')
    i = np.searchsorted(zs, redshift)
    indices = [i] if zs[i] == redshift else [i - 1, i]
    solar = metallicity / .014
    he = 10**-1.07 * (.71553 + .28447 * solar)
    helium = 4 * he * (1 - metallicity) / (1 + 4 * he)
    rates = []
    for j in indices:
        with h5py.File(tables[j][1], 'r') as f:
            g = f['Metal_free']
            k = np.argmin(abs(g['Helium_mass_fraction_bins'][:] - helium))
            grid = tuple(np.log10(g[key][:]) for key in ('Temperature_bins', 'Hydrogen_density_bins'))
            rate = g['Net_Cooling'][k] + solar * f['Total_Metals/Net_cooling'][:]
            rates.append(RegularGridInterpolator(grid, rate)(
                np.log10([[temperature, density]]))[0])
    return float(np.interp(redshift, zs[indices], rates))


def measure_inner_cgm(base_path, snap_num, subfind_id, *, cooling_table_dir=None):
    """Measure at 0.1 host R200c using only the requested subhalo's members.

    The host defines the center and radius even for satellites. This is not a
    complete spatial halo measurement. All member gas enters the pressure and
    metallicity estimate, including star-forming gas. Missing gas, undefined
    shock states, and out-of-range or nonpositive cooling raise ValueError.
    """
    base_path = str(base_path)
    sub = il.groupcat.loadSingle(base_path, snap_num, subhaloID=subfind_id)
    host_id = int(sub['SubhaloGrNr'])
    host = il.groupcat.loadSingle(base_path, snap_num, haloID=host_id)
    with h5py.File(il.snapshot.snapPath(base_path, snap_num), 'r') as f:
        header = dict(f['Header'].attrs)
    rc = float(host['Group_R_Crit200'])
    if not np.isfinite(rc) or rc <= 0 or host['Group_M_Crit200'] <= 0:
        raise ValueError('Host has no valid R200c or M200c')
    hubble = header['HubbleParam']
    r200c = rc * header['Time'] / hubble
    box = header['BoxSize']
    edges = 10.**(np.arange(-24, 1) * .05)
    mids = np.sqrt(edges[:-1] * edges[1:])
    q = np.array([.1, .2])
    step = .2  # Symmetric step in natural log radius for the velocity slope.
    nodes = np.unique(np.r_[mids, q, q * np.exp(-step), q * np.exp(step), 1.])
    mass_bins = np.zeros(len(nodes))
    gas_mass = None
    for p in (0, 1, 4, 5):
        if not sub['SubhaloLenType'][p]:
            continue
        fields = ['Coordinates']
        if p != 1:
            fields.append('Masses')
        if p == 0:
            fields.append('GFM_Metallicity')
        data = il.snapshot.loadSubhalo(base_path, snap_num, subfind_id, p, fields=fields)
        xyz = data if p == 1 else data['Coordinates']
        delta = (xyz.astype(float) - host['GroupPos'] + box / 2) % box - box / 2
        r = np.linalg.norm(delta, axis=1) / rc
        m = np.full(len(r), header['MassTable'][1]) if p == 1 else data['Masses'].astype(float)
        mass_bins += np.bincount(np.searchsorted(nodes, r, side='right'),
                                 weights=m, minlength=len(nodes) + 1)[:-1]
        if p == 0:
            gas_mass = np.histogram(r, edges, weights=m)[0]
            lo, hi = .1 * 10.**np.array([-.025, .025])
            shell = (r >= lo) & (r < hi)
            if m[shell].sum() <= 0:
                raise ValueError('No member gas mass in the metallicity shell')
            metallicity = float(np.average(data['GFM_Metallicity'][shell], weights=m[shell]))
    if gas_mass is None:
        raise ValueError('Subhalo has no member gas')
    enclosed = np.cumsum(mass_bins) * 1e10 / hubble

    def vc(r):
        return np.sqrt(4.30093918e-6 * enclosed[np.searchsorted(nodes, r)] / (r * r200c))

    k_B, m_p, kpc = 1.380649e-16, 1.67262192369e-24, 3.085677581e21
    v, vlo, vhi = vc(q), vc(q * np.exp(-step)), vc(q * np.exp(step))
    if np.any(vlo <= 0) or np.any(v <= 0):
        raise ValueError('Insufficient enclosed mass for the shock estimate')
    A = .9 * (1 - np.log(vhi / vlo) / step)
    if np.any(~np.isfinite(A) | (A <= 0)):
        raise ValueError('Circular-velocity slope gives no positive shock temperature')
    temperature = .6 * .62 * m_p * (v * 1e5)**2 / (k_B * A)
    rho = gas_mass * (1e10 / hubble) / (4 * np.pi / 3 * np.diff((edges * r200c)**3))
    widths = np.maximum(0, edges[1:] - np.maximum(edges[:-1], q[:, None]))
    pressure = (widths @ (rho * vc(mids)**2 / mids)) * 1.98847e33 * 1e10 / kpc**3
    density = pressure / (2.3 * k_B * temperature)
    cap = 8 * density[1]
    n = min(density[0], cap)
    if not np.isfinite(n) or n <= 0 or not np.isfinite(metallicity) or metallicity < 0:
        raise ValueError('Invalid shock density or metallicity')
    cooling = wiersma_cooling(cooling_table_dir, temperature[0], n, metallicity,
                              header['Redshift'])
    if not np.isfinite(cooling) or cooling <= 0:
        raise ValueError('Shock state does not have positive net cooling')
    tff = np.sqrt(2) * q[0] * r200c * kpc / (v[0] * 1e5) / 3.15576e13
    tcool = 1.5 * 2.3 * k_B * temperature[0] / (n * cooling) / 3.15576e13
    return InnerCGM(host_id, host['GroupFirstSub'] == subfind_id, float(r200c),
                    float(enclosed[-1] / (host['Group_M_Crit200'] * 1e10 / hubble)),
                    float(temperature[0]), float(density[0]), float(cap), metallicity,
                    float(tff), float(tcool))
