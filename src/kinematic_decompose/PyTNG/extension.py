import math
import numpy as np
from functools import lru_cache

import pynbody
from pynbody import filt, units
from pynbody.array import SimArray
from pynbody.analysis.cosmology import _a_dot

def rho_crit(sim, z=None):
    if z is None:
        z = sim.properties['z']

    omM = sim.properties['omegaM0']
    omL = sim.properties['omegaL0']
    h0  = sim.properties['h']

    a = 1.0 / (1.0 + z)
    Hz_num = _a_dot(a, h0, omM, omL) / a
    H_z = SimArray(Hz_num, units.Unit("100 km s^-1 Mpc^-1"))

    rho_c = (3 * H_z**2) / (8 * math.pi * units.G)

    return rho_c.in_units('Msol kpc**-3')


"""
Add the IndexedSubSnap for different kinematic structures: -> galaxy.disk
"""
disk_filter      = filt.BandPass('label', -0.5, 1.5) # -> 0, 1
spheroid_filter  = filt.BandPass('label', 1.5, 3.5) # -> 2, 3
cold_disk_filter = filt.BandPass('label', -0.5, 0.5)
warm_disk_filter = filt.BandPass('label',  0.5, 1.5)
bulge_filter     = filt.BandPass('label',  1.5, 2.5)
halo_filter      = filt.BandPass('label',  2.5, 3.5)
cold_gas_filter  = filt.BandPass('temp',   0  , 10**5)
disk_gas_filter  = filt.BandPass('jzojc',  0.85, 1.5)
counter_rotate_disk_filter = filt.BandPass('label',  3.5, 4.5)

def _disk(self):
    return self.s[disk_filter]

def _spheroid(self):
    return self.s[spheroid_filter]

def _colddisk(self):
    return self.s[cold_disk_filter]

def _warmdisk(self):
    return self.s[warm_disk_filter]

def _bulge(self):
    return self.s[bulge_filter]

def _halo(self):
    return self.s[halo_filter]

def _counter_rotate_disk(self):
    return self.s[counter_rotate_disk_filter]

def _coldgas(self):
    return self.g[cold_gas_filter]#&disk_gas_filter]

"""
Some useful function -> r50, R50, z50, t50, Rvir, Vvir, Mvir, Tvir, Spin, AM
"""
@lru_cache(maxsize=None) 
def _r(self, weight='mass', percent=0.50):
    if len(self['r']) == 0: return np.nan
    r_filtered = self['r']
    w_filtered = self[weight]
    sort_idx = np.argsort(r_filtered)
    r_sorted = r_filtered[sort_idx]
    w_sorted = w_filtered[sort_idx]
    cum_mass = np.cumsum(w_sorted)
    total_mass = cum_mass[-1]
    r_idx = np.searchsorted(cum_mass, total_mass * percent)
    return SimArray(r_sorted[r_idx], units=self['pos'].units)

@lru_cache(maxsize=None)
def _R(self, weight='mass', percent=0.50):
    if len(self['R']) == 0: return np.nan
    r_filtered = self['R']
    w_filtered = self[weight]
    sort_idx = np.argsort(r_filtered)
    r_sorted = r_filtered[sort_idx]
    w_sorted = w_filtered[sort_idx]
    cum_mass = np.cumsum(w_sorted)
    total_mass = cum_mass[-1]
    r_idx = np.searchsorted(cum_mass, total_mass * percent)
    return  SimArray(r_sorted[r_idx], units=self['pos'].units)

@lru_cache(maxsize=None)
def _z(self, weight='mass', percent=0.50):
    if len(self['z']) == 0: return np.nan
    z_filtered = self['z'].abs()
    w_filtered = self[weight]
    sort_idx = np.argsort(z_filtered)
    z_sorted = z_filtered[sort_idx]
    w_sorted = w_filtered[sort_idx]
    cum_mass = np.cumsum(w_sorted)
    total_mass = cum_mass[-1]
    z_idx = np.searchsorted(cum_mass, total_mass * percent)
    return  SimArray(z_sorted[z_idx], units=self['pos'].units)

@lru_cache(maxsize=None)
def _t(self, weight='mass', percent=0.50):
    if len(self['tform']) == 0: return np.nan
    t_filtered = self['tform']
    w_filtered = self[weight]
    sort_idx = np.argsort(t_filtered)
    t_sorted = t_filtered[sort_idx]
    w_sorted = w_filtered[sort_idx]
    cum_mass = np.cumsum(w_sorted)
    total_mass = cum_mass[-1]
    t_idx = np.searchsorted(cum_mass, total_mass * percent)
    return SimArray(t_sorted[t_idx], units=self['tform'].units)

@lru_cache(maxsize=None)
def _vel_disp(self):
    if len(self['r']) == 0: return np.nan 
    return SimArray(np.linalg.norm(np.std(self['vel'][self['r']<self.r50], axis=0)), units=self['vel'].units)

@lru_cache(maxsize=None)
# TODO: Add radial vel_disp
def _vr_disp(self):
    if len(self['r']) == 0: return np.nan 
    return SimArray(np.std(self['vr'][self['r']<self.r50], axis=0), units=self['vel'].units)

@lru_cache(maxsize=None)
def _vz_disp(self):
    if len(self['r']) == 0: return np.nan 
    return SimArray(np.std(self['vz'][self['r']<self.r50], axis=0), units=self['vel'].units)

@lru_cache(maxsize=None) 
def _vR_disp(self):
    if len(self['R']) == 0: return np.nan 
    return SimArray(np.std(self['vR'][self['R']<self.R50], axis=0), units=self['vel'].units)

@lru_cache(maxsize=None) 
def _ke(self):
    if len(self['r']) == 0: return np.nan 
    return SimArray((self['ke'][self['r']<self.r50]).mean(), units=self['ke'].units)

@lru_cache(maxsize=None)
def _mass_frac(self):
    sim = self.ancestor
    return self['mass'].sum()/sim.s['mass'].sum()

@lru_cache(maxsize=None) 
def _mdyn(self):
    if len(self['r']) == 0: return np.nan 
    sim = self.ancestor
    return sim['mass'][sim['r']<self.r50].sum()

@lru_cache(maxsize=None)
def _mcold(self):
    if len(self['r']) == 0: return np.nan 
    sim = self.ancestor
    return sim.coldgas['mass'][sim.coldgas['r']<self.r50].sum() + sim.colddisk['mass'][sim.colddisk['r']<self.r50].sum()

@lru_cache(maxsize=None) 
def _mbary(self):
    if len(self['r']) == 0: return np.nan 
    sim = self.ancestor
    return sim.s['mass'][sim.s['r']<self.r50].sum() + sim.g['mass'][sim.g['r']<self.r50].sum()

@lru_cache(maxsize=None) 
def _v_circ(self):
    if len(self['r']) == 0: return np.nan 
    sim = self.ancestor
    v_circ = (units.G*(sim['mass'][sim['r']<self.r50]).sum()/self.r50)**0.5#np.mean(self['vcxy'][self['r']<self.r50])
    return v_circ.in_units(self['vel'].units)

@lru_cache(maxsize=None)
def _v_rot(self):
    if len(self['r']) == 0: return np.nan 
    return np.mean(self['vcxy'][self['r']<self.r50])

@lru_cache(maxsize=None)
def _mvir(self):
    return self['mass'].sum()

@lru_cache(maxsize=None)
def _rvir(self):
    sim = self.ancestor
    overden = 200
    rho_c = rho_crit(sim, sim.properties['z'])           
    mvir  = self.M_vir          
    rvir = (3 * mvir / (4 * np.pi * overden * rho_c))**(1/3)
    return rvir.in_units('kpc')

@lru_cache(maxsize=None)
def _vvir(self):
    vvir = np.sqrt(units.G * self.M_vir / self.R_vir)
    return vvir.in_units('km s**-1')

@lru_cache(maxsize=None)
def _Tvir(self):
    mu = 0.62
    mp = units.m_p
    kb = units.k
    T = mu * mp * self.V_vir**2 / (2 * kb)
    return T.in_units('K')

#@lru_cache(maxsize=None)
def _AM(self):
    angmom = (self['mass'][:, None] *
              np.cross(self['pos'], self['vel'])).sum(axis=0)
    result = angmom.view(SimArray)
    result.units = self['mass'].units * self['pos'].units * self['vel'].units
    return result

@lru_cache(maxsize=None)
def _spin(self):
    sim = self.ancestor
    j = np.sqrt((self.AM**2).sum()) / self.M_vir
    return j / (np.sqrt(2) * sim.R_vir * sim.V_vir)

@lru_cache(maxsize=None)
def _krot(self):
    return (0.5 * self['mass'] * self['vcxy']**2).sum() / \
           (self['mass'] * self['ke']).sum()

@lru_cache(maxsize=None) 
def _beta(self):
    return 1 - (self['vt']**2 + self['vphi']**2).mean() / \
               (2 * self['vr']**2).mean()

@lru_cache(maxsize=None) 
def _tff(self):
    r_circ = self.r_circ
    mask = self['r'] < r_circ
    Menc = self['mass'][mask].sum()
    vc = np.sqrt(units.G * Menc / r_circ)
    tff = np.sqrt(2) * r_circ / vc
    return tff.in_units('Gyr')

@lru_cache(maxsize=None)
def _rcirc(self):
    return (np.sqrt(2) * self.spin * self.r_vir).in_units('kpc')

@lru_cache(maxsize=None)
def _shape(self):
    pos  = self['pos']
    mass = self['mass']
    rmax = np.inf
    evec = np.eye(3)   # initial guess for axes orientation
    axes = np.ones(3)  # and axes ratios; these are updated at each iteration
    for _ in range(100):
        # use particles within the elliptical radius less than the provided value
        ellpos  = pos.dot(evec) / axes
        filter  = np.sum(ellpos**2, axis=1) < rmax**2
        inertia = pos[filter].T.dot(pos[filter] * mass[filter,None])
        val,vec = np.linalg.eigh(inertia)
        order   = np.argsort(-val)  # sort axes in decreasing order
        evec    = vec[:,order]         # updated axes directions
        axesnew = (val[order] / np.prod(val)**(1./3))**0.5  # updated axes ratios, normalized so that ax*ay*az=1
        if sum(abs(axesnew-axes))<1e-2: break
        axes    = axesnew 
    if np.linalg.det(evec)<0: evec *= -1
    if evec[2,2]<0: evec[:,1:3] *= -1
    if evec[1,1]<0: evec[:,0:2] *= -1
    return axes#, filter, evec


def register():
    Sim = pynbody.snapshot.SimSnap
    classes = [Sim, pynbody.snapshot.subsnap.FamilySubSnap, pynbody.snapshot.subsnap.IndexedSubSnap]

    Sim.disk = property(_disk)
    Sim.spheroid = property(_spheroid)
    Sim.colddisk = property(_colddisk)
    Sim.warmdisk = property(_warmdisk)
    Sim.bulge = property(_bulge)
    Sim.halo = property(_halo)
    Sim.counter_rotate_disk = property(_counter_rotate_disk)
    Sim.coldgas = property(_coldgas)

    for cls in classes:
        cls.r50 = property(_r)
        cls.R50 = property(_R)
        cls.z50 = property(_z)
        cls.t50 = property(_t)
        cls.krot = property(_krot)
        cls.beta = property(_beta)
        cls.AM = property(_AM)
        cls.M_vir = property(_mvir)
        cls.V_vir = property(_vvir)
        cls.R_vir = property(_rvir)
        cls.T_vir = property(_Tvir)
        cls.spin  = property(_spin)
        cls.vel_disp = property(_vel_disp)
        cls.vr_disp = property(_vr_disp)
        cls.vR_disp = property(_vR_disp)
        cls.vz_disp = property(_vz_disp)
        cls.v_circ = property(_v_circ)
        cls.v_rot = property(_v_rot)
        cls.ke = property(_ke)
        cls.Mdyn = property(_mdyn)
        cls.Mcold= property(_mcold)
        cls.Mbary= property(_mbary)
        cls.Mass_frac = property(_mass_frac)
        cls.shape = property(_shape)

register()

import pynbody.sph.renderers as renderers

def _calculate_wrapping_repeat_array(self, x1, x2):
    if 'boxsize' in self._snapshot.properties:
        boxsize = self._snapshot.properties['boxsize'].in_units(
            self._snapshot['pos'].units,
            **self._snapshot.conversion_context()
        )
    else:
        boxsize = None

    if boxsize is not None:
        ratio = ((x2 - x1) / (2 * boxsize)).item()
        num_repeats = int(round(ratio)) + 1
        repeat_array = np.linspace(-num_repeats * boxsize,
                                   num_repeats * boxsize,
                                   num_repeats * 2 + 1)
    else:
        repeat_array = [0.0]

    return repeat_array

renderers.ImageRenderer._calculate_wrapping_repeat_array = _calculate_wrapping_repeat_array

from pynbody.snapshot.util import ContainerWithPhysicalUnitsOption 

def physical_units(self, distance='kpc', velocity='km s^-1', mass='Msol', persistent=False):
    standard_units = self.properties.get('standard_units', [])
    if len(standard_units) == 0:
        self.physical_units(distance, velocity, mass, persistent)
    else:
        from functools import reduce
        dims = [units.kpc, units.km/units.s, units.Msol, units.a, units.h]
        urc = len(dims) - 2
        all = list(self._arrays.values()) 
        for x in self._family_arrays:
            if x in standard_units:
                continue
            else:
                all += list(self._family_arrays[x].values())

        for ar in all:
            if ar.units is not units.no_unit:
                self._autoconvert_array_unit(ar.ancestor, dims, urc)

        for k in list(self.properties):
            if k in standard_units: 
                continue
            v = self.properties[k]
            if isinstance(v, units.UnitBase):
                try:
                    new_unit = v.dimensional_project(dims)
                except units.UnitsException:
                    continue
                new_unit = reduce(
                    lambda x, y: x * y, [a**b for a, b in zip(dims, new_unit[:])]
                )
                new_unit *= v.ratio(new_unit, **self.conversion_context())
                self.properties[k] = new_unit
            if isinstance(v, SimArray):
                v.units.dimensional_project(dims) 
                if (v.units is not None) and (v.units is not units.no_unit):
                    try:
                        d = v.units.dimensional_project(dims)
                    except units.UnitsException:
                        return
                    new_unit = reduce(
                        lambda x, y: x * y, [a**b for a, b in zip(dims, d[:urc])]
                    )
                    if new_unit != v.units:
                        self.properties[k].convert_units(new_unit)
        if persistent:
            self._autoconvert = dims
        else:
            self._autoconvert = None

ContainerWithPhysicalUnitsOption.physical_units = physical_units
