import math
import warnings
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
counter_rotating_disk_filter = filt.BandPass('label',  3.5, 4.5)

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

def _counter_rotating_disk(self):
    return self.s[counter_rotating_disk_filter]

def _coldgas(self):
    return self.g[cold_gas_filter]#&disk_gas_filter]

"""
Some useful function -> r50, R50, z50, t50, Rvir, Vvir, Mvir, Tvir, Spin, AM
"""

def _mass_percentile(self, field: str, percent: float, weight: str = 'mass',
                     abs_first: bool = False):
    """Mass-weighted percentile of ``field`` (a single percentile helper).

    Sorts particles by ``field`` and returns its value at the mass
    percentile ``percent`` (e.g. 0.50 for the half-mass radius).

    Parameters
    ----------
    field : str
        Array name to take the percentile of (e.g. ``'r'``, ``'R'``, ``'z'``).
    percent : float
        Mass percentile in [0, 1].
    weight : str
        Array used as the percentile weight (default ``'mass'``).
    abs_first : bool
        If True, sort by ``|field|`` (used for ``'z'``).
    """
    arr = np.abs(self[field]) if abs_first else self[field]
    if len(arr) == 0:
        return np.nan
    sort_idx = np.argsort(arr)
    cum_mass = np.cumsum(np.asarray(self[weight], dtype=np.float64)[sort_idx])
    idx = np.searchsorted(cum_mass, cum_mass[-1] * percent)
    return SimArray(arr[sort_idx][idx], units=self[field].units)


def _r(self, weight='mass', percent=0.50):
    return _mass_percentile(self, 'r', percent, weight)


def _r90(self, weight='mass', percent=0.90):
    return _mass_percentile(self, 'r', percent, weight)


def _r25(self, weight='mass', percent=0.25):
    return _mass_percentile(self, 'r', percent, weight)


def _R(self, weight='mass', percent=0.50):
    return _mass_percentile(self, 'R', percent, weight)


def _z(self, weight='mass', percent=0.50):
    return _mass_percentile(self, 'z', percent, weight, abs_first=True)


def _t(self, weight='mass', percent=0.50):
    return _mass_percentile(self, 'tform', percent, weight)

def _vel_disp(self):
    if len(self['r']) == 0: return np.nan 
    return SimArray(np.linalg.norm(np.std(self['vel'][self['r']<self.r50], axis=0)), units=self['vel'].units)

def _vr_disp(self):
    if len(self['r']) == 0: return np.nan 
    return SimArray(np.std(self['vr'][self['r']<self.r50], axis=0), units=self['vel'].units)

def _vz_disp(self):
    if len(self['r']) == 0: return np.nan 
    return SimArray(np.std(self['vz'][self['r']<self.r50], axis=0), units=self['vel'].units)

def _vR_disp(self):
    if len(self['R']) == 0: return np.nan 
    return SimArray(np.std(self['vR'][self['R']<self.R50], axis=0), units=self['vel'].units)

def _ke(self):
    if len(self['r']) == 0: return np.nan 
    return SimArray((self['ke'][self['r']<self.r50]).mean(), units=self['ke'].units)

def _mass_frac(self):
    sim = self.ancestor
    return self['mass'].sum()/sim.s['mass'].sum()
 
def _mdyn(self):
    if len(self['r']) == 0: return np.nan 
    sim = self.ancestor
    return sim['mass'][sim['r']<self.r50].sum()

def _mcold(self):
    if len(self['r']) == 0: return np.nan 
    sim = self.ancestor
    return sim.coldgas['mass'][sim.coldgas['r']<self.r50].sum() + sim.colddisk['mass'][sim.colddisk['r']<self.r50].sum()
 
def _mbary(self):
    if len(self['r']) == 0: return np.nan 
    sim = self.ancestor
    return sim.s['mass'][sim.s['r']<self.r50].sum() + sim.g['mass'][sim.g['r']<self.r50].sum()

def _v_circ(self):
    if len(self['r']) == 0: return np.nan 
    sim = self.ancestor
    v_circ = (units.G*(sim['mass'][sim['r']<self.r50]).sum()/self.r50)**0.5#np.mean(self['vcxy'][self['r']<self.r50])
    return v_circ.in_units(self['vel'].units)

def _v_rot(self):
    if len(self['r']) == 0: return np.nan 
    return np.mean(self['vcxy'][self['r']<self.r50])

def _mass(self):
    return self['mass'].sum()

def _mvir(self):
    return self['mass'].sum()

def _rvir(self):
    sim = self.ancestor
    rho_c = rho_crit(sim, sim.properties['z'])           
    mvir  = self.M_vir          
    rvir = (3 * mvir / (4 * np.pi * _VIRIAL_OVERDENSITY * rho_c))**(1/3)
    return rvir.in_units('kpc')

def _vvir(self):
    vvir = np.sqrt(units.G * self.M_vir / self.R_vir)
    return vvir.in_units('km s**-1')

def _Tvir(self):
    mu = _TVIR_MEAN_MOLECULAR_WEIGHT
    mp = units.m_p
    kb = units.k
    T = mu * mp * self.V_vir**2 / (2 * kb)
    return T.in_units('K')

def _AM(self):
    angmom = (self['mass'][:, None] *
              np.cross(self['pos'], self['vel'])).sum(axis=0)
    result = angmom.view(SimArray)
    result.units = self['mass'].units * self['pos'].units * self['vel'].units
    return result

def _spin(self):
    sim = self.ancestor
    j = np.sqrt((self.AM**2).sum()) / self.mass
    return j / (np.sqrt(2) * sim.R_vir * sim.V_vir)

def _krot(self):
    return (0.5 * self['mass'] * self['vcxy']**2).sum() / \
           (self['mass'] * self['ke']).sum()

def _beta(self):
    return 1 - (self['vt']**2 + self['vphi']**2).mean() / \
               (2 * self['vr']**2).mean()

def _tff(self):
    r_circ = self.r_circ
    mask = self['r'] < r_circ
    Menc = self['mass'][mask].sum()
    vc = np.sqrt(units.G * Menc / r_circ)
    tff = np.sqrt(2) * r_circ / vc
    return tff.in_units('Gyr')

def _rcirc(self):
    return (np.sqrt(2) * self.spin * self.r_vir).in_units('kpc')

# --- physical constants / tolerances (named instead of magic numbers) ---
_VIRIAL_OVERDENSITY = 200   # Delta_vir used for R_vir (spherical top-hat)
_TVIR_MEAN_MOLECULAR_WEIGHT = 0.62  # primordial plasma mu (T_vir formula)
_TFORM_FALLBACK_AGE = 14.0  # Gyr, assigned when tform == 0 (no star formation)
_SHAPE_MAX_ITER = 100       # inertia-tensor iteration cap
_SHAPE_CONV_TOL = 1e-2      # ||axes_new - axes|| convergence threshold

def _shape(self):
    pos  = self['pos']
    mass = self['mass']
    rmax = np.inf
    evec = np.eye(3)   # initial guess for axes orientation
    axes = np.ones(3)  # and axes ratios; these are updated at each iteration
    for _ in range(_SHAPE_MAX_ITER):
        # use particles within the elliptical radius less than the provided value
        ellpos  = pos.dot(evec) / axes
        sel     = np.sum(ellpos**2, axis=1) < rmax**2
        inertia = pos[sel].T.dot(pos[sel] * mass[sel,None])
        val,vec = np.linalg.eigh(inertia)
        order   = np.argsort(-val)  # sort axes in decreasing order
        evec    = vec[:,order]         # updated axes directions
        axesnew = (val[order] / np.prod(val)**(1./3))**0.5  # updated axes ratios, normalized so that ax*ay*az=1
        if sum(abs(axesnew-axes))<_SHAPE_CONV_TOL: break
        axes    = axesnew 
    else:
        warnings.warn(
            f"Shape tensor did not converge within 100 iterations for {self}",
            RuntimeWarning,
        )
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
    Sim.counter_rotating_disk = property(_counter_rotating_disk)
    Sim.coldgas = property(_coldgas)

    for cls in classes:
        cls.r50 = property(_r)
        cls.r25 = property(_r25)
        cls.r90 = property(_r90)
        cls.R50 = property(_R)
        cls.z50 = property(_z)
        cls.t50 = property(_t)
        cls.krot= property(_krot)
        cls.beta= property(_beta)
        cls.AM  = property(_AM)
        cls.mass  = property(_mass) 
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

from pynbody import family
from pynbody.snapshot import SimSnap

def new(n_particles = 0, order = None, class_ = SimSnap, **families) -> SimSnap:
    """Create a blank SimSnap, with the specified number of particles.

    Position, velocity and mass arrays are created and filled with zeros.

    By default all particles are taken to be dark matter.

    To specify otherwise, pass in keyword arguments specifying the number of particles for each family, e.g.

    >>> f = new(dm=50, star=25, gas=25)

    The order in which the different families appear in the snapshot is unspecified unless you add an 'order' argument:

    >>> f = new(dm=50, star=25, gas=25, order='star,gas,dm')

    guarantees the stars, then gas, then dark matter particles appear in sequence.
    """

    if len(families) == 0:
        families = {'dm': n_particles}

    t_fam = []
    tot_particles = 0

    if order is None:
        for k, v in list(families.items()):

            assert isinstance(v, int)
            t_fam.append((family.get_family(k), v))
            tot_particles += v
    else:
        for k in order.split(","):
            v = families[k]
            assert isinstance(v, int)
            t_fam.append((family.get_family(k), v))
            tot_particles += v

    x = class_()
    x._num_particles = tot_particles
    x._filename = "<created>"

    x._create_arrays(["pos"], 3)
    #x._create_arrays(["mass"], 1)

    rt = 0
    for k, v in t_fam:
        x._family_slice[k] = slice(rt, rt + v)
        rt += v

    x._decorate()
    return x

pynbody.new = new

from pynbody.transformation import Transformation
class GenericTranslation(Transformation):

    def __init__(self, f, arname, shift, description=None):
        self.shift = shift
        self.arname = arname
        super().__init__(f, description=description)

    def _find_targets(self, f):
        families = []
        for fam_name in ['gas', 'dm', 'star', 'bh']:
            fam = getattr(f, fam_name, None)
            if fam is not None and len(fam) > 0 and self.arname in fam:
                families.append(fam)
        return families

    def _apply_to_snapshot(self, f):
        for target in self._find_targets(f):
            target[self.arname] += self.shift

    def _unapply_to_snapshot(self, f):
        for target in self._find_targets(f):
            target[self.arname] -= self.shift

    def _apply_to_array(self, array):
        if array.name == self.arname:
            array += self.shift

pynbody.transformation.GenericTranslation = GenericTranslation
