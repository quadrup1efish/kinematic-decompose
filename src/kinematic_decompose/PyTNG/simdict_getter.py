import numpy as np

from pynbody import units
from pynbody.simdict import SimDict

@SimDict.getter
def t(d):
    """
    Calculate the age of the snapshot

    This function uses cosmological parameters and redshift to compute the age of the snapshot.
    The formula is derived from Peebles (p. 317, eq. 13.2).
    """
    omega_m = d['omegaM0']
    redshift = d['z']
    H0_kmsMpc = 100.0 * d['h'] * units.km / units.s / units.Mpc

    return get_t(omega_m, redshift, H0_kmsMpc)

@SimDict.getter
def rho_crit(d):
    z = d['z']
    omM = d['omegaM0']
    omL = d['omegaL0']
    h0 = d['h']
    a = d['a']
    omK = 1.0 - omM - omL
    _a_dot = h0 * a * np.sqrt(omM * (a**-3) + omK * (a**-2) + omL)
    H_z = _a_dot / a
    H_z = units.Unit("100 km s^-1 Mpc^-1") * H_z

    rho_crit = (3 * H_z**2) / (8 * np.pi * units.G)
    return rho_crit

@SimDict.getter
def tLB(d):
    """
    Calculate the lookback time.
    """

    omega_m = d['omegaM0']
    redshift = 0.0
    H0_kmsMpc = 100.0 * d['h'] * units.km / units.s / units.Mpc

    tlb = get_t(omega_m, redshift, H0_kmsMpc) - d['t']
    return tlb


@SimDict.getter
def cosmology(d):
    cos = {}
    cos['h'] = d.get('h')
    cos['omegaM0'] = d.get('omegaM0')
    cos['omegaL0'] = d.get('omegaL0')
    cos['omegaB0'] = d.get('omegaB0')
    cos['sigma8'] = d.get('sigma8')
    cos['ns'] = d.get('ns')
    return cos


def get_t(omega_m, redshift, H0_kmsMpc):
    import math

    omega_fac = math.sqrt((1 - omega_m) / omega_m) * pow(1 + redshift, -3.0 / 2.0)
    AGE = 2.0 * math.asinh(omega_fac) / (H0_kmsMpc * 3 * math.sqrt(1 - omega_m))
    return AGE.in_units('Gyr') * units.Gyr





