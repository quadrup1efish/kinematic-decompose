import pynbody
import numpy as np
from pynbody import units
from pynbody.array import SimArray

@pynbody.derived_array
def R(sim):
    return sim['rxy']

@pynbody.derived_array
def vR(sim):
    return sim['vxy']

@pynbody.derived_array
def jp(sim):
    j = np.cross(sim['pos'], sim['vel'])
    jp= SimArray(np.sqrt(j[:, 0]**2 + j[:, 1]**2), units.kpc*units.km/units.s)
    return jp

@pynbody.derived_array
def e(sim):
    return sim['phi']+sim['ke']

@pynbody.derived_array
def e_over_emin(sim):
    return sim['e']/sim['e'].min().abs()

@pynbody.derived_array
def eoemin(sim):
    return sim['e']/sim['e'].min().abs()

@pynbody.derived_array
def jzojc(sim):
    return sim['jz']/sim['jc']

@pynbody.derived_array
def jpojc(sim):
    return sim['jp']/sim['jc']

@pynbody.derived_array
def jojc(sim):
    return np.linalg.norm(sim['j'], axis=1)/sim['jc']

@pynbody.derived_array
def age(sim):
    ag = sim.properties['t'] - sim['tform']
    ag.convert_units('Gyr')
    return ag 

@pynbody.derived_array
def tform(
    sim,
):
    """
    Calculates the stellar formation time based on the 'aform' array.

    Notes:
    ------
    The function uses the 'aform' array to compute the formation time, which is then converted to Gyr.
    The calculation requires cosmological parameters like `omegaM0` and `h` from the simulation properties.
    """
    if 'aform' not in sim:
        print('need aform to cal: GFM_StellarFormationTime')
    import numpy as np

    omega_m = sim.properties['omegaM0']
    a = sim['aform'].view(np.ndarray).copy()
    a[a < 0] = 0
    omega_fac = np.sqrt((1 - omega_m) / omega_m) * a ** (3 / 2)
    H0_kmsMpc = 100.0 * sim.ancestor.properties['h']
    t = SimArray(
        2.0 * np.arcsinh(omega_fac) / (H0_kmsMpc * 3 * np.sqrt(1 - omega_m)),
        units.Mpc / units.km * units.s,
    )
    t.convert_units('Gyr')
    t[t == 0] = 14.0
    return t

@pynbody.derived_array
def ne(sim):
    """
    Calculates the electron number density from the electron abundance and hydrogen number density.

    Notes:
    ------
    This function computes the electron number density using the electron abundance and the hydrogen number density.
    It assumes that `ElectronAbundance` and `nH` are available in the simulation object.

    Formula:
    --------
    n_e = ElectronAbundance * n_H
    where:
    - ElectronAbundance is the fraction of electrons per hydrogen atom.
    - n_H is the hydrogen number density in cm^-3.
    """
    n = sim['ElectronAbundance'] * sim['nH'].in_units('cm^-3')
    n.units = units.cm**-3
    return n

@pynbody.derived_array
def em(sim):
    """
    Calculates the Emission Measure (n_e^2) per particle, which is used to be integrated along the line of sight (LoS).

    Formula:
    --------
    EM = n_e^2
    where:
    - n_e is the electron number density in cm^-3.
    """
    return (sim['ne'] * sim['ne']).in_units('cm^-6')

@pynbody.derived_array
def p(sim):
    """
    Calculates the pressure in the gas.

    Notes:
    ------
    The pressure is calculated using the formula:
    P = (2 / 3) * u * rho
    where:
    - u is the internal energy per unit mass.  e.g., InternalEnergy in TNG
    - rho is the gas density in units of solar masses per cubic kiloparsec (Msol kpc^-3). e.g.,  Density in TNG
    """
    p = sim["u"] * sim["rho"].in_units('Msol kpc^-3') * (2.0 / 3)
    p.convert_units("Pa")
    return p


@pynbody.derived_array
def cs(sim):
    """
    Calculates the sound speed in the gas.

    Notes:
    ------
    The sound speed is calculated using the formula:
    c_s = sqrt( (5/3) * (k_B * T) / μ )
    where:
    - k_B is the Boltzmann constant.
    - T is the gas temperature.
    - μ is the mean molecular weight.
    """
    return (np.sqrt(5.0 / 3.0 * units.k * sim['temp'] / sim['mu'])).in_units('km s^-1')


@pynbody.derived_array
def c_s(self):
    """
    Calculates the sound speed of the gas based on pressure and density.

    ------
    The sound speed is calculated using the formula:
    c_s = sqrt( (5/3) * (p / rho) )
    where:
    - p is the gas pressure.
    - rho is the gas density.
    """
    # x = np.sqrt(5./3.*units.k*self['temp']*self['mu'])
    x = np.sqrt(5.0 / 3.0 * self['p'] / self['rho'].in_units('Msol kpc^-3'))
    x.convert_units('km s^-1')
    return x

@pynbody.derived_array
def c_n_sq(sim):
    """
    Calculates the turbulent amplitude C_N^2 for use in spectral calculations,
    As in Eqn 20 of Macquart & Koay 2013 (ApJ 776 2).

    ------
    This calculation assumes a Kolmogorov spectrum of turbulence below the SPH resolution.

    The formula used is:
    C_N^2 = ((beta - 3) / (2 * (2 * π)^(4 - beta))) * L_min^(3 - beta) * EM

    Where:
    - beta = 11/3
    - L_min = 0.1 Mpc (minimum scale of turbulence)
    - EM = emission measure
    """

    ## Spectrum of turbulence below the SPH resolution, assume Kolmogorov
    beta = 11.0 / 3.0
    L_min = 0.1 * units.Mpc
    c_n_sq = (
        ((beta - 3.0) / ((2.0) * (2.0 * np.pi) ** (4.0 - beta)))
        * L_min ** (3.0 - beta)
        * sim["em"]
    )
    c_n_sq.units = units.m ** (-20, 3)

    return c_n_sq

@pynbody.derived_array
def Halpha(sim):
    """
    Compute the H-alpha intensity for each gas particle based on the emission measure.

    References:
    - Draine, B. T. (2011). "Physics of the Interstellar and Intergalactic Medium".
    - For more details on the H-alpha intensity and its calculation, see:
      https://pynbody.readthedocs.io/latest/_modules/pynbody/snapshot/gadgethdf.html
    - Additional information can be found at:
      http://astro.berkeley.edu/~ay216/08/NOTES/Lecture08-08.pdf

    """
    # Define the H-alpha coefficient based on Planck's constant and the speed of light
    coeff = (
        (6.6260755e-27) * (299792458.0 / 656.281e-9) / (4.0 * np.pi)
    )  ## units : erg sr^-1

    # Compute the recombination coefficient for H-alpha
    alpha = coeff * 7.864e-14 * (1e4 / sim['temp'].in_units('K'))

    # Set units for the alpha coefficient
    alpha.units = (
        units.erg * units.cm ** (3) * units.s ** (-1) * units.sr ** (-1)
    )  ## intensity in erg cm^3 s^-1 sr^-1

    # Calculate and return the H-alpha intensity
    return (alpha * sim["em"]).in_units(
        'erg cm^-3 s^-1 sr^-1'
    )  # Flux erg cm^-3 s^-1 sr^-1

@pynbody.derived_array
def nH(sim):
    """
    Calculate the total hydrogen number density for each gas particle.

    The hydrogen number density is computed using the following formula:
    - Total Hydrogen Number Density: X_H * (rho / m_p)
      where X_H is the hydrogen mass fraction, rho is the gas density, and m_p is the proton mass.
    """
    nh = sim['XH'] * (sim['rho'].in_units('g cm^-3') / units.m_p).in_units('cm^-3')
    nh.units = units.cm**-3
    return nh

@pynbody.derived_array
def XH(sim):
    """
    Calculate the hydrogen mass fraction for each gas particle.

    If the 'GFM_Metals' data is available in the simulation, the hydrogen mass fraction is extracted
    from this data. If 'GFM_Metals' is not present, a default value of 0.76 is used.
    """
    if 'GFM_Metals' in sim:
        Xh = sim['GFM_Metals'].view(np.ndarray).T[0]
        return SimArray(Xh)
    else:
        #print('No GFM_Metals, use hydrogen mass fraction XH=0.76')
        return SimArray(0.76 * np.ones(len(sim)))

@pynbody.derived_array
def mu(sim):
    """
    Calculate the mean molecular weight of the gas.

    The mean molecular weight is computed using the hydrogen mass fraction (XH) and the electron
    abundance. The formula used is:
        μ = 4 / (1 + 3 * XH + 4 * XH * ElectronAbundance)
    """
    if 'ElectronAbundance' not in sim:
        print('need gas ElectronAbundance to cal: ElectronAbundance')
    muu = SimArray(
        4
        / (1 + 3 * sim['XH'] + 4 * sim['XH'] * sim['ElectronAbundance']).astype(
            np.float64
        ),
        units.m_p,
    )
    return muu.in_units('m_p')

@pynbody.derived_array
def temp(sim):
    """
    Calculates the gas temperature based on the internal energy.

    Notes:
    ------
    This function uses the two-phase ISM sub-grid model to calculate the gas temperature.
    The formula used is based on the internal energy and gas properties.
    For more information, refer to Sec.6 of the TNG FAQ:
    https://www.tng-project.org/data/docs/faq/
    """
    if 'u' not in sim:
        print('need gas InternalEnergy to cal: InternalEnergy')
    gamma = 5.0 / 3
    UnitEtoUnitM = ((units.kpc / units.Gyr).in_units('km s^-1')) ** 2
    T = (gamma - 1) / units.k * sim['mu'] * sim['u'] * UnitEtoUnitM

    T.convert_units('K')
    return T
