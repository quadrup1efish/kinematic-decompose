"""
units.py
"""
import numpy as np
from pynbody import units
from pynbody.array import SimArray

UnitLength = units.kpc / units.h
UnitMass = 1e10 * units.Msol / units.h
UnitMassdTime = UnitMass / (0.978 * units.Gyr / units.h)
UnitVel = units.km / units.s
UnitComvingLength = units.a * UnitLength
UnitPressure = (UnitMass / UnitLength) * (units.km / units.s / units.kpc) ** 2
UnitNo = units.no_unit

def get_particle_field_unit(field: str) -> units.Unit:
    field_units = {
        'CenterOfMass': UnitComvingLength,
        'Coordinates': UnitComvingLength,
        'Density': (UnitMass) / (UnitComvingLength) ** 3,
        'ElectronAbundance': UnitNo,
        'EnergyDissipation': (1 / units.a * 1e10 * units.Msol) / (units.a * units.kpc) * (UnitVel) ** 3,
        'GFM_AGNRadiation': units.erg / units.s / units.cm**2 * (4 * np.pi),
        'GFM_CoolingRate': units.erg * units.cm**3 / units.s,
        'GFM_Metallicity': UnitNo,
        'GFM_Metals': UnitNo,
        'GFM_MetalsTagged': UnitNo,
        'GFM_WindDMVelDisp': UnitVel,
        'GFM_WindHostHaloMass': UnitMass,
        'InternalEnergy': (UnitVel) ** 2,
        'InternalEnergyOld': (UnitVel) ** 2,
        'Machnumber': UnitNo,
        'MagneticField': (units.h / units.a**2) * UnitPressure ** (1, 2),
        'MagneticFieldDivergence': (units.h**3 / units.a * 2)
        * (1e10 * units.Msol) ** (1, 2)
        * (UnitVel)
        * (units.a * units.kpc) ** (-5, 2),
        'Masses': (UnitMass),
        'NeutralHydrogenAbundance': UnitNo,
        'ParticleIDs': UnitNo,
        'Potential': (UnitVel) ** 2 / units.a,
        'StarFormationRate': units.Msol / units.yr,
        'SubfindDMDensity': (UnitMass) / (UnitComvingLength) ** 3,
        'SubfindDensity': (UnitMass) / (UnitComvingLength) ** 3,
        'SubfindHsml': UnitComvingLength,
        'SubfindVelDisp': UnitVel,
        'Velocities': units.km * units.a ** (1, 2) / units.s,
        'BirthPos': UnitComvingLength,
        'BirthVel': units.km * units.a ** (1, 2) / units.s,
        'GFM_InitialMass': UnitMass,
        'GFM_StellarFormationTime': UnitNo,
        'GFM_StellarPhotometrics': UnitNo,
        'StellarHsml': UnitComvingLength,
        'BH_BPressure': (units.h / units.a) ** 4
        * 1e10
        * units.Msol
        * (UnitVel) ** 2
        / (units.a * units.kpc) ** 3,
        'BH_CumEgyInjection_QM': UnitMass
        * (UnitComvingLength) ** 2
        / (0.978 * units.Gyr / units.h) ** 2,
        'BH_CumEgyInjection_RM': UnitMass
        * (UnitComvingLength) ** 2
        / (0.978 * units.Gyr / units.h) ** 2,
        'BH_CumMassGrowth_QM': UnitMass,
        'BH_CumMassGrowth_RM': UnitMass,
        'BH_Density': UnitMass / (UnitComvingLength) ** 3,
        'BH_HostHaloMass': UnitMass,
        'BH_Hsml': UnitComvingLength,
        'BH_Mass': UnitMass,
        'BH_Mdot': UnitMassdTime,
        'BH_MdotBondi': UnitMassdTime,
        'BH_MdotEddington': UnitMassdTime,
        'BH_Pressure': UnitMass
        / (UnitComvingLength)
        / (0.978 * units.Gyr / units.h) ** 2,
        'BH_Progs': UnitNo,
        'BH_U': (UnitVel) ** 2,
        'AllowRefinement': UnitNo,
        'BH_WindCount': UnitNo,
        'BH_WindTimes': UnitNo,
        'BH_MPB_CumEgyLow': UnitMass
        * (UnitComvingLength) ** 2
        / (0.978 * units.Gyr / units.h) ** 2,
        'BH_MPB_CumEgyHigh': UnitMass
        * (UnitComvingLength) ** 2
        / (0.978 * units.Gyr / units.h) ** 2,
    }
    return field_units[field]

def get_groupcat_field_unit(field: str) -> units.Unit:
    """
    This function provides the unit corresponding to a given halo or subhalo parameter.

    Parameters:
    -----------
    field : str
        The name of the halo or subhalo parameter for which the unit is requested.

    Returns:
    --------
    unit
        The unit associated with the input parameter. If the parameter is not
        defined in `Matchfieldunits`, the function will raise a KeyError.

    Notes:
    ------
    The function uses a dictionary `Matchfieldunits` to map parameter names to
    their respective units. The units are specified based on the TNG project's data
    specifications as detailed in their documentation:
    https://www.tng-project.org/data/docs/specifications/#sec2
    """
    Matchfieldunits = {
        ### halo properties
        'GroupBHMass': UnitMass,
        'GroupBHMdot': UnitMassdTime,
        'GroupCM': UnitComvingLength,
        'GroupFirstSub': UnitNo,
        'GroupGasMetalFractions': UnitNo,
        'GroupGasMetallicity': UnitNo,
        'GroupLen': UnitNo,
        'GroupLenType': UnitNo,
        'GroupMass': UnitMass,
        'GroupMassType': UnitMass,
        'GroupNsubs': UnitNo,
        'GroupPos': UnitComvingLength,
        'GroupSFR': units.Msol / units.yr,
        'GroupStarMetalFractions': UnitNo,
        'GroupStarMetallicity': UnitNo,
        'GroupVel': units.km / units.s / units.a,
        'GroupWindMass': UnitMass,
        'Group_M_Crit200': UnitMass,
        'Group_M_Crit500': UnitMass,
        'Group_M_Mean200': UnitMass,
        'Group_M_TopHat200': UnitMass,
        'Group_R_Crit200': UnitComvingLength,
        'Group_R_Crit500': UnitComvingLength,
        'Group_R_Mean200': UnitComvingLength,
        'Group_R_TopHat200': UnitComvingLength,
        
        #TNG-Cluster 
        'GroupContaminationFracByMass': UnitNo,
        'GroupContaminationFracByNumPart': UnitNo,
        'GroupOrigHaloID': UnitNo,
        'GroupPrimaryZoomTarget': UnitNo,
        'GroupOffsetType': UnitNo,
        
        ### subhalo properties
        'SubhaloFlag': UnitNo,
        'SubhaloBHMass': UnitMass,
        'SubhaloBHMdot': UnitMassdTime,
        'SubhaloBfldDisk': (units.h / units.a**2) * UnitPressure ** (1, 2),
        'SubhaloBfldHalo': (units.h / units.a**2) * UnitPressure ** (1, 2),
        'SubhaloCM': UnitComvingLength,
        'SubhaloGasMetalFractions': UnitNo,
        'SubhaloGasMetalFractionsHalfRad': UnitNo,
        'SubhaloGasMetalFractionsMaxRad': UnitNo,
        'SubhaloGasMetalFractionsSfr': UnitNo,
        'SubhaloGasMetalFractionsSfrWeighted': UnitNo,
        'SubhaloGasMetallicity': UnitNo,
        'SubhaloGasMetallicityHalfRad': UnitNo,
        'SubhaloGasMetallicityMaxRad': UnitNo,
        'SubhaloGasMetallicitySfr': UnitNo,
        'SubhaloGasMetallicitySfrWeighted': UnitNo,
        'SubhaloGrNr': UnitNo,
        'SubhaloHalfmassRad': UnitComvingLength,
        'SubhaloHalfmassRadType': UnitComvingLength,
        'SubhaloIDMostbound': UnitNo,
        'SubhaloLen': UnitNo,
        'SubhaloLenType': UnitNo,
        'SubhaloMass': UnitMass,
        'SubhaloMassInHalfRad': UnitMass,
        'SubhaloMassInHalfRadType': UnitMass,
        'SubhaloMassInMaxRad': UnitMass,
        'SubhaloMassInMaxRadType': UnitMass,
        'SubhaloMassInRad': UnitMass,
        'SubhaloMassInRadType': UnitMass,
        'SubhaloMassType': UnitMass,
        'SubhaloParent': UnitNo,
        'SubhaloPos': UnitComvingLength,
        'SubhaloSFR': units.Msol / units.yr,
        'SubhaloSFRinHalfRad': units.Msol / units.yr,
        'SubhaloSFRinMaxRad': units.Msol / units.yr,
        'SubhaloSFRinRad': units.Msol / units.yr,
        'SubhaloSpin': UnitLength * UnitVel,
        'SubhaloStarMetalFractions': UnitNo,
        'SubhaloStarMetalFractionsHalfRad': UnitNo,
        'SubhaloStarMetalFractionsMaxRad': UnitNo,
        'SubhaloStarMetallicity': UnitNo,
        'SubhaloStarMetallicityHalfRad': UnitNo,
        'SubhaloStarMetallicityMaxRad': UnitNo,
        'SubhaloStellarPhotometrics': UnitNo,
        'SubhaloStellarPhotometricsMassInRad': UnitMass,
        'SubhaloStellarPhotometricsRad': UnitComvingLength,
        'SubhaloVel': UnitVel,
        'SubhaloVelDisp': UnitVel,
        'SubhaloVmax': UnitVel,
        'SubhaloVmaxRad': UnitComvingLength,
        'SubhaloWindMass': UnitMass,
        
        #TNG-Cluster
        'SubhaloOrigHaloID': UnitNo,
        'SubhaloOffsetType': UnitNo,
        
        #TNG-Cluster Snap 99 only
        'TracerLengthType': UnitNo,
        'TracerOffsetType': UnitNo,
        'SubhaloLengthType': UnitNo,
        'SubhaloOffsetType': UnitNo,
    }
    if field in Matchfieldunits:
        return Matchfieldunits[field]
    else:
        raise KeyError(f"Parameter '{field}' not found in Matchfieldunits.")

"""
field_aliases.py
"""
def get_particle_field_name(field: str) -> str:
    Mapping = {
        'Coordinates': 'pos',
        'Density': 'rho',
        'ParticleIDs': 'iord',
        'Potential': 'phi',
        'Masses': 'mass',
        'Velocities': 'vel',
        'GFM_StellarFormationTime': 'aform',
        'GFM_Metallicity': 'metals',
        'InternalEnergy': 'u',
        'StarFormationRate': 'sfr',
        'InternalEnergy': 'u',
        'ElectronAbundance': 'ElectronAbundance',
        'BH_Mass': 'mass'
    }
    return Mapping[field]

"""
get_eps_mDM.py
"""
def get_eps_mDM(properties) -> tuple[SimArray, SimArray]: 
    MatchRun = {
        'TNG50-1': [0.39, 3.1e5 / 1e10],
        'TNG50-2': [0.78, 2.5e6 / 1e10],
        'TNG50-3': [1.56, 2e7 / 1e10],
        'TNG50-4': [3.12, 1.6e8 / 1e10],
        'TNG100-1': [1., 5.1e6 / 1e10],
        'TNG100-2': [2., 4e7 / 1e10],
        'TNG100-3': [4., 3.2e8 / 1e10],
        'TNG300-1': [2., 4e7 / 1e10],
        'TNG300-2': [4., 3.2e8 / 1e10],
        'TNG300-3': [8., 2.5e9 / 1e10],
        'TNG-Cluster':[2, 6.1e7 / 1e10]
    }

    if properties['Redshift'] > 1:
        return SimArray(
            MatchRun[properties['run']][0], units.a * units.kpc / units.h
        ), SimArray(
            MatchRun[properties['run']][1], 1e10 * units.Msol / units.h
        )
    else:
        return SimArray(
            MatchRun[properties['run']][0] / 2., units.kpc / units.h
        ), SimArray(
            MatchRun[properties['run']][1], 1e10 * units.Msol / units.h
        )
