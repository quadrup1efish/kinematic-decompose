"""Binned star formation rates and cumulative formed stellar masses."""

import numpy as np
import scipy.stats


def get_sfr(mass, tform, bins=137, range=(0, 13.8)):
    """Return bin edges and mean SFR in solar masses per year.

    Input masses are initial stellar masses in solar masses; times are in Gyr.
    The rate array starts with zero, followed by one rate per bin.
    """
    valid = ~np.isnan(mass) & ~np.isnan(tform)
    mass_sum, bin_edges, _ = scipy.stats.binned_statistic(
        tform[valid], mass[valid], statistic='sum', bins=bins, range=range)
    delta_t = np.diff(bin_edges)
    SFR = mass_sum / delta_t / 1e9
    SFR = np.insert(SFR, 0, 0)
    return bin_edges, SFR


def get_cumsum_sf(mass, tform, bins=137, range=(0, 13.8)):
    """Return bin edges and cumulative initial stellar mass in solar masses.

    Input formation times are in Gyr. The first edge has zero cumulative mass.
    """
    valid = ~np.isnan(mass) & ~np.isnan(tform)
    mass_sum, bin_edges, _ = scipy.stats.binned_statistic(
        tform[valid], mass[valid], statistic='sum', bins=bins, range=range)
    mass_sum = np.cumsum(np.insert(mass_sum, 0, 0))
    return bin_edges, mass_sum
