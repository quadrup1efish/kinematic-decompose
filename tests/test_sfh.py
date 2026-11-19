import numpy as np

from kinematic_decompose.analysis.sfh import get_cumsum_sf, get_sfr


def test_sfh_bin_edges_and_mass():
    mass = np.array([2e9, 3e9, 4e9, np.nan, 1e9])
    times = np.array([0.2, 0.8, 2.0, 0.5, 3.0])
    edges, sfr = get_sfr(mass, times, bins=2, range=(0, 2))
    cumulative_edges, cumulative_mass = get_cumsum_sf(mass, times, bins=2, range=(0, 2))
    np.testing.assert_array_equal(edges, [0, 1, 2])
    np.testing.assert_array_equal(cumulative_edges, edges)
    np.testing.assert_array_equal(sfr, [0, 5, 4])
    np.testing.assert_array_equal(cumulative_mass, [0, 5e9, 9e9])
