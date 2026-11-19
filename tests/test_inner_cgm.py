import h5py
import numpy as np
import pytest

from kinematic_decompose.analysis import inner_cgm
from kinematic_decompose.analysis.inner_cgm import wiersma_cooling


def test_catalog_backfills_inner_cgm_without_redecomposition(tmp_path, monkeypatch):
    from kinematic_decompose.analysis.catalog import catalog, _snapshot

    path = tmp_path / 'catalog.h5'
    with h5py.File(path, 'w') as f:
        f.create_group('21/78482')['existing'] = 42
    monkeypatch.setattr(catalog, '_cached_payload_valid', lambda grp: True)

    calls = []

    def measure(base, snap, sid, *, cooling_table_dir):
        calls.append((snap, sid, cooling_table_dir))
        return inner_cgm.InnerCGM(277, True, 37., .84, 1.4e6, .06, .17, .002, 35., 14.)

    monkeypatch.setattr(_snapshot, 'measure_inner_cgm', measure)
    for _ in range(2):
        assert catalog.build_catalog(tmp_path, path, snapshot_targets=[(21, 78482)],
                                     cooling_table_dir=tmp_path) == (0, 1, 0)
    assert calls == [(21, 78482, tmp_path)]
    with h5py.File(path, 'r') as f:
        assert f['21/78482/existing'][()] == 42
        grp = f['21/78482/inner_cgm']
        assert not list(grp)
        assert 'catalog_v2' not in f
        assert len(f['catalog']) == 1
        row = f['catalog'][0]
        assert row['tcool_Myr'] == 14.
        assert row['tff_Myr'] == 35.
        assert row['tcool_tff'] == pytest.approx(.4)


def test_wiersma_interpolates_redshift_and_metallicity(tmp_path):
    for z, rate in [(3., 2.), (5., 4.)]:
        with h5py.File(tmp_path / f'z_{z:.3f}.hdf5', 'w') as f:
            g = f.create_group('Metal_free')
            g['Helium_mass_fraction_bins'] = [.25]
            g['Temperature_bins'] = [1e5, 1e6]
            g['Hydrogen_density_bins'] = [.01, .1]
            g['Net_Cooling'] = [[[rate, rate], [rate, rate]]]
            f.create_group('Total_Metals')['Net_cooling'] = [[2., 2.], [2., 2.]]
    assert wiersma_cooling(tmp_path, 2e5, .02, .007, 4.) == pytest.approx(4.)
    with pytest.raises(ValueError):
        wiersma_cooling(tmp_path, 2e5, .02, .007, 6.)
    with pytest.raises(ValueError):
        wiersma_cooling(tmp_path, 2e5, 1., .007, 4.)


@pytest.mark.parametrize(('power', 'capped'), [(2, False), (4, True)])
def test_inner_cgm_closed_form(tmp_path, monkeypatch, power, capped):
    # M(<r) = M_R * r/R; gas density is rho_R * (r/R)**(-power)
    # outside 0.01R. A second component supplies the remaining mass.
    boundaries = np.linspace(0, 1, 100_001)
    r = (boundaries[1:] + boundaries[:-1]) / 2
    gas = np.zeros_like(r)
    mask = boundaries[:-1] >= .01
    amplitude = 1e-7  # 4*pi*R**3*rho_R in native mass units.
    power_mass = 3 - power
    gas[mask] = amplitude * (boundaries[1:][mask]**power_mass
                             - boundaries[:-1][mask]**power_mass) / power_mass
    remaining = np.diff(boundaries) - gas
    assert np.all(remaining >= 0)
    # R=100 native units = 50 physical kpc; center crosses the box seam.
    center = np.array([980., 20., 30.])
    xyz = (center + r[:, None] * [100., 0., 0.]) % 1000
    path = tmp_path / 'snapshot.hdf5'
    with h5py.File(path, 'w') as f:
        f.create_group('Header').attrs.update(Time=.25, HubbleParam=.5, BoxSize=1000., Redshift=3.)
    host = dict(GroupPos=center, Group_R_Crit200=100., Group_M_Crit200=1., GroupFirstSub=7)
    sub = dict(SubhaloGrNr=2, SubhaloLenType=[len(r), 0, 0, 0, len(r), 0])
    monkeypatch.setattr(inner_cgm.il.groupcat, 'loadSingle',
                        lambda *args, **ids: sub if 'subhaloID' in ids else host)
    monkeypatch.setattr(inner_cgm.il.snapshot, 'snapPath', lambda *args: path)

    def load_members(base, snap, sid, p, fields):
        assert sid == 7 and p in (0, 4)
        return dict(Coordinates=xyz, Masses=gas if p == 0 else remaining,
                    GFM_Metallicity=np.full(len(r), .007))

    monkeypatch.setattr(inner_cgm.il.snapshot, 'loadSubhalo', load_members)
    table = tmp_path / 'z_3.000.hdf5'
    cooling = 2e-23
    with h5py.File(table, 'w') as f:
        g = f.create_group('Metal_free')
        g['Helium_mass_fraction_bins'] = [.25]
        g['Temperature_bins'] = [1e4, 1e7]
        g['Hydrogen_density_bins'] = [1e-12, 1.]
        g['Net_Cooling'] = np.full((1, 2, 2), cooling)
        f.create_group('Total_Metals')['Net_cooling'] = np.zeros((2, 2))
    result = inner_cgm.measure_inner_cgm(tmp_path, 19, 7, cooling_table_dir=tmp_path)

    # Independent CGS solution with finite outer boundary P(R)=0.
    radius = 50 * 3.085677581e21
    mass = 2e10 * 1.98847e33
    vc2 = 6.67430e-8 * mass / radius
    rho_R = amplitude * mass / (4 * np.pi * radius**3)
    kT = (2 / 3) * .62 * 1.67262192369e-24 * vc2

    def density(x):
        return rho_R * vc2 * (x**(-power) - 1) / (power * 2.3 * kT)

    cap = 8 * density(.2)
    tff = np.sqrt(2) * .1 * radius / np.sqrt(vc2)
    tcool = 1.5 * 2.3 * kT / (min(density(.1), cap) * cooling)
    print(power, 'analytic', tcool / tff, 'numeric', result.ratio)
    assert bool(result.density_capped) is capped
    assert result.enclosed_mass_fraction == pytest.approx(1.)
    assert result.temperature_K == pytest.approx(kT / 1.380649e-16, rel=.001)
    assert result.tff_Myr == pytest.approx(tff / 3.15576e13, rel=.001)
    # The production 0.05-dex shell average and midpoint quadrature
    # approximate the continuous integral; the steep profile differs by ~1.3%.
    assert result.hydrogen_density_cm3 == pytest.approx(density(.1), rel=.015)
    assert result.density_cap_cm3 == pytest.approx(cap, rel=.015)
    assert result.ratio == pytest.approx(tcool / tff, rel=.015)
