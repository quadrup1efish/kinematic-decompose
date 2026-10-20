"""TNG cutout loading, filename helpers and periodic snapshot centering."""
import os
import sys
import h5py
import warnings
import numpy as np
import pynbody
import pytest
import matplotlib.pyplot as plt
from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.PyTNG.cutout_loader import (
    make_cutout_filename,
    parse_cutout_filename,
    find_cutout_file,
    list_cutout_files,
    load_cutout,
)
from kinematic_decompose.PyTNG import snapshot_loader
from kinematic_decompose.PyTNG.tng_config import get_eps_mDM
from pynbody.array import SimArray
from pynbody import units


def _dm_header(redshift=0.) -> dict:
    """Synthetic TNG metadata with a non-rounded fixed DM particle mass."""
    return dict(SnapshotNumber=99, Time=1/(1+redshift), HubbleParam=.6774,
                Redshift=redshift, Omega0=.3089, OmegaLambda=.6911, BoxSize=35000.,
                NumPart_ThisFile=np.array([0, 2, 0, 0, 0, 0]),
                MassTable=np.array([0., 3.073677086264643e-5, 0., 0., 0., 0.]))


@pytest.mark.parametrize("redshift", [0., 2.])
def test_snapshot_properties_use_exact_run_mass_without_snapshot_io(tmp_path, monkeypatch, redshift):
    header = _dm_header(redshift)
    output = tmp_path / 'TNG50-1' / 'output'
    # Only normal group-catalog metadata is needed; no snapshot chunk is opened.
    group_header = {k: v for k, v in header.items() if k != 'MassTable'}
    monkeypatch.setattr(snapshot_loader.il.groupcat, 'loadHeader', lambda *args: group_header)
    def unexpected_snapshot_access(*args):
        raise AssertionError('DM mass lookup must not open a snapshot chunk')
    monkeypatch.setattr(snapshot_loader.il.snapshot, 'snapPath', unexpected_snapshot_access)
    snapshot = Snapshot(str(output), 99)
    assert float(snapshot.properties['mDM']) == header['MassTable'][1]
    assert 'mDM_source' not in snapshot.properties
    assert float(snapshot.properties['mDM'].in_units('Msol', h=.6774)) == pytest.approx(
        header['MassTable'][1] * 1e10 / .6774)
    assert float(snapshot.properties['eps']) == pytest.approx(.39 if redshift > 1 else .195)


def test_cutout_dm_particles_use_exact_header_mass(tmp_path):
    header = _dm_header()
    path = tmp_path / 'snapNum_99_subID_42_fields_Coordinates.hdf5'
    with h5py.File(path, 'w') as handle:
        for key, value in header.items():
            handle.require_group('Header').attrs[key] = value
        handle.create_group('PartType1').create_dataset('Coordinates', data=np.ones((2, 3)))
    container, _ = load_cutout(str(path), run='TNG50-1')
    np.testing.assert_array_equal(container.dm['mass'],
                                  np.full(2, header['MassTable'][1], dtype=np.float32))
    assert 'mDM_source' not in container.properties
    container.physical_units()
    np.testing.assert_allclose(container.dm['mass'], header['MassTable'][1] * 1e10 / .6774,
                               rtol=2e-7)


def test_dm_mass_run_lookup_requires_no_header_or_fallback():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, mass = get_eps_mDM(dict(run='TNG50-1', Redshift=0.))
    assert float(mass) == 3.073677086264643e-5
    assert not caught


def test_cutout_without_mass_table_uses_same_exact_run_constant(tmp_path):
    from kinematic_decompose.PyTNG.cutout_loader import header_to_properties
    with h5py.File(tmp_path / 'legacy.hdf5', 'w') as handle:
        header = handle.create_group('Header')
        for key, value in _dm_header().items():
            if key != 'MassTable':
                header.attrs[key] = value
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            props = header_to_properties(header, 'TNG50-1', str(tmp_path))
    assert 'mDM_source' not in props
    assert float(props['mDM']) == 3.073677086264643e-5
    assert not caught


@pytest.mark.parametrize('run', ['TNG50', 'TNG50-9', 'TNG100-1-Dark'])
def test_dm_mass_unknown_run_is_not_guessed(run):
    with pytest.raises(ValueError, match='Unsupported simulation run'):
        get_eps_mDM(dict(run=run, Redshift=0.))


@pytest.mark.parametrize('redshift', [0., 2.])
@pytest.mark.parametrize('run,mass', [
    # Independent published values from each official downloads/<run>/ page,
    # in 1e10 Msun/h. Metadata formatting can omit the last Header digits.
    ('TNG50-1', 3.07367708626464e-05),
    ('TNG50-2', 0.000245894166901171),
    ('TNG50-3', 0.00196715333520937),
    ('TNG50-4', 0.015737226681675),
    ('TNG100-1', 0.000505574296436975),
    ('TNG100-2', 0.0040445943714958),
    ('TNG100-3', 0.0323567549719664),
    ('TNG300-1', 0.00398342749867548),
    ('TNG300-2', 0.0318674199894039),
    ('TNG300-3', 0.254939359915231),
    ('TNG-Cluster', 0.00413211373831018),
])
def test_all_run_masses_match_official_metadata(run, mass, redshift):
    _, actual = get_eps_mDM(dict(run=run, Redshift=redshift))
    np.testing.assert_allclose(float(actual), mass, rtol=1e-14, atol=0.)
    assert actual.units == units.Unit('1e10 Msol h^-1')


@pytest.mark.parametrize('run,mass,path', [
    ('TNG50-1', 3.073677086264643e-5,
     '/Users/yuwa/sims.TNG/TNG50-1/cutouts/snapNum_99_subID_715286_fields_Coordinates_Velocities_Masses.hdf5'),
    ('TNG100-1', 0.0005055742964369746,
     '/Users/yuwa/sims.TNG/TNG100-1/cutouts/snapNum_99_subID_472362_fields_Coordinates_Velocities_Masses.hdf5'),
    ('TNG100-3', 0.03235675497196638,
     '/Users/yuwa/sims.TNG/TNG100-3/output/snapdir_099/snap_099.0.hdf5'),
])
def test_exact_run_constants_match_independent_snapshot_headers(run, mass, path):
    _, actual = get_eps_mDM(dict(run=run, Redshift=0.))
    assert float(actual) == mass
    if not os.path.isfile(path):
        pytest.skip('Independent local TNG Header not available')
    with h5py.File(path, 'r') as handle:
        assert float(actual) == float(handle['Header'].attrs['MassTable'][1])


# Example load cutout

"""
Example: load a TNG cutout HDF5 file via the PyTNG Snapshot interface.

Data: /Users/yuwa/sims.TNG/TNG50-1/cutouts/
    snapNum_99_subID_550475_fields_Coordinates_Velocities_Masses.hdf5

The cutout contains a single subhalo (IDs 550475) at snapshot 99 with
gas + dm + star + bh particles, so it can be loaded without the full
snapshot directory (no output/, no loadHeader) — the header metadata
comes from the cutout file itself.
"""




CUTOUT_DIR = "/Users/yuwa/sims.TNG/TNG50-1/cutouts"
SUB_ID = 550475
SNAPNUM = 99


def test_filename_helpers():
    """Naming convention round-trip + legacy parsing."""
    std = make_cutout_filename(SNAPNUM, SUB_ID,
                               ["Coordinates", "Velocities", "Masses"])
    assert std == "snapNum_99_subID_550475_fields_Coordinates_Velocities_Masses.hdf5", std

    p = parse_cutout_filename(std)
    assert p["snapNum"] == SNAPNUM and p["subID"] == SUB_ID
    assert p["fields"] == ["Coordinates", "Velocities", "Masses"], p

    # legacy names still parse
    legacy = parse_cutout_filename("sub550475_snap99_pos_vel_mass.hdf5")
    assert legacy["snapNum"] == 99 and legacy["subID"] == 550475, legacy
    assert legacy["fields"] == ["Coordinates", "Velocities", "Masses"], legacy

    legacy_full = parse_cutout_filename("sub472362_snap99_cutout_full.hdf5")
    assert legacy_full["fields"] is None, legacy_full  # load everything

    # directory scan
    hits = list_cutout_files(CUTOUT_DIR)
    assert len(hits) > 0, "no cutout files found in %s" % CUTOUT_DIR
    print("cutout files found:")
    for h in hits:
        print("  ", os.path.basename(h[3]), "->", h[0], h[1], h[2])

    found = find_cutout_file(CUTOUT_DIR, SNAPNUM, SUB_ID)
    assert os.path.basename(found) == std, found
    print("find_cutout_file ->", os.path.basename(found))
    print("filename helpers: OK")


def test_load_cutout():
    """Load the full cutout through the Snapshot interface."""
    snapshot = Snapshot(basePath="/Users/yuwa/sims.TNG/TNG50-1/output",
                        snapNum=SNAPNUM, header_source="cutout")
    container = snapshot.load_cutout(SUB_ID, cutout_dir=CUTOUT_DIR)

    print("\nproperties:")
    for k in ["run", "a", "h", "Redshift", "omegaM0", "omegaL0",
              "boxsize", "mDM", "eps"]:
        print(f"  {k} = {container.properties[k]}")

    print("\nfamilies loaded:")
    for fam in ["gas", "dm", "star", "bh"]:
        part = getattr(container, fam)
        print(f"  {fam:>4}: {len(part)}")

    # sanity: particle numbers must match the file header
    assert len(container.gas) == 363567
    assert len(container.dm) == 1961772
    assert len(container.star) == 610401
    assert len(container.bh) == 1

    # sanity: fields mapped to pynbody aliases
    for fam in ["gas", "star", "dm", "bh"]:
        part = getattr(container, fam)
        assert "pos" in part and "vel" in part and "mass" in part, fam
        print(f"  {fam}: pos{part['pos'].shape} {part['pos'].units}")

    container.physical_units()
    print("\nafter physical_units:")
    print("  star pos[0] =", container.s["pos"][0], container.s["pos"].units)
    print("  star r[0:3] =", container.s["r"][:3])
    print("  star j[0]   =", container.s["j"][0])

    # derived arrays / simdict getters work (register side effects)
    assert np.isfinite(container.s["r"]).all()
    assert np.isfinite(container.s["j"]).all()
    print("  t =", container.properties["t"], "(SimDict getter works)")

    # limitation: only the fields present in the cutout are loaded, so
    # derived quantities that need extra datasets (u, ElectronAbundance,
    # GFM_*) raise KeyError instead of silently being wrong.
    try:
        container.g["temp"]
        raise AssertionError("temp should not be derivable without u/el. abund.")
    except KeyError:
        print("  temp (needs u+ElectronAbundance): KeyError as expected")

    # galaxy view via __getitem__
    gv = snapshot[SUB_ID]
    assert len(gv) == len(container)
    print("\n__getitem__ galaxy view OK:", len(gv), "particles")
    print("load_cutout: OK")


if __name__ == "__main__":
    test_filename_helpers()
    test_load_cutout()

    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        snapshot = Snapshot(basePath="/Users/yuwa/sims.TNG/TNG50-1/output",
                            snapNum=SNAPNUM, header_source="cutout")
        snapshot.load_cutout(SUB_ID, cutout_dir=CUTOUT_DIR)
        snapshot.center()
        snapshot.faceon(align_with="star", as_context=False)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, fam, label in zip(axes, ["star", "dm"],
                                  ["stars", "dark matter"]):
            img = pynbody.plot.image(getattr(snapshot.container, fam),
                                     axes=ax, width="40 kpc",
                                     units="Msol kpc^-2", cmap="bone",
                                     title=label, return_array=True)
            assert img is not None, f"image render failed for {fam}"
        plt.tight_layout()
        out_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, "images"))
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "example_load_cutout.png")
        plt.savefig(out, dpi=150)
        print("saved:", out)


# Test snapshot center periodic

def _snapshot_with_positions(positions, boxsize=None):
    container = pynbody.new(star=len(positions))
    container.star["pos"] = SimArray(np.asarray(positions, dtype=float), units.kpc)
    if boxsize is not None:
        container.properties["boxsize"] = SimArray(float(boxsize), units.kpc)
    snapshot = Snapshot.__new__(Snapshot)
    snapshot.container = container
    snapshot.group_catalog = {}
    return snapshot


def test_explicit_center_wraps_particles_across_periodic_boundary():
    snapshot = _snapshot_with_positions(
        [[0.2, 5.0, 5.0], [9.7, 5.0, 5.0], [0.1, 5.2, 5.0]], boxsize=10.0
    )

    snapshot.center(cen=SimArray([9.8, 5.0, 5.0], units.kpc), with_velocity=False)

    positions = np.asarray(snapshot.container.star["pos"])
    np.testing.assert_allclose(positions[:, 0], [0.4, -0.1, 0.3], atol=1e-6)
    assert np.max(np.linalg.norm(positions, axis=1)) < 1.0
    assert np.isclose(np.linalg.norm(positions[0] - positions[1]), 0.5)


def test_explicit_center_keeps_nonperiodic_translation_when_boxsize_missing():
    snapshot = _snapshot_with_positions([[12.0, 0.0, 0.0]])

    snapshot.center(cen=SimArray([10.0, 0.0, 0.0], units.kpc), with_velocity=False)

    np.testing.assert_allclose(snapshot.container.star["pos"], [[2.0, 0.0, 0.0]])


def test_automatic_center_still_uses_pynbody_wrap(monkeypatch):
    snapshot = _snapshot_with_positions([[1.0, 0.0, 0.0]], boxsize=10.0)
    calls = {}

    def fake_center(container, **kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(pynbody.analysis, "center", fake_center)
    snapshot.center(cen=None, with_velocity=False)

    assert calls["wrap"] is True
    assert calls["with_velocity"] is False
