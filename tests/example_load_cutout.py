"""
Example: load a TNG cutout HDF5 file via the PyTNG Snapshot interface.

Data: /Users/yuwa/sims.TNG/TNG50-1/cutouts/
    snapNum_99_subID_550475_fields_Coordinates_Velocities_Masses.hdf5

The cutout contains a single subhalo (IDs 550475) at snapshot 99 with
gas + dm + star + bh particles, so it can be loaded without the full
snapshot directory (no output/, no loadHeader) — the header metadata
comes from the cutout file itself.
"""

import os
import sys
import numpy as np

import pynbody
import matplotlib.pyplot as plt

from kinematic_decompose.PyTNG.snapshot_loader import Snapshot
from kinematic_decompose.PyTNG.cutout_loader import (
    make_cutout_filename,
    parse_cutout_filename,
    find_cutout_file,
    list_cutout_files,
)

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