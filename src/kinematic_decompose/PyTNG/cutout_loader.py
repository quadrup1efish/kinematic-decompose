"""
cutout_loader.py
================
Read TNG cutout HDF5 files (downloaded from the TNG web API or produced by
the local cutout service) and load them into a pynbody SimSnap through the
same PyTNG conventions as `load_particle`.

Naming convention (standard)
---------------------------
snapNum_{snapNum}_subID_{subID}_fields_{field1}_{field2}_...hdf5

Example::

    snapNum_99_subID_550475_fields_Coordinates_Velocities_Masses.hdf5

The field segment uses the **raw** dataset names stored inside the cutout
file (e.g. ``Coordinates``, ``Velocities``, ``Masses``, ``ParticleIDs``,
``GFM_StellarFormationTime``), not the pynbody aliases.

Legacy naming is also understood for backward compatibility::

    sub550475_snap99_pos_vel_mass.hdf5
    sub472362_snap99_cutout_full.hdf5

Layout of a cutout file::

    /
    ├── Header/            # simulation metadata (Time, HubbleParam, ...)
    ├── PartType0/         # gas   : Coordinates(f64), Masses(f32), Velocities(f32)
    ├── PartType1/         # dm    : Coordinates(f64), Velocities(f32)  [no Masses]
    ├── PartType4/         # star  : Coordinates(f64), Masses(f32), Velocities(f32)
    └── PartType5/         # bh    : Coordinates(f64), Masses(f32), Velocities(f32)

Key differences with `Snapshot.load_particle`:

* particle data comes from a single self-contained cutout file instead of
  the full snapshot directory (no `output/`, no `getSnapOffsets`);
* the simulation header is read from the cutout's own `Header` group, so
  the loader works even when the local full snapshot is incomplete;
* only one subhalo is loaded per call (no multi-ID merging), and the
  in-container galaxy index reduces to a single full-range slice.

Units and field aliases follow `tng_config` exactly, so everything
downstream (derived arrays, filters, physical_units, ...) behaves
identically to the full-snapshot path.
"""

import re
import warnings
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import h5py

import pynbody
from pynbody.array import SimArray
from pynbody.simdict import SimDict

from .tng_config import (
    PARTICLE_DTYPE,
    UnitComvingLength,
    enforce_dtype,
    get_eps_mDM,
    get_particle_field_name,
    get_particle_field_unit,
)

# ---------------------------------------------------------------------------
# Filename parsing / matching
# ---------------------------------------------------------------------------

# Standard (new) naming:  snapNum_99_subID_550475_fields_Coordinates_Velocities_Masses.hdf5
_STANDARD_RE = re.compile(
    r"^snapNum_(\d+)_subID_(\d+)_fields_(.+)\.hdf5$"
)
# Legacy naming:         sub550475_snap99_pos_vel_mass.hdf5
#                        sub472362_snap99_cutout_full.hdf5
_LEGACY_RE = re.compile(
    r"^sub(\d+)_snap(\d+)_(.+)\.hdf5$"
)

# legacy short-name segment -> raw dataset names
_LEGACY_FIELD_ALIASES = {
    "pos": "Coordinates",
    "vel": "Velocities",
    "mass": "Masses",
    "coord": "Coordinates",
}


def parse_cutout_filename(filename: str) -> Optional[Dict]:
    """Parse a cutout filename into its components.

    Returns a dict with keys ``snapNum``, ``subID``, ``fields`` (list of raw
    dataset names, or ``None`` for the legacy ``cutout_full`` marker which
    means "load every dataset present in the file"). Returns ``None`` when
    the name does not look like a cutout file.
    """
    name = str(filename).rsplit("/", 1)[-1]

    m = _STANDARD_RE.match(name)
    if m:
        snap_num = int(m.group(1))
        sub_id = int(m.group(2))
        fields = [seg for seg in m.group(3).split("_") if seg]
        return {"snapNum": snap_num, "subID": sub_id, "fields": fields}

    m = _LEGACY_RE.match(name)
    if m:
        sub_id = int(m.group(1))
        snap_num = int(m.group(2))
        seg = m.group(3)
        if seg in ("cutout_full", "full"):
            fields = None  # load everything present
        else:
            fields = [
                _LEGACY_FIELD_ALIASES.get(s, s)
                for s in seg.split("_")
                if s
            ]
        return {"snapNum": snap_num, "subID": sub_id, "fields": fields}

    return None


def make_cutout_filename(snapNum: int, subID: int, fields: List[str],
                         with_directory: str = "") -> str:
    """Build a standard cutout filename.

    ``fields`` must be the raw dataset names (``Coordinates``, ...).
    The segment order is preserved as given.
    """
    seg = "_".join(str(f) for f in fields)
    name = f"snapNum_{snapNum}_subID_{subID}_fields_{seg}.hdf5"
    if with_directory:
        return str(with_directory).rstrip("/") + "/" + name
    return name


def list_cutout_files(cutout_dir: str) -> List[Tuple[int, int, List, str]]:
    """List every parseable cutout file under *cutout_dir*.

    Returns ``[(snapNum, subID, fields, full_path), ...]`` sorted by
    (snapNum, subID). Files that do not parse are skipped silently.
    """
    import os

    hits = []
    if not os.path.isdir(cutout_dir):
        return hits
    for entry in sorted(os.listdir(cutout_dir)):
        parsed = parse_cutout_filename(entry)
        if parsed is None:
            continue
        hits.append((
            parsed["snapNum"],
            parsed["subID"],
            parsed["fields"],
            os.path.join(cutout_dir, entry),
        ))
    return hits


def find_cutout_file(cutout_dir: str, snapNum: int, subID: int,
                     fields: Optional[List[str]] = None) -> str:
    """Locate the cutout file for a (snapNum, subID) pair.

    Prefers the standard name; falls back to scanning for a legacy name
    (first match with same snapNum+subID). Raises ``FileNotFoundError``
    when nothing matches.
    """
    import os

    if fields is not None:
        std = os.path.join(
            cutout_dir, make_cutout_filename(snapNum, subID, fields))
        if os.path.isfile(std):
            return std

    hits = []
    if os.path.isdir(cutout_dir):
        for entry in sorted(os.listdir(cutout_dir)):
            parsed = parse_cutout_filename(entry)
            if parsed is None:
                continue
            if parsed["snapNum"] == snapNum and parsed["subID"] == subID:
                hits.append(os.path.join(cutout_dir, entry))
    if hits:
        if len(hits) > 1:
            warnings.warn(
                f"Multiple cutout files match snap={snapNum} sub={subID}: "
                f"{hits}. Using the first one.",
                RuntimeWarning,
            )
        return hits[0]

    raise FileNotFoundError(
        f"No cutout file for snapNum={snapNum}, subID={subID} in "
        f"{cutout_dir}"
    )


# ---------------------------------------------------------------------------
# Header -> pynbody properties
# ---------------------------------------------------------------------------

_PT_TO_FAMILY = {0: "gas", 1: "dm", 4: "star", 5: "bh"}


def header_to_properties(header: h5py.Group, run: str,
                         filedir: str) -> SimDict:
    """Build a SimDict of simulation properties from a cutout Header group.

    Mirrors ``Snapshot._set_snapshot_properties`` but reads the metadata
    from the cutout file itself instead of ``il.groupcat.loadHeader``.
    The run name (TNG50-1, ...) must be passed explicitly because the
    cutout Header only stores the internal simulation name
    (``L35n2160TNG``), which cannot be mapped to a run key.
    """
    props = SimDict()
    props["filedir"] = filedir
    props["Snapshot"] = int(header.attrs["SnapshotNumber"])
    props["run"] = run
    props.update({
        "a": header.attrs["Time"],
        "h": header.attrs["HubbleParam"],
        "Redshift": header.attrs["Redshift"],
        "omegaM0": header.attrs["Omega0"],
        "omegaL0": header.attrs["OmegaLambda"],
        "boxsize": SimArray(header.attrs["BoxSize"], UnitComvingLength),
    })

    eps, mDM = get_eps_mDM(props)
    props["mDM"] = mDM
    props["eps"] = eps
    props["standard_units"] = [
        "nH", "Halpha", "em", "ne", "temp", "mu", "c_n_sq", "p", "cs",
        "c_s", "acc", "phi", "age", "tform", "SubhaloPos", "sfr",
    ]
    return props


def infer_run_from_dir(cutout_dir: str, basePath: Optional[str] = None) -> str:
    """Infer the run name from a directory path.

    Priority: explicit ``basePath`` (its last meaningful component), then
    the parent of the ``cutouts`` directory, then the last component of
    ``cutout_dir`` itself.
    """
    import os

    path = os.path.abspath(cutout_dir)
    parts = path.rstrip("/").split("/")

    if basePath:
        bp = basePath.rstrip("/").split("/")
        if bp[-1] in ("output", "cutouts"):
            return bp[-2]
        return bp[-1]

    if parts[-1] == "cutouts":
        return parts[-2]
    return parts[-1]


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------


def load_cutout(cutout_path: str, *, run: Optional[str] = None,
                fields: Optional[List[str]] = None,
                float32: bool = True):
    """Load a cutout file into a fresh pynbody SimSnap.

    Parameters
    ----------
    cutout_path : str
        Path to the ``.hdf5`` cutout file.
    run : str, optional
        Run key used by ``get_eps_mDM`` (TNG50-1, TNG100-1, ...). Inferred
        from the directory structure when omitted.
    fields : list of str, optional
        Raw dataset names to load (e.g. ``['Coordinates', 'Velocities',
        'Masses']``). Defaults to parsing the filename; ``None`` means
        "load every dataset present".
    float32 : bool
        Down-cast float64 coordinates to float32 to match the full-snapshot
        path (``loadSubset(..., float32=True)``).

    Returns
    -------
    container : pynbody.SimSnap
    info : dict
        Metadata about the file: ``path``, ``snapNum``, ``subID``,
        ``fields``, ``NumPart``, and the matched filename components.
    """
    import os

    parsed = parse_cutout_filename(cutout_path)
    filename_fields = parsed["fields"] if parsed else None
    if fields is None or (isinstance(fields, list) and len(fields) == 0):
        fields = filename_fields  # may be None -> load everything

    with h5py.File(cutout_path, "r") as f:
        header = cast(h5py.Group, f["Header"])

        if run is None:
            run = infer_run_from_dir(os.path.dirname(cutout_path))

        props = header_to_properties(header, run=run,
                                     filedir=os.path.dirname(cutout_path))

        num_part = np.asarray(header.attrs["NumPart_ThisFile"], dtype=int)

        # Build the empty container with the same parts as the file.
        new_kwargs = {}
        order = []
        for pt, family in _PT_TO_FAMILY.items():
            count = int(num_part[pt])
            if count > 0:
                new_kwargs[family] = count
                order.append(family)
        if not new_kwargs:
            raise ValueError(f"No particles in cutout {cutout_path}")
        new_kwargs["order"] = ",".join(order)
        container = pynbody.new(**new_kwargs)

        for i in props:
            if isinstance(props[i], SimArray):
                props[i].sim = container
        container.properties = props

        family_map = {
            "star": container.s, "gas": container.g,
            "dm": container.dm, "bh": container.bh,
        }
        loaded_num = {}
        for pt, family in _PT_TO_FAMILY.items():
            count = int(num_part[pt])
            if count == 0:
                continue
            group = cast(h5py.Group, f[f"PartType{pt}"])
            fam = family_map[family]
            loaded_num[family] = count

            for ds_name_raw in group:
                if not isinstance(ds_name_raw, str):
                    continue
                ds_name = ds_name_raw
                # field filtering: respect requested fields, honour the
                # legacy 'cutout_full' marker (fields=None -> everything)
                if fields is not None and ds_name not in fields:
                    continue
                if ds_name == "Masses" and family == "dm":
                    # DM masses are never stored in TNG snapshots/cutouts;
                    # filled from the header / run table below.
                    continue

                try:
                    field_name = get_particle_field_name(ds_name)
                    field_unit = get_particle_field_unit(ds_name)
                except KeyError:
                    warnings.warn(
                        f"Skipping dataset '{ds_name}' with no registered "
                        f"name/unit mapping in tng_config.",
                        RuntimeWarning,
                    )
                    continue

                ds = cast(h5py.Dataset, group[ds_name])
                data = ds[...]
                if float32 and data.dtype != PARTICLE_DTYPE:
                    data = data.astype(PARTICLE_DTYPE)

                if field_name not in fam:
                    ndim = 1 if data.ndim == 1 else data.shape[1]
                    fam._create_array(field_name, ndim, data.dtype)
                    fam[field_name].units = field_unit
                fam[field_name][:] = data
                # pynbody.new() pre-creates pos/vel/mass arrays with
                # NoUnit(); the guard above is then skipped, so enforce
                # the mapping unit after assignment (same as load_particle).
                if fam[field_name].units != field_unit:
                    fam[field_name].units = field_unit

        # DM mass: identical logic to load_particle — kept in the project
        # particle dtype (PARTICLE_DTYPE); KDTree requires pos/mass to
        # share a dtype.
        if "dm" in family_map and loaded_num.get("dm", 0) > 0:
            dm_mass = np.full(loaded_num["dm"], container.properties["mDM"],
                              dtype=PARTICLE_DTYPE)
            container.dm["mass"] = SimArray(
                dm_mass, units=get_particle_field_unit("Masses"))

        # Belt-and-braces: whatever arrays ended up on the container, force
        # them all to the project particle dtype. This is what guarantees
        # KDTree (pos vs mass) compatibility regardless of file dtypes.
        enforce_dtype(container, PARTICLE_DTYPE)

        snap_num = int(header.attrs["SnapshotNumber"])
        sub_id = int(parsed["subID"]) if parsed else None

        info = {
            "path": os.path.abspath(cutout_path),
            "snapNum": snap_num,
            "subID": sub_id,
            "fields": fields,
            "NumPart": {fam: int(num_part[pt])
                        for pt, fam in _PT_TO_FAMILY.items()
                        if int(num_part[pt]) > 0},
        }
        return container, info


# ---------------------------------------------------------------------------
# CLI smoke check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    container, info = load_cutout(path)
    print(f"Loaded: {info['path']}")
    print(f"  snapNum={info['snapNum']}  subID={info['subID']}  "
          f"fields={info['fields']}")
    for fam, n in info["NumPart"].items():
        print(f"  {fam:>5}: {n} particles")
    container.physical_units()
    print("  r[0:3] =", container.s["r"][:3])