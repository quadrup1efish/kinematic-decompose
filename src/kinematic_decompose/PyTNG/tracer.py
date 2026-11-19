"""Trace fixed cohorts of kinematically labelled TNG stars through PartType3."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .illustris_python.snapshot import getNumPart, snapPath
from ..mixture.util import STRUCTURES


PHASES = ('gas', 'star', 'wind', 'black hole', 'unknown')
_PARENT_TYPES = (0, 4, 5)


def relative_positions(frame, center, *, physical=True):
    """Minimum-image offsets from a native (ckpc/h) box-frame center."""
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Each center must be one finite three-vector in ckpc/h')
    half_box = frame.box_size / 2
    offsets = (frame.coordinates - center + half_box) % frame.box_size - half_box
    return offsets * (frame.scale_factor / frame.hubble if physical else 1.0)


@dataclass(frozen=True)
class TracerCohort:
    seed_snap: int
    subhalo_id: int
    star_ids: np.ndarray
    star_labels: np.ndarray
    tracer_ids: np.ndarray
    seed_star_index: np.ndarray

    @property
    def labels(self):
        """Seed decomposition label, aligned with tracer_ids."""
        return self.star_labels[self.seed_star_index]

    def coverage(self, structure):
        """Counts of seed stars, represented seed stars, and their tracers."""
        labels = STRUCTURES[structure]
        covered = np.zeros(len(self.star_ids), dtype=bool)
        covered[self.seed_star_index] = True
        selected = np.isin(self.star_labels, labels)
        return (int(selected.sum()), int((selected & covered).sum()),
                int(np.isin(self.labels, labels).sum()))


@dataclass(frozen=True)
class TracerFrame:
    snap: int
    redshift: float
    scale_factor: float
    hubble: float
    box_size: float
    tracer_ids: np.ndarray
    parent_ids: np.ndarray
    tracer_found: np.ndarray
    parent_found: np.ndarray
    phase: np.ndarray
    coordinates: np.ndarray  # native absolute ckpc/h, NaN if parent unresolved


@dataclass(frozen=True)
class _Snapshot:
    paths: tuple
    total: np.ndarray
    redshift: float
    scale_factor: float
    hubble: float
    box_size: float


def _snapshot(base_path, snap):
    first = snapPath(str(base_path), snap)
    if not Path(first).is_file():
        raise FileNotFoundError(f'Snapshot {snap}: missing first chunk: {first}')
    with h5py.File(first, 'r') as f:
        h = f['Header'].attrs
        try:
            n = int(h['NumFilesPerSnapshot'])
            total = getNumPart(h)
            z, a, hubble, box = (float(h[k]) for k in ('Redshift', 'Time', 'HubbleParam', 'BoxSize'))
        except KeyError as exc:
            raise ValueError(f'Snapshot {snap}: incomplete Header in {first}') from exc
    paths = tuple(snapPath(str(base_path), snap, i) for i in range(n))
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'Snapshot {snap}: {len(missing)}/{n} chunks missing; first: {missing[0]}')
    if not np.isfinite([z, a, hubble, box]).all() or min(a, hubble, box) <= 0:
        raise ValueError(f'Snapshot {snap}: invalid cosmology/box metadata')
    if total[3] == 0:
        raise ValueError(f'Snapshot {snap}: no PartType3 tracers available')
    return _Snapshot(paths, total, z, a, hubble, box)


def preflight_tracer_snapshots(base_path, snapshots):
    """Fail before a scan if any requested snapshot is absent or incomplete."""
    for snap in dict.fromkeys(snapshots):
        _snapshot(base_path, snap)


def _uint_ids(ids):
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'ui' or (ids.dtype.kind == 'i' and np.any(ids < 0)):
        raise ValueError('IDs must be a one-dimensional array of nonnegative integers')
    return ids.astype(np.uint64, copy=False)


def _matches(sorted_ids, ids):
    """Return local source rows and their indices into sorted_ids."""
    ix = np.searchsorted(sorted_ids, ids)
    rows = np.flatnonzero(ix < len(sorted_ids))
    rows = rows[sorted_ids[ix[rows]] == ids[rows]]
    return rows, ix[rows]


def _blocks(meta, part_types, key, block_size):
    if block_size < 1:
        raise ValueError('block_size must be positive')
    seen = np.zeros(len(meta.total), dtype=np.int64)
    for path in meta.paths:
        with h5py.File(path, 'r') as f:
            counts = f['Header'].attrs['NumPart_ThisFile']
            for p in part_types:
                n = int(counts[p])
                group = f.get(f'PartType{p}')
                if group is None and not n:
                    continue
                if group is None or len(group[key]) != n:
                    raise ValueError(f'{path}: PartType{p}/{key} disagrees with Header')
                seen[p] += n
                for lo in range(0, n, block_size):
                    sl = slice(lo, min(lo + block_size, n))
                    yield p, group, sl, _uint_ids(group[key][sl])
    for p in part_types:
        if seen[p] != meta.total[p]:
            raise ValueError(f'Incomplete PartType{p}: scanned {seen[p]}, expected {meta.total[p]}')


def seed_tracer_cohort(base_path, seed_snap, subhalo_id, star_ids, star_labels,
                       *, structures=None, block_size=1_000_000):
    """Associate PartType3 tracers with already-classified, real seed stars.

    star_ids and star_labels must be aligned, e.g. the decomposition's iord
    and label arrays after filtering wind particles out. Structures use the
    existing label convention (disk = cold + warm, excludes counter-rotation).
    The result retains stars with zero tracers so coverage is measurable.
    """
    star_ids = _uint_ids(star_ids)
    labels = np.asarray(star_labels)
    if labels.shape != star_ids.shape or labels.dtype.kind not in 'iu' or not np.isin(labels, range(5)).all():
        raise ValueError('Provide one integer decomposition label (0–4) per seed star')
    if structures is not None:
        wanted = [label for name in structures for label in STRUCTURES[name]]
        keep = np.isin(labels, wanted)
        star_ids, labels = star_ids[keep], labels[keep]
    order = np.argsort(star_ids)
    star_ids, labels = star_ids[order], labels[order].astype(np.int8)
    if not len(star_ids) or np.any(star_ids[1:] == star_ids[:-1]):
        raise ValueError('Selected seed star IDs must be nonempty and unique')
    meta = _snapshot(base_path, seed_snap)
    tracer_ids, star_index = [], []
    for _, group, sl, ids in _blocks(meta, (3,), 'ParentID', block_size):
        rows, ix = _matches(star_ids, ids)
        if rows.size:
            tracer_ids.append(_uint_ids(group['TracerID'][sl])[rows])
            star_index.append(ix)
    ids = np.concatenate(tracer_ids) if tracer_ids else np.empty(0, dtype=np.uint64)
    indices = np.concatenate(star_index) if star_index else np.empty(0, dtype=np.intp)
    order = np.argsort(ids)
    ids = ids[order]
    if np.any(ids[1:] == ids[:-1]):
        raise ValueError('Duplicate TracerID in seed snapshot')
    return TracerCohort(seed_snap, subhalo_id, star_ids, labels, ids, indices[order])


def seed_decomposed_galaxy(base_path, seed_snap, subhalo_id, galaxy, **kwargs):
    """Use the existing decomposition's iord, aform and label arrays."""
    stars = galaxy.s
    aform = np.asarray(stars['aform'])
    real = np.isfinite(aform) & (aform > 0)
    return seed_tracer_cohort(base_path, seed_snap, subhalo_id,
                              np.asarray(stars['iord'])[real],
                              np.asarray(stars['label'])[real], **kwargs)


def _require_unique(ix, found, kind):
    if np.any(found[ix]) or len(np.unique(ix)) != len(ix):
        raise ValueError(f'Duplicate matched {kind} in snapshot')


def trace_snapshot(base_path, cohort, snap, *, block_size=1_000_000):
    """Resolve the same cohort at one earlier snapshot (or the seed itself).

    tracer_found and parent_found are distinct lookup outcomes. Unresolved
    phases are 'unknown'; parent_ids entries are only valid if tracer_found.
    No spatial or subhalo membership restriction is applied to parent lookup.
    """
    if snap > cohort.seed_snap:
        raise ValueError('Trace snapshots must not be later than the seed')
    meta = _snapshot(base_path, snap)
    n = len(cohort.tracer_ids)
    if snap == cohort.seed_snap:
        parent_ids = cohort.star_ids[cohort.seed_star_index]
        tracer_found = np.ones(n, dtype=bool)
    else:
        tracer_found = np.zeros(n, dtype=bool)
        parent_ids = np.zeros(n, dtype=np.uint64)
        if n:
            for _, group, sl, ids in _blocks(meta, (3,), 'TracerID', block_size):
                rows, ix = _matches(cohort.tracer_ids, ids)
                if rows.size:
                    _require_unique(ix, tracer_found, 'TracerID')
                    parent_ids[ix] = _uint_ids(group['ParentID'][sl])[rows]
                    tracer_found[ix] = True

    parents, inverse = np.unique(parent_ids[tracer_found], return_inverse=True)
    found = np.zeros(len(parents), dtype=bool)
    coords = np.full((len(parents), 3), np.nan)
    phase = np.full(len(parents), 'unknown', dtype='U10')
    if len(parents):
        for p, group, sl, ids in _blocks(meta, _PARENT_TYPES, 'ParticleIDs', block_size):
            rows, ix = _matches(parents, ids)
            if not rows.size:
                continue
            _require_unique(ix, found, 'parent ParticleID')
            found[ix] = True
            coords[ix] = group['Coordinates'][sl][rows]
            if p != 4:
                phase[ix] = 'gas' if p == 0 else 'black hole'
            elif 'GFM_StellarFormationTime' in group:
                aform = group['GFM_StellarFormationTime'][sl][rows]
                finite = np.isfinite(aform)
                phase[ix[finite & (aform > 0)]] = 'star'
                phase[ix[finite & (aform <= 0)]] = 'wind'

    frame_found = np.zeros(n, dtype=bool)
    frame_coords = np.full((n, 3), np.nan)
    frame_phase = np.full(n, 'unknown', dtype='U10')
    frame_found[tracer_found] = found[inverse]
    frame_coords[tracer_found] = coords[inverse]
    frame_phase[tracer_found] = phase[inverse]
    return TracerFrame(snap, meta.redshift, meta.scale_factor, meta.hubble, meta.box_size,
                       cohort.tracer_ids, parent_ids, tracer_found, frame_found, frame_phase, frame_coords)
