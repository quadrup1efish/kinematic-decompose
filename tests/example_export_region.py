"""Export a complete spatial gas region on the Lab, then download/verify locally.

Run ``python tests/example_export_region.py --help``. Export defaults to a bounded
coordinate-I/O preflight; --run explicitly starts the full serial snapshot scan.
No scientific gas cuts are applied. This script never installs dependencies.
"""
import argparse
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

# Set before importing NumPy/pynbody on the shared Lab server.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import h5py
import numpy as np

from kinematic_decompose.PyTNG import illustris_python as il
from kinematic_decompose.PyTNG.snapshot_loader import Snapshot


GAS_FIELDS = ['Coordinates', 'Velocities', 'Masses', 'ParticleIDs', 'Density',
              'InternalEnergy', 'ElectronAbundance', 'StarFormationRate',
              'GFM_Metals']
OPTIONAL_GAS_FIELDS = {'GFM_Metals'}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def target_catalog(base_path, snap_num, subhalo_id, radius_factor):
    subhalo = il.groupcat.loadSingle(base_path, snap_num, subhaloID=subhalo_id)
    halo_id = int(subhalo['SubhaloGrNr'])
    group = il.groupcat.loadSingle(base_path, snap_num, haloID=halo_id)
    half_size = radius_factor * float(group['Group_R_Crit200'])
    if half_size <= 0:
        raise ValueError('Target host has no positive R200c')
    return subhalo, group, half_size


def preflight(base_path, snap_num, center, half_size, sample_files=2, block_size=500_000):
    """Read a few full coordinate chunks; extrapolate coordinate scan time only."""
    with h5py.File(il.snapshot.snapPath(base_path, snap_num), 'r') as f:
        header = dict(f['Header'].attrs)
        dtype = f['PartType0/Coordinates'].dtype
        available = set(f['PartType0'])
        missing = set(GAS_FIELDS) - OPTIONAL_GAS_FIELDS - available
        if missing:
            raise KeyError(f'Missing required gas fields: {sorted(missing)}')
        fields = [name for name in GAS_FIELDS if name in available]
    num_files = int(header['NumFilesPerSnapshot'])
    total_gas = int(il.snapshot.getNumPart(header)[0])
    total_bytes = total_gas * 3 * dtype.itemsize
    rows, selected = 0, 0
    start = time.monotonic()
    for chunk in range(min(sample_files, num_files)):
        with h5py.File(il.snapshot.snapPath(base_path, snap_num, chunk), 'r') as f:
            coordinates = f['PartType0/Coordinates']
            for offset in range(0, len(coordinates), block_size):
                pos = coordinates[offset:offset + block_size]
                delta = pos - center
                delta -= header['BoxSize'] * np.rint(delta / header['BoxSize'])
                selected += int(np.count_nonzero(np.all(abs(delta) <= half_size, axis=1)))
                rows += len(pos)
    elapsed = time.monotonic() - start
    result = dict(snapshot=snap_num, redshift=float(header['Redshift']),
                  num_files=num_files, total_gas=total_gas,
                  coordinate_bytes=total_bytes, tested_rows=rows,
                  tested_selected_gas=selected, test_seconds=elapsed,
                  fields=fields, missing_optional_fields=sorted(OPTIONAL_GAS_FIELDS - available),
                  estimated_coordinate_scan_seconds=elapsed * total_gas / max(rows, 1),
                  note='Coordinate-only extrapolation; NOT a full export time or region-size estimate')
    print(json.dumps(result, indent=2), flush=True)
    return result


def nearby_subhalos(base_path, snap_num, center, half_size, properties):
    names = ['SubhaloPos', 'SubhaloVel', 'SubhaloHalfmassRadType', 'SubhaloGrNr']
    arrays = il.groupcat.loadSubhalos(base_path, snap_num, fields=names)
    delta = arrays['SubhaloPos'] - center
    box = float(properties['boxsize'])
    delta -= box * np.rint(delta / box)
    # Include centres outside the cube if the diagnostic exclusion sphere overlaps.
    radii = np.maximum(2 * arrays['SubhaloHalfmassRadType'][:, 0],
                       5 * properties['h'] / properties['a'])
    distance_to_box = np.maximum(abs(delta) - half_size, 0)
    mask = np.sum(distance_to_box**2, axis=1) <= radii**2
    return {'SubhaloID': np.flatnonzero(mask),
            **{name: arrays[name][mask] for name in names}}


def export_region(args):
    base_path = str(Path(args.base_path).expanduser())
    output = Path(args.output).expanduser()
    # Never spend a full scan discovering that an output already exists.
    for path in (output, Path(str(output) + '.part'), Path(str(output) + '.json')):
        if path.exists():
            raise FileExistsError(path)
    subhalo, group, half_size = target_catalog(
        base_path, args.snap, args.subhalo, args.radius_factor)
    center = np.asarray(subhalo['SubhaloPos'], dtype=float)
    print(json.dumps(dict(subhalo=args.subhalo, host=int(subhalo['SubhaloGrNr']),
                          center_ckpc_h=center.tolist(), half_size_ckpc_h=half_size)),
          flush=True)
    inspection = preflight(base_path, args.snap, center, half_size, args.sample_files)
    if not args.run:
        print('Preflight only: pass --run to export the full region.', flush=True)
        return
    snapshot = Snapshot(base_path, args.snap, header_source='snapshot')
    start = time.monotonic()

    def progress(chunk, total, counts):
        print(json.dumps(dict(chunk=chunk, total_chunks=total, selected=counts,
                              elapsed_seconds=time.monotonic() - start)), flush=True)
        if counts['gas'] > args.max_selected_gas:
            raise MemoryError('Selected gas exceeds --max-selected-gas; no file published')

    snapshot.load_region(center, half_size, {'gas': inspection['fields']},
                         block_size=args.block_size, progress=progress)
    catalog = {'Target': {'SubhaloID': args.subhalo,
                          'HaloID': int(subhalo['SubhaloGrNr'])},
               'Subhalo': subhalo, 'Group': group,
               'NearbySubhalos': nearby_subhalos(
                   base_path, args.snap, center, half_size, snapshot.properties)}
    snapshot.save_region(output, catalog=catalog)
    count = len(snapshot.gas)
    del snapshot
    restored = Snapshot.from_region(output)  # Reopen and read every dataset.
    assert len(restored.gas) == count
    manifest = dict(file=output.name, bytes=output.stat().st_size,
                    sha256=sha256(output), gas_count=count, complete=True,
                    snapshot=args.snap, subhalo=args.subhalo,
                    fields=inspection['fields'],
                    missing_optional_fields=inspection['missing_optional_fields'],
                    elapsed_seconds=time.monotonic() - start)
    with open(str(output) + '.json', 'x') as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


def download_region(lab_base, remote_path, cookie_jar, output):
    """Stream raw bytes with resume; verify the remote manifest and all HDF5 data.

    The cookie jar is read locally, never printed or copied into command arguments.
    Login is deliberately not implemented here; reuse an authorized Lab session.
    """
    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(output)
    cookies = http.cookiejar.MozillaCookieJar(str(Path(cookie_jar).expanduser()))
    cookies.load(ignore_discard=True, ignore_expires=False)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
    url = lab_base.rstrip('/') + '/files/' + urllib.parse.quote(remote_path.lstrip('/'), safe='/@')
    with opener.open(url + '.json', timeout=60) as response:
        try:
            manifest = json.load(response)
        except (ValueError, UnicodeError):
            raise RuntimeError('Lab did not return a JSON manifest; check authentication') from None
    if not manifest.get('complete') or manifest.get('file') != Path(remote_path).name:
        raise ValueError('Remote manifest does not describe a completed target file')
    partial = Path(str(output) + '.part')
    output.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > manifest['bytes']:
        raise ValueError('Local partial file is larger than the remote export')
    if offset < manifest['bytes']:
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        with opener.open(request, timeout=120) as response:
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise RuntimeError('Lab returned a login page, not HDF5 bytes')
            if offset and (response.status != 206 or not response.headers.get(
                    'Content-Range', '').startswith(f'bytes {offset}-')):
                raise RuntimeError('Server refused the requested resume offset')
            with open(partial, 'ab' if offset else 'xb') as stream:
                for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                    stream.write(block)
    if partial.stat().st_size != manifest['bytes'] or sha256(partial) != manifest['sha256']:
        raise ValueError('Downloaded byte size or SHA-256 does not match remote manifest')
    restored = Snapshot.from_region(partial)
    if len(restored.gas) != manifest['gas_count']:
        raise ValueError('Downloaded HDF5 gas count does not match remote manifest')
    del restored
    os.link(partial, output)  # Atomic publication without overwriting.
    partial.unlink()
    with open(str(output) + '.json', 'x') as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(dict(local_file=str(output), **manifest), indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='Run on the TNG Lab')
    export.add_argument('--base-path', default='/home/tnguser/sims.TNG/TNG50-1/output')
    export.add_argument('--snap', type=int, default=19)
    export.add_argument('--subhalo', type=int, default=11184)
    export.add_argument('--radius-factor', type=float, default=5.)
    export.add_argument('--output', default='TNG50-1_snap019_sub11184_region_5R200c.hdf5')
    export.add_argument('--block-size', type=int, default=500_000)
    export.add_argument('--sample-files', type=int, default=2)
    export.add_argument('--max-selected-gas', type=int, default=5_000_000,
                        help='Memory guard on selected gas, not a scientific cut')
    export.add_argument('--run', action='store_true')
    download = commands.add_parser('download', help='Run locally with an authorized Lab cookie jar')
    download.add_argument('--lab-base', required=True, help='Authorized user-server base URL')
    download.add_argument('--remote-path', required=True, help='Export file path relative to Lab home')
    download.add_argument('--cookie-jar', required=True)
    download.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'export':
        if args.sample_files <= 0 or args.radius_factor <= 0 or args.block_size <= 0:
            parser.error('Sample files, radius factor and block size must be positive')
        export_region(args)
    else:
        download_region(args.lab_base, args.remote_path, args.cookie_jar, args.output)


if __name__ == '__main__':
    main()
