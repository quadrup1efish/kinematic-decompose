"""Self-contained spatial-region files in native TNG units (not halo cutouts)."""
from pathlib import Path

import h5py
import numpy as np
import pynbody
from pynbody.array import SimArray

from .cutout_loader import header_to_properties
from .illustris_python.util import partTypeNum
from .tng_config import (UnitComvingLength, enforce_dtype,
                         get_particle_field_name, get_particle_field_unit)


FORMAT = 'PyTNG-region-v1'


def save_region(container, filename, catalog=None):
    """Save a load_region result before centering/rotation; never overwrite.

    Unit conversion is allowed: arrays are converted back to native TNG units.
    Source Header counts are preserved separately from selected-region counts.
    Floating fields retain the loader's float32 precision; IDs stay integers.
    A failed write leaves only a .part file, not a completed region file.
    """
    if not hasattr(container, '_region_fields'):
        raise ValueError('save_region requires a container returned by load_region')
    if container._transformations:
        raise ValueError('Save the box-frame region before centering or rotation')
    path = Path(filename)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    props = container.properties
    context = dict(a=props['a'], h=props['h'])
    if catalog is None:
        catalog = getattr(container, '_region_catalog', None)
    counts = np.zeros(6, dtype=np.uint64)
    with h5py.File(partial, 'x') as f:
        f.attrs.update(Format=FORMAT, Run=props['run'])
        source = f.create_group('SourceHeader')
        source.attrs.update(container._region_source_header)
        header = f.create_group('Header')
        header.attrs.update({key: value for key, value in source.attrs.items()
                             if not key.startswith('FileOffsets')})
        region = f.create_group('Region')
        region.attrs.update(shape=props['region_shape'],
                            periodic=bool(props['region_periodic']),
                            coordinate_frame='absolute box',
                            membership_selection=False)
        for key in ('center', 'half_size'):
            array = props['region_' + key].in_units(UnitComvingLength, **context)
            dataset = region.create_dataset(key, data=np.asarray(array))
            dataset.attrs['units'] = str(UnitComvingLength)
        if catalog is not None:
            from .tng_config import get_groupcat_field_unit
            for section, arrays in catalog.items():
                group = f.require_group('Catalog/' + section)
                for name, values in arrays.items():
                    dataset = group.create_dataset(name, data=np.asarray(values))
                    try:
                        dataset.attrs['units'] = str(get_groupcat_field_unit(name))
                    except KeyError:
                        dataset.attrs['units'] = '1'
        for family, names in container._region_fields.items():
            fam = getattr(container, family)
            pt = partTypeNum(family)
            counts[pt] = len(fam)
            group = f.create_group(f'PartType{pt}')
            for name in names:
                # Empty families have no loaded arrays; retain dataset shapes.
                if not len(fam):
                    shape, dtype = container._region_field_specs[family][name]
                    values = np.empty((0, *shape), dtype=dtype)
                else:
                    array = fam[get_particle_field_name(name)]
                    values = np.asarray(array.in_units(get_particle_field_unit(name),
                                                       **context))
                dataset = group.create_dataset(name, data=values,
                                               compression='gzip', compression_opts=1,
                                               shuffle=True)
                dataset.attrs['units'] = str(get_particle_field_unit(name))
        header.attrs.update(SnapshotNumber=int(props['Snapshot']),
                            NumFilesPerSnapshot=1, NumPart_ThisFile=counts,
                            NumPart_Total=(counts & np.uint64(0xffffffff)).astype('u4'),
                            NumPart_Total_HighWord=(counts >> np.uint64(32)).astype('u4'))
        f.attrs['Complete'] = True
    # Hard-link publication is atomic and cannot overwrite an existing target.
    import os
    os.link(partial, path)
    partial.unlink()
    return path


def read_region(filename):
    """Read a region without snapshot files, group catalogues or filename rules."""
    path = Path(filename)
    with h5py.File(path, 'r') as f:
        if f.attrs.get('Format') != FORMAT or not f.attrs.get('Complete', False):
            raise ValueError('Not a complete PyTNG region file')
        props = header_to_properties(f['Header'], str(f.attrs['Run']), str(path.parent))
        counts = {family: int(f['Header'].attrs['NumPart_ThisFile'][pt])
                  for pt, family in ((0, 'gas'), (1, 'dm'), (4, 'star'), (5, 'bh'))
                  if f'PartType{pt}' in f}
        container = pynbody.new(**counts, order=','.join(counts))
        enforce_dtype(container)
        container.properties = props
        region = f['Region']
        for key in ('center', 'half_size'):
            props['region_' + key] = SimArray(region[key][...], UnitComvingLength)
        props['region_shape'] = str(region.attrs['shape'])
        props['region_periodic'] = bool(region.attrs['periodic'])
        for value in props.values():
            if isinstance(value, SimArray):
                value.sim = container
        fields, specs = {}, {}
        for family, count in counts.items():
            group = f[f'PartType{partTypeNum(family)}']
            fields[family] = list(group)
            specs[family] = {name: (ds.shape[1:], ds.dtype) for name, ds in group.items()}
            if not count:
                continue
            fam = getattr(container, family)
            for name, dataset in group.items():
                if dataset.shape[0] != count:
                    raise ValueError(f'Incorrect row count: {dataset.name}')
                values = dataset[...]
                # Never cast particle IDs to float; family assignment avoids shadows.
                fam[get_particle_field_name(name)] = SimArray(
                    values, get_particle_field_unit(name))
            if family == 'dm' and 'Masses' not in group:
                fam['mass'] = SimArray(np.full(count, float(props['mDM']), dtype='f4'),
                                       get_particle_field_unit('Masses'))
        container._region_fields = fields
        container._region_field_specs = specs
        container._region_source_header = dict(f['SourceHeader'].attrs)
        container._region_catalog = {
            section: {name: ds[...] for name, ds in group.items()}
            for section, group in f.get('Catalog', {}).items()}
    return container
