from functools import reduce
import warnings
import numpy as np
from typing import List, Tuple
from collections.abc import Iterable
from contextlib import contextmanager

from . import illustris_python as il

import pynbody
pynbody.config['number_of_threads']=1

from pynbody import new
from pynbody.array import SimArray
from pynbody.simdict import SimDict
from pynbody.snapshot.simsnap import SimSnap

from .illustris_python.util import partTypeNum

from .tng_config import *
from .extension import *
from .derived_array import *
from .simdict_getter import *

class Snapshot():
    def __init__(self, basePath, snapNum):
        self.basePath = basePath
        self.snapNum  = snapNum 
        self._set_snapshot_properties()
        self._original_transform = None

        self.group_catalog = SimDict()

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._get_galaxy_view([key])
        if isinstance(key, Iterable) and not isinstance(key, (str, bytes)):
            return self._get_galaxy_view(list(key))
        return self.container[key]

    def __setitem__(self, key, value):
        self.container[key] = value

    def __getattr__(self, name):
        if "container" in self.__dict__:
            return getattr(self.container, name)
        raise AttributeError(name)
    
    def _create_container(self, ID, groupType):
        ID = self._parse_id(ID)
        if isinstance(groupType, str):
            groupType = [groupType] * len(ID)
        
        self.ID = ID
        self.groupType = groupType

        self.snap_offsets_lenType = {}
        total_lenType = np.zeros(6, dtype=int)

        for single_id, group_type in zip(ID, groupType):
            snap_offsets = il.snapshot.getSnapOffsets(
                basePath=self.basePath,
                snapNum=self.snapNum,
                id=single_id,
                type=group_type
            )
            lenType = snap_offsets['lenType']
            total_lenType += lenType

            self.snap_offsets_lenType[single_id] = {
                'snap_offsets': snap_offsets,
                'lenType': lenType
            }

        self.new_kwargs = {}
        self.order = []
        for partType in self.partTypes:
            partNum = partTypeNum(partType)
            count = int(total_lenType[partNum])
            if count > 0:
                self.new_kwargs[partType] = count
                self.order.append(partType)
        self.new_kwargs['order'] = ','.join(self.order) 
        self.container = new(**self.new_kwargs)

        for i in self.properties:
                if isinstance(self.properties[i], SimArray):
                    self.properties[i].sim = self.container
        self.container.properties = self.properties
    
    def load_particle(self, ID, groupType='Subhalo', load_particle_fields='default'):
        self._set_load_particle_fields(template=load_particle_fields) 
        self._create_container(ID, groupType)
        self._calculate_galaxy_index() 
        family_map = {
            'star': self.container.s,
            'gas': self.container.g,
            'dm': self.container.dm,
            'bh': self.container.bh
        } 
        for partType in self.partTypes:
            family_snap = family_map[partType]
            fields = list(self.load_particle_fields[partType])
            if partType == 'dm' and 'Masses' in fields:
                fields.remove('Masses')

            for single_id in self.ID:

                snap_offsets = self.snap_offsets_lenType[single_id]['snap_offsets']
                lenType = self.snap_offsets_lenType[single_id]['lenType']

                if lenType[partTypeNum(partType)] == 0:
                    continue

                subset = il.snapshot.loadSubset(
                    self.basePath,
                    self.snapNum,
                    partType,
                    fields,
                    snap_offsets,
                    float32=True,
                    sq=False
                )

                sl = self.galaxy_index[single_id][partType]
                
                for field in fields:

                    field_name = get_particle_field_name(field)
                    field_unit = get_particle_field_unit(field)
                    
                    data = subset[field]

                    if field_name not in family_snap:
                        ndim = 1 if data.ndim == 1 else data.shape[1]

                        family_snap._create_array(
                            field_name,
                            ndim,
                            data.dtype
                        )
                        family_snap[field_name].units = field_unit
                    family_snap[field_name][sl] = data 
                    if family_snap[field_name].units != field_unit:
                        family_snap[field_name].units = field_unit

        if 'dm' in self.partTypes:
            self.container.dm['mass'] = SimArray(
                    np.full(len(self.container.dm), self.container.properties['mDM']),
                    units=get_particle_field_unit('Masses'))
        return self.container
    
    def load_group_catalog(self, ID, groupType='Subhalo'):
        """
        ID = self._parse_id(ID)
        if isinstance(groupType, str):
            groupType = [groupType] * len(ID)
        if getattr(self, 'ID') and ID != self.ID:
            raise Warning("Load ID's group catalog is different from particle, \
                            please care the sequence or subhalo/halo id!")
        """
        #for single_id, group_type in zip(self.ID, groupType):
        #self.group_catalog[single_id] = SimDict()
        single_id = ID
        if groupType == 'Subhalo':
            gc = il.groupcat.loadSingle(self.basePath, self.snapNum, haloID=-1, subhaloID=single_id)
        elif groupType == 'Group':
            gc = il.groupcat.loadSingle(self.basePath, self.snapNum, haloID=single_id, subhaloID=-1)
        for i in gc:
            try:
                self.group_catalog[i] = SimArray(gc[i], get_groupcat_field_unit(i))
                self.group_catalog[i].sim = self.container
            except:
                continue

    def GC_physical_units(self, distance='kpc', velocity='km s^-1', mass='Msol'):
        """
        Converts the units of the group catalog (GC) properties to physical units.

        This method updates the `GC` attribute of the `Subhalo` instance to use physical units
        for its properties, based on predefined unit conversions and the current unit context.

        Conversion is applied only to properties that are not listed in `NotneedtransGCPa`.

        Notes:
        -----
        - `self.ancestor.properties['baseunits']` provides the base units for dimensional analysis.
        - The dimensional projection and conversion are handled using the `units` library.
        - Properties listed in `NotneedtransGCPa` are skipped during the conversion process.
        """
        NotneedtransGCPa = [
            'SubhaloSFR',
            'SubhaloSFRinHalfRad',
            'SubhaloSFRinMaxRad',
            'SubhaloSFRinRad',
            'SubhaloStellarPhotometrics',
            'GroupSFR',
        ]
        dims = [units.kpc, units.km/units.s, units.Msol, units.a, units.h]
        urc = len(dims) - 2
        for k in list(self.group_catalog):
            if k in NotneedtransGCPa:
                continue
            v = self.group_catalog[k]
            if isinstance(v, units.UnitBase):
                try:
                    new_unit = v.dimensional_project(dims)
                except units.UnitsException:
                    continue
                new_unit = reduce(
                    lambda x, y: x * y, [a**b for a, b in zip(dims, new_unit[:urc])]
                )
                new_unit *= v.ratio(new_unit, **self.conversion_context())
                self.group_catalog[k] = new_unit
            if isinstance(v, SimArray):
                if (v.units is not None) and (v.units is not units.no_unit):
                    try:
                        d = v.units.dimensional_project(dims)
                    except units.UnitsException:
                        return
                    new_unit = reduce(
                        lambda x, y: x * y, [a**b for a, b in zip(dims, d[:urc])]
                    )
                    if new_unit != v.units:
                        self.group_catalog[k].convert_units(new_unit)

    def center(self, cen = None, vel_cen=None, with_velocity=True, return_cen=False, mode: str = 'ssc', cen_size='2 kpc' ,**kwargs) -> SimSnap:
        if cen is None:
            pynbody.analysis.center(self.container, mode=mode, wrap=True, with_velocity=with_velocity, return_cen=False, cen_size="2 kpc")
            if return_cen:
                print("Be cautious: have not solve this return_cen but not moving snapshot.")
            return self.container
        else:
            self.container.translate(-cen)
            if with_velocity:
                if vel_cen is None:
                    cen = self.container.star[filt.Sphere(cen_size)]
                    if len(cen) < 5:
                        print("Insufficient particles around center to get velocity, \
                        try to use SubhaloVel")
                        if getattr(self, "group_catalog"):
                            vcen = self.group_catalog['SubhaloVel']
                            if vcen.units != self.container['vel'].units:
                                raise ValueError("SubhaloVel units != Vel units, Please using GC_physical_units")
                        else:
                            raise ValueError("Insufficient particles around center to get velocity and \
                                            do not find SubhaloVel.")
                    else:
                        vcen = (cen['vel'].transpose() * cen['mass']).sum(axis=1) / cen['mass'].sum()
                        vcen.units = cen['vel'].units
                    self.container.offset_velocity(-vcen)
                else:
                    self.container.offset_velocity(-vel_cen)
            if return_cen:
                return self.container, cen, vel_cen
            else:
                return self.container 
    
    def faceon(self, align_with='star', range: Tuple[float, float] = (2, 30), as_context: bool = False, **kwargs):
        rmin, rmax = range
        if align_with == 'star':
            family = self.container.s
        elif align_with == 'gas':
            family = self.container.g
        elif align_with == 'dm':
            family = self.container.dm
        else:
            raise ValueError(f"Unknown family to align with: {align_with}")

        selected_region = family[(family['r'] >= rmin) & (family['r'] <= rmax)]
        if len(selected_region) <= 10:
            warnings.warn(
                f"Too few particles in radial range {rmin}-{rmax} kpc for {align_with}",
                RuntimeWarning
            )
            selected_region = family
        angmom = pynbody.analysis.angmom.ang_mom_vec(selected_region)
        trans = pynbody.analysis.angmom.calc_faceon_matrix(angmom)

        # Check the Transformation matrix.
        if not np.allclose(trans.T @ trans, np.eye(3), atol=1e-6):
            warnings.warn(
                f"Transformation matrix is not orthogonal",
                RuntimeWarning
            )
            trans = np.eye(3)

        if not as_context:
            self.container.rotate(trans)
            return self.container

        @contextmanager
        def _context():
            try:
                self.container.rotate(trans)
                yield self.container
            finally:
                self.container.rotate(trans.T)
        return _context()

    def sideon(self, range: Tuple[float, float] = (2,30), **kwargs) -> SimSnap:
        rmin, rmax = range[0], range[1]
        selected_region = self.container.s[(self.container.s['r'] >= rmin) & (self.container.s['r'] <= rmax)]

        angmom = pynbody.analysis.angmom.ang_mom_vec(selected_region)
        trans = pynbody.analysis.angmom.calc_sideon_matrix(angmom)
        self.container.rotate(trans)
    
    def dump_container(self, file_name=None):
        if file_name is None:
            file_name = f"./{self.properties['run']}_{self.snapNum}.hdf5"
        import h5py
        target_families = ['star']
        target_fields   = ['pos', 'vel', 'mass', 'iord', 'aform', 'phi', 'jc', 'label', 'prob']
        with h5py.File(file_name, "w") as f:
            gp = f.create_group("properties")
            for k, v in self.container.properties.items():
                if isinstance(v, SimArray):
                    d = gp.create_dataset(k, data=v)
                    d.attrs["units"] = str(v.units)
                else:
                    gp.attrs[k] = v
            for family in self.container.families():
                if family.name in target_families:
                    fs = self.container[family]
                    g = f.create_group(family.name)
                    for field in fs.keys():
                        if field in target_fields:
                            arr = fs[field]
                            d = g.create_dataset(field, data=arr)
                            if isinstance(arr, SimArray):
                                d.attrs["units"] = str(arr.units)

    def load_container(self, file_name, store_in_instance=False):
        import h5py
        from pynbody.units import NoUnit
        with h5py.File(file_name, "r") as f:
            counts = {
                family: len(f[family]["pos"])
                for family in f
                if family != "properties"
            }

            container = pynbody.new(**counts)

            if "properties" in f:
                gp = f["properties"]
                container.properties.update(gp.attrs)

                for k in gp:
                    d = gp[k]
                    arr = SimArray(d[...], units=d.attrs.get("units", NoUnit()) if d.attrs.get("units") != "NoUnit()" else NoUnit())
                    arr.sim = container
                    container.properties[k] = arr

            for family in counts:
                fs = getattr(container, family)
                g = f[family]
                for field in g:
                    d = g[field]
                    fs[field] = SimArray(d[...], units=d.attrs.get("units", NoUnit()) if d.attrs.get("units") != "NoUnit()" else NoUnit())
        if store_in_instance: 
            self.container = container
            self.properties = self.container.properties
            for i in self.properties:
                if isinstance(self.properties[i], SimArray):
                    self.properties[i].sim = self.container
        return container

    def _set_snapshot_properties(self):
        """
        Get the simulation properties 
        """
        run    = self.basePath.rstrip('/').split('/')[-2]
        header = il.groupcat.loadHeader(self.basePath, self.snapNum)

        self.properties = SimDict()
        self.properties['filedir'] = self.basePath
        self.properties['Snapshot'] = self.snapNum
        self.properties.update({
            'run': run,
            'a': header['Time'],
            'h': header['HubbleParam'],
            'Redshift': header['Redshift'],
            'omegaM0': header['Omega0'],
            'omegaL0': header['OmegaLambda'],
            'boxsize': SimArray(header['BoxSize'], UnitComvingLength), 
        })

        eps, mDM = get_eps_mDM(self.properties)
        self.properties['mDM'] = mDM 
        self.properties['eps'] = eps 
        self.properties['standard_units'] = [
                'nH',
                'Halpha',
                'em',
                'ne',
                'temp',
                'mu',
                'c_n_sq',
                'p',
                'cs',
                'c_s',
                'acc',
                'phi',
                'age',
                'tform',
                'SubhaloPos',
                'sfr',
        ]
    
    def _set_load_particle_fields(self, template='default'):
        if isinstance(template, str):
            if template == 'default':
                field = {}
                field['star'] = ['Coordinates', 'Velocities', 'Masses']
                field['gas']  = ['Coordinates', 'Velocities', 'Masses']
                field['dm']   = ['Coordinates', 'Velocities', 'Masses']
            elif template == 'potential':
                field = {}
                field['star'] = ['Coordinates', 'Velocities', 'Masses']
                field['gas']  = ['Coordinates', 'Masses']
                field['dm']   = ['Coordinates', 'Masses']
            else:
                print("Not include this template, using default.")
                field = {}
                field['star'] = ['Coordinates', 'Velocities', 'Masses']
                field['gas']  = ['Coordinates', 'Velocities', 'Masses']
                field['dm']   = ['Coordinates', 'Velocities', 'Masses']
        elif isinstance(template, dict):
            field = template
        self.load_particle_fields = field
        self.partTypes = list(field.keys())
    
    def _parse_id(self, ids):
        if isinstance(ids, (int, np.integer)):
            return [int(ids)]

        if isinstance(ids, Iterable) and not isinstance(ids, (str, bytes)):
            return [int(i) for i in ids if isinstance(i, (int, np.integer))]

        raise TypeError(f"Invalid ID type: {type(ids)}")

    def _calculate_galaxy_index(self):
        self.galaxy_index = {}

        cursor = {
            'star': 0,
            'gas': 0,
            'dm': 0,
            'bh': 0
        }

        for single_id in self.ID:

            self.galaxy_index[single_id] = {}
            lenType = self.snap_offsets_lenType[single_id]['lenType']

            # gas
            if lenType[0] > 0:
                start = cursor['gas']
                stop = start + lenType[0]
                self.galaxy_index[single_id]['gas'] = slice(start, stop)
                cursor['gas'] = stop

            # dm
            if lenType[1] > 0:
                start = cursor['dm']
                stop = start + lenType[1]
                self.galaxy_index[single_id]['dm'] = slice(start, stop)
                cursor['dm'] = stop

            # star
            if lenType[4] > 0:
                start = cursor['star']
                stop = start + lenType[4]
                self.galaxy_index[single_id]['star'] = slice(start, stop)
                cursor['star'] = stop

            # bh
            if lenType[5] > 0:
                start = cursor['bh']
                stop = start + lenType[5]
                self.galaxy_index[single_id]['bh'] = slice(start, stop)
                cursor['bh'] = stop
    
    def _get_galaxy_view(self, ids):
        ids = self._parse_id(ids)
        subs = []

        for id in ids:
            if id not in self.ID:
                raise ValueError(f"id {id} not loaded")
            for fname, sl in self.galaxy_index[id].items():
                family = getattr(self.container, fname)
                subs.append(family[sl])

        if not subs:
            raise RuntimeError("No particles found")

        galaxy_view = subs[0]
        for sub in subs[1:]:
            galaxy_view = galaxy_view.union(sub)

        return galaxy_view
