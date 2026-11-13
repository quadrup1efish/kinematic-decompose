"""Pynbody surface densities of a spatial region, including its environment.

Run: .venv/bin/python tests/example_load_region.py
Cached projections are reused; pass --recompute to reread the snapshot.
Widths/depths are physical Mpc. No halo membership or rotation is used.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pynbody
from pynbody.array import SimArray

from kinematic_decompose.PyTNG.snapshot_loader import Snapshot


FAMILIES = ('dm', 'gas', 'star')
ROOT = Path(__file__).resolve().parents[1]


def surface_density_maps(container, width_kpc, resolution):
    """Return linear Sigma maps in Msol/kpc^2, computed only by pynbody."""
    maps, statistics = {}, {}
    pixel_area = (width_kpc / resolution)**2
    for family in FAMILIES:
        particles = getattr(container, family)
        print(f'Projecting {family}: {len(particles):,} particles', flush=True)
        start = perf_counter()
        # Adaptive neighbour lengths suit the diffuse DM/gas distribution.
        # Sparse stars instead use a pixel-scale kernel, not Mpc-sized kernels.
        if family == 'star':
            # Pin rho before changing smooth. In this projection rho cancels
            # the SPH volume factor m/rho, leaving a mass-weighted kernel sum.
            particles['rho'] = particles['rho'].copy()
            particles['smooth'] = SimArray(
                np.full(len(particles), width_kpc / resolution,
                        dtype=particles['pos'].dtype), particles['pos'].units)
        # Surface-density output units request a line-of-sight integral, not a slice.
        sigma = pynbody.plot.sph.image(
            particles, qty='rho', width=width_kpc, resolution=resolution,
            units='Msol kpc^-2', weight=False, log=False, noplot=True,
            return_array=True, threaded=False, approximate_fast=False,
            denoise=False, fill_nan=False)
        maps[family] = np.asarray(sigma)
        assert sigma.shape == (resolution, resolution)
        assert np.isfinite(sigma).all() and np.all(sigma >= 0)
        assert np.any(sigma > 0)
        statistics[family] = dict(
            count=len(particles),
            particle_mass_Msol=float(np.asarray(particles['mass']).sum(dtype=np.float64)),
            image_mass_Msol=float(maps[family].sum(dtype=np.float64) * pixel_area),
            smooth_median_kpc=float(np.median(particles['smooth'])),
            seconds=perf_counter() - start)
        print(f'  completed in {statistics[family]["seconds"]:.1f} s', flush=True)
    return maps, statistics


def project_region(basepath, center, width_mpc, depth_mpc, resolution):
    """Load only positions/masses, convert units, and project along box z."""
    snapshot = Snapshot(str(basepath), 99, header_source='snapshot')
    half_size = SimArray(np.array([width_mpc, width_mpc, depth_mpc]) * 500., 'kpc')
    fields = {family: ['Coordinates', 'Masses'] for family in FAMILIES}
    snapshot.load_region(center, half_size, load_particle_fields=fields)
    snapshot.physical_units()
    snapshot.center(cen=snapshot.properties['region_center'], with_velocity=False)
    maps, statistics = surface_density_maps(snapshot.container, width_mpc * 1000., resolution)
    return maps, dict(statistics=statistics, a=float(snapshot.properties['a']),
                      h=float(snapshot.properties['h']),
                      smooth_neighbours=int(pynbody.config['sph']['smooth-particles']),
                      pynbody_version=pynbody.__version__)


def plot_surface_density(maps, width_mpc):
    """Lay out cached rasters without repeating the particle calculation."""
    style = {'font.family': 'serif',
             'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
             'mathtext.fontset': 'stix', 'font.size': 14,
             'axes.labelsize': 19, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
             'axes.linewidth': .9}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(17, 6.8))
        grid = fig.add_gridspec(2, 3, height_ratios=[1, .045], hspace=.22, wspace=.16)
        cmap = plt.get_cmap('magma').copy()
        cmap.set_under(cmap(0))
        cmap.set_bad(cmap(0))
        extent = np.array([-1, 1, -1, 1]) * width_mpc / 2
        labels = ('Dark matter', 'Gas', 'Stars')
        limits = {}
        for column, (family, label) in enumerate(zip(FAMILIES, labels)):
            ax = fig.add_subplot(grid[0, column])
            sigma = maps[family]
            # Independent ranges keep the lower-density stellar distribution visible.
            vmin, vmax = np.quantile(sigma[sigma > 0], [.01, .999])
            limits[family] = [float(vmin), float(vmax)]
            image = ax.imshow(np.ma.masked_less_equal(sigma, 0), origin='lower',
                              extent=extent, cmap=cmap, norm=LogNorm(vmin, vmax),
                              interpolation='nearest')
            ax.set_aspect('equal')
            ax.set_xticks(np.linspace(-width_mpc / 2, width_mpc / 2, 5))
            ax.set_yticks(np.linspace(-width_mpc / 2, width_mpc / 2, 5))
            ax.set_xlabel(r'$x\;[\mathrm{Mpc}]$')
            if column == 0:
                ax.set_ylabel(r'$y\;[\mathrm{Mpc}]$')
            else:
                ax.tick_params(labelleft=False)
            ax.text(.04, .95, label, transform=ax.transAxes, color='white',
                    va='top', fontsize=19)
            colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[1, column]),
                                    orientation='horizontal')
            colorbar.set_label(r'$\Sigma\;[\mathrm{M}_{\odot}\,\mathrm{kpc}^{-2}]$', fontsize=17)
            colorbar.ax.tick_params(labelsize=13)
    return fig, limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basepath', type=Path,
                        default=Path('/Users/yuwa/sims.TNG/TNG100-3/output'))
    parser.add_argument('--center', type=float, nargs=3,
                        default=[850.66307177, 26334.70465888, 18281.44870506],
                        help='Box coordinates in native ckpc/h')
    parser.add_argument('--width', type=float, default=20., help='Physical width in Mpc')
    parser.add_argument('--depth', type=float, default=10., help='Physical depth in Mpc')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    config = dict(basepath=str(args.basepath.resolve()), snapshot=99,
                  center_ckpc_h=args.center, width_Mpc=args.width,
                  depth_Mpc=args.depth, resolution=args.resolution,
                  stellar_smoothing='one_pixel')
    stem = 'example_load_region_surface_density'
    cache = ROOT / 'data' / f'{stem}.npz'
    cache.parent.mkdir(exist_ok=True)
    if cache.exists() and not args.recompute:
        with np.load(cache) as saved:
            metadata = json.loads(str(saved['metadata']))
            maps = {family: saved[family] for family in FAMILIES}
        if metadata['config'] != config:
            parser.error('Projection settings changed; pass --recompute to replace the cache')
        print(f'Using cached pynbody projections: {cache}')
    else:
        maps, metadata = project_region(args.basepath, args.center, args.width,
                                         args.depth, args.resolution)
        metadata['config'] = config
        np.savez_compressed(cache, **maps, metadata=json.dumps(metadata))
    fig, limits = plot_surface_density(maps, args.width)
    image = ROOT / 'images' / f'{stem}.png'
    fig.savefig(image, dpi=300, bbox_inches='tight')
    fig.savefig(image.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    metadata['color_limits_Msol_kpc2'] = limits
    truth = ROOT / 'images' / 'data_truth'
    truth.mkdir(exist_ok=True)
    (truth / f'{stem}.json').write_text(json.dumps(metadata, indent=2) + '\n')
    lines = ['# Spatial-region surface densities', '',
             'Reproduce: `.venv/bin/python tests/example_load_region.py`', '',
             f'Source: `{config["basepath"]}`, snapshot 99; a={metadata["a"]}, h={metadata["h"]}.',
             f'Center (ckpc/h): {args.center}.',
             f'Physical field: {args.width} x {args.width} Mpc; depth: {args.depth} Mpc.',
             f'Resolution: {args.resolution} squared; pixel: {args.width * 1000 / args.resolution:g} kpc.',
             'Projection: box z axis, periodic spatial selection, no halo membership or rotation.',
             'Renderer: pynbody.plot.sph.image, rho, Msol kpc^-2, unweighted integral.',
             'Smoothing: DM/gas adaptive 32 neighbours; stars fixed to one pixel.', '',
             '| Family | N | Particle mass [Msol] | Image integral [Msol] | Difference [%] | Color limits [Msol/kpc^2] |',
             '|---|---:|---:|---:|---:|---|']
    for family in FAMILIES:
        stat = metadata['statistics'][family]
        error = 100 * (stat['image_mass_Msol'] / stat['particle_mass_Msol'] - 1)
        lines.append(f'| {family} | {stat["count"]} | {stat["particle_mass_Msol"]:.12g} | '
                     f'{stat["image_mass_Msol"]:.12g} | {error:+.6f} | {limits[family]} |')
    (truth / f'{stem}.md').write_text('\n'.join(lines) + '\n')
    notes = ROOT / 'images' / 'notes'
    notes.mkdir(exist_ok=True)
    (notes / f'{stem}.md').write_text(
        '# Reading and limitations\n\n'
        '- Same spatial slab and orientation in all three square panels.\n'
        '- Independent logarithmic color scales (positive-pixel 1st/99.9th percentiles).\n'
        '- Read extended filaments from DM/gas; do not infer inflow from density alone.\n'
        '- Stars use a one-pixel kernel to avoid Mpc-sized sparse-neighbour blobs.\n'
        '- Gas is an SPH visualization of cell masses, not an exact Voronoi projection.\n'
        '- PartType4 is shown without an additional wind-particle cut.\n'
        '- Finite slab boundaries truncate kernels; pixel sampling also affects mass integrals.\n'
        '- The figure demonstrates region loading, not quantitative filament identification.\n')
    print(json.dumps(metadata, indent=2))
    print(f'Saved {image}')


if __name__ == '__main__':
    main()
