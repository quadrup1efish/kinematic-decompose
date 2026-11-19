"""Redraw six stellar birth intervals and an all-ages column from a cache."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kinematic_decompose.analysis.stellar_birth import (
    compute_birth_circularity_history, compute_birth_spatial_history,
)
from kinematic_decompose.visualization.stellar_birth import plot_birth_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('mergers', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--hubble', type=float, required=True, help='Dimensionless h')
    parser.add_argument('--omega-m', type=float, required=True, help='Matter density in flat ΛCDM')
    parser.add_argument('--top-panel', choices=('R', 'jz/jc'), default='R')
    parser.add_argument('--circularity', type=Path,
                        help='Optional NPZ with ids, circularity, and status for this cohort')
    args = parser.parse_args()
    use_circularity = args.top_panel == 'jz/jc'
    if use_circularity and args.circularity is None:
        parser.error('--top-panel jz/jc requires --circularity')
    if not np.isfinite(args.hubble) or args.hubble <= 0 or not 0 < args.omega_m < 1:
        parser.error('Require h > 0 and 0 < Omega_m < 1')
    omega_lambda = 1 - args.omega_m
    age_factor = 2 * 977.79222168 / (300 * args.hubble * np.sqrt(omega_lambda))

    def redshift_at_time(age):
        a = (np.sqrt(args.omega_m / omega_lambda) * np.sinh(age / age_factor))**(2 / 3)
        return 1 / a - 1

    with np.load(args.cache, allow_pickle=False) as cache:
        labels, birth, current, time, mass, rotation, branch, r200 = (
            cache[k] for k in ('labels', 'birth_position', 'seed_position', 'birth_time',
                               'initial_mass', 'rotation', 'branch_time', 'r200c_pkpc'))
        ids = cache['ids'] if use_circularity else None
    circularity_data = None
    if use_circularity:
        with np.load(args.circularity, allow_pickle=False) as data:
            source, epsilon, status = (data[k] for k in ('ids', 'circularity', 'status'))
        if (ids is None or ids.shape != labels.shape or source.shape != ids.shape
                or epsilon.shape != source.shape or status.shape != source.shape
                or any(x.dtype.kind not in 'iu' or np.any(x < 0)
                       or len(np.unique(x)) != x.size for x in (ids, source))):
            parser.error('Circularity and spatial caches must contain unique integer IDs and aligned data')
        ids, source = ids.astype(np.uint64), source.astype(np.uint64)
        order = np.argsort(source)
        if not np.array_equal(source[order], np.sort(ids)):
            parser.error('Circularity IDs must match the spatial cohort exactly')
        rows = order[np.searchsorted(source[order], ids)]
        circularity_data = epsilon[rows], status[rows]
    current_time = float(branch[-1])
    edges = np.linspace(0, current_time - branch[0], 7)
    events = json.loads(args.mergers.read_text())['events']
    merger_times = [current_time - event['lookback_gyr'] for event in events]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'lookback_edges_gyr': edges.tolist(),
               'redshift_edges': redshift_at_time(current_time - edges).tolist(), 'components': {}}
    suffix = 'circularity_redshift' if circularity_data is not None else 'redshift'
    for name, label, extent, map_bins in (('bulge', 2, 6., 96), ('stellar_halo', 3, 45., 180)):
        member = labels == label
        if not member.any():
            continue
        result = compute_birth_spatial_history(
            birth[member], current[member], time[member], mass[member], rotation=rotation,
            seed_time=current_time, lookback_edges=edges,
            time_edges=np.linspace(edges[0], edges[-1], 421),
            radius_edges=np.linspace(0, extent, 91),
            projection_edges=np.linspace(-extent, extent, map_bins + 1))
        circular = None
        if circularity_data is not None:
            epsilon, status = circularity_data
            circular = compute_birth_circularity_history(
                time[member], mass[member], epsilon[member], status[member],
                seed_time=current_time, time_edges=result.time_edges_gyr,
                circularity_edges=np.linspace(-1.2, 1.2, 91))
        fig = plot_birth_history(
            result,
            top_panel=args.top_panel,
            circularity_history=circular,
            virial_history=(branch, r200) if label == 3 and circular is None else None,
            merger_times=merger_times, redshift_at_time=redshift_at_time)

        lookback = current_time - time[member]
        in_window = (lookback >= edges[0]) & (lookback <= edges[-1])
        np.testing.assert_allclose(result.window_mass, mass[member][in_window].sum())
        projected = result.projected_mass
        if in_window.all():
            np.testing.assert_allclose(projected[:, 0], projected[:, 1:].sum(axis=1))
        summary['components'][name] = {
            'stars_per_bin': np.histogram(lookback, edges)[0].tolist(),
            'sfh_time_bins': len(result.time_edges_gyr) - 1,
            'sfh_radius_bins': len(result.radius_edges_pkpc) - 1,
            'sfh_fraction': result.sfh_mass / result.total_mass,
            'outside_window_mass_msun': float(mass[member][~in_window].sum()),
        }
        if circular is not None:
            summary['components'][name]['circularity'] = {
                'window_mass_msun': circular.window_mass,
                'valid_mass_msun': circular.valid_mass,
                'plotted_mass_msun': circular.plotted_mass,
            }
        for ext in ('png', 'pdf'):
            fig.savefig(args.output / f'galaxypose_{name}_{suffix}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    (args.output / f'galaxypose_{suffix}_stats.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
