"""Run or redraw a small TNG stellar-structure traceback pilot.

Centers JSON maps snapshot numbers to [x, y, z] in native ckpc/h, typically
SubhaloPos along a main progenitor branch. See docs/tracer-history.md.
"""

import argparse
import json
from dataclasses import fields
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from kinematic_decompose.PyTNG.tracer import (
    STRUCTURES, TracerCohort, TracerFrame, preflight_tracer_snapshots,
    seed_decomposed_galaxy, seed_tracer_cohort, trace_snapshot,
)
from kinematic_decompose.visualization.tracer import visualize_tracer_history


def run(args):
    if not args.plot_only:
        from kinematic_decompose.config import BASEPATH
        if args.subhalo is None or not args.snapshots:
            raise ValueError('A new run requires --subhalo and --snapshots')
        if max(args.snapshots) > args.seed or len(set(args.snapshots)) != len(args.snapshots):
            raise ValueError('Trace snapshots must be unique and no later than the seed')
        missing = set(args.snapshots) - args.centers.keys()
        if missing:
            raise ValueError(f'Missing projection centers: {sorted(missing)}')
        base = Path(BASEPATH) / args.run / 'output'
        preflight_tracer_snapshots(base, [args.seed, *args.snapshots])
        if args.seed_particles:
            with np.load(args.seed_particles, allow_pickle=False) as labels:
                if (labels['run'].item(), labels['seed_snap'].item(), labels['subhalo_id'].item()) != (
                    args.run, args.seed, args.subhalo
                ):
                    raise ValueError('Seed label file does not match the requested galaxy')
                cohort = seed_tracer_cohort(base, args.seed, args.subhalo,
                                            labels['star_ids'], labels['star_labels'])
        else:
            from kinematic_decompose.pipeline import kinematic_decomposition_pipeline

            _, galaxy, _, _ = kinematic_decomposition_pipeline(args.run, args.seed, args.subhalo)
            cohort = seed_decomposed_galaxy(base, args.seed, args.subhalo, galaxy)
            del galaxy
        for name in args.structures:
            n, covered, ntr = cohort.coverage(name)
            print(f'{name}: {covered}/{n} seed stars represented; {ntr} tracers', flush=True)
        if not cohort.tracer_ids.size:
            raise ValueError('No seed tracers matched the selected galaxy')
        data = {f'cohort_{f.name}': getattr(cohort, f.name) for f in fields(cohort)}
        data['run'] = args.run
        data['seed_source'] = str(args.seed_particles.resolve()) if args.seed_particles else 'pipeline'
        data['snapshots'] = sorted(args.snapshots)
        for snap in data['snapshots']:
            print(f'Scanning snapshot {snap}', flush=True)
            frame = trace_snapshot(base, cohort, snap)
            print(f'  {frame.tracer_found.sum()}/{len(cohort.tracer_ids)} tracers; '
                  f'{frame.parent_found.sum()} parents resolved (per tracer)', flush=True)
            for field in fields(frame):
                if field.name != 'tracer_ids':
                    data[f'{snap}_{field.name}'] = getattr(frame, field.name)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=args.cache.parent, delete=False) as tmp:
            temporary = Path(tmp.name)
            try:
                np.savez_compressed(tmp, **data)
                tmp.flush()
                temporary.replace(args.cache)
            finally:
                temporary.unlink(missing_ok=True)

    with np.load(args.cache, allow_pickle=False) as data:
        def read(key):
            value = data[key]
            return value.item() if value.ndim == 0 else value

        cohort = TracerCohort(**{f.name: read(f'cohort_{f.name}') for f in fields(TracerCohort)})
        frames = [TracerFrame(**{f.name: (cohort.tracer_ids if f.name == 'tracer_ids'
                                         else read(f'{snap}_{f.name}'))
                                 for f in fields(TracerFrame)}) for snap in data['snapshots']]
    fig = visualize_tracer_history(cohort, frames, args.centers, half_width=args.half_width,
                                   structures=args.structures, physical=not args.comoving)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=200, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='TNG50-1')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--subhalo', type=int)
    parser.add_argument('--seed-particles', type=Path, help='Optional NPZ of existing real-star IDs and labels')
    parser.add_argument('--snapshots', nargs='+', type=int)
    parser.add_argument('--structures', nargs='+', choices=STRUCTURES, default=['bulge', 'disk', 'halo'])
    parser.add_argument('--centers', required=True, type=Path)
    parser.add_argument('--half-width', required=True, type=float)
    parser.add_argument('--comoving', action='store_true')
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--figure', required=True, type=Path)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    args.centers = {int(k): v for k, v in json.loads(args.centers.read_text()).items()}
    run(args)


if __name__ == '__main__':
    main()
