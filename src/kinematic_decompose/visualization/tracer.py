"""Spatial projections of a frozen stellar-structure tracer cohort."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..PyTNG.tracer import PHASES, relative_positions
from ..mixture.util import STRUCTURES


_PHASE_COLORS = dict(zip(PHASES, ('#188879', '#ce473f', '#4165a9', '#242424', '#888888')))


def visualize_tracer_history(cohort, frames, centers, *, half_width,
                             structures=('bulge', 'disk', 'halo'), axes=(0, 1),
                             physical=True):
    """Rows are seed structures, columns are epochs; draw one mark per tracer.

    Centers are keyed by snapshot (e.g. progenitor SubhaloPos), in ckpc/h.
    All panels use the same box-axis projection and limits. This is a full
    line-of-sight projection, not an aperture cut in the third dimension.
    """
    frames, structures = tuple(frames), tuple(structures)
    if not frames or not structures or len({f.snap for f in frames}) != len(frames):
        raise ValueError('Provide snapshots and structures; snapshots must be unique')
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError('half_width must be positive and finite')
    if len(axes) != 2 or len(set(axes)) != 2 or any(i not in (0, 1, 2) for i in axes):
        raise ValueError('Select two distinct box axes from (0, 1, 2)')
    selected = [np.isin(cohort.labels, STRUCTURES[name]) for name in structures]
    offsets = []
    for frame in frames:
        if not np.array_equal(frame.tracer_ids, cohort.tracer_ids):
            raise ValueError(f'Snapshot {frame.snap} does not match the tracer cohort')
        if frame.snap not in centers:
            raise ValueError(f'Missing center for snapshot {frame.snap}')
        offsets.append(relative_positions(frame, centers[frame.snap], physical=physical))
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 11}):
        fig, grid = plt.subplots(len(structures), len(frames), squeeze=False,
                                 sharex=True, sharey=True, layout='constrained',
                                 figsize=(3 * len(frames) + 1, 3 * len(structures) + .4))
        units = 'pkpc' if physical else 'ckpc/h'
        for col, (frame, pos) in enumerate(zip(frames, offsets)):
            for row, (name, member) in enumerate(zip(structures, selected)):
                ax = grid[row, col]
                x, y = pos[:, axes].T
                resolved = member & frame.parent_found & np.isfinite(x) & np.isfinite(y)
                in_view = resolved & (np.abs(x) <= half_width) & (np.abs(y) <= half_width)
                for phase, color in _PHASE_COLORS.items():
                    ok = in_view & (frame.phase == phase)
                    if ok.any():
                        ax.scatter(x[ok], y[ok], s=3, color=color, alpha=.55,
                                   linewidths=0, rasterized=True)
                ax.text(.035, .965, f'{in_view.sum():,} / {member.sum():,} in view',
                        transform=ax.transAxes, va='top', fontsize='small')
                ax.set(xlim=(-half_width, half_width), ylim=(-half_width, half_width), aspect='equal')
                if row == 0:
                    ax.set_title(f'$z={frame.redshift:.2f}$')
                if col == 0:
                    ax.set_ylabel(f'{name.replace("_", " ").title()}\nΔ{"xyz"[axes[1]]} [{units}]')
                if row == len(structures) - 1:
                    ax.set_xlabel(f'Δ{"xyz"[axes[0]]} [{units}]')
        handles = [Line2D([], [], marker='o', linestyle='none', color=c, label=p)
                   for p, c in _PHASE_COLORS.items()]
        fig.legend(handles=handles, title=f'Parent phase · seed snapshot {cohort.seed_snap}',
                   loc='outside lower center', ncol=len(handles), frameon=False)
    return fig
