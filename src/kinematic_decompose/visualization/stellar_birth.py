"""Render fixed-cohort birth histories without opening simulation snapshots."""

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
import numpy as np


from ..analysis.stellar_birth import compute_birth_distributions, compute_birth_spatial_history


def visualize_stellar_birth(birth, *, time_edges, radius_edges, circularity_edges,
                            structures=('bulge', 'disk', 'halo'), seed_circularity=None):
    """Plot binned formation rates and optionally matched birth/seed circularity.

    The first two columns show initial stellar mass per time bin per plotted
    bin, not the total progenitor SFR. The optional last column shows initial
    stellar mass per circularity-bin pair. seed_circularity is aligned to
    birth.particle_ids. Return the figure and per-structure coverage records.
    """
    result = compute_birth_distributions(
        birth, time_edges=time_edges, radius_edges=radius_edges, circularity_edges=circularity_edges,
        structures=structures, seed_circularity=seed_circularity)
    return plot_birth_distributions(result), result.coverage


def plot_birth_distributions(result):
    """Render the binned output of `compute_birth_distributions`."""
    items, panels = result.structures, result.panels
    ncol = len(panels[0])
    ylabel = [r'$R_{\rm birth}$ [pkpc]', r'$\epsilon_{\rm birth}$', r'$\epsilon_{\rm seed}$']
    rc = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
          'mathtext.fontset': 'stix', 'axes.labelsize': 14, 'font.size': 11}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(len(items), ncol, squeeze=False, layout='constrained',
                                 figsize=(3.7 * ncol, 2.8 * len(items) + .6))
        for col in range(ncol):
            mesh = None
            ticks = None
            positive = np.concatenate([row[col][0][row[col][0] > 0] for row in panels])
            if positive.size:
                lo, hi = float(positive.min()), float(positive.max())
                if hi / lo < 1.01:
                    lo, hi = lo / 2, hi * 2
                norm = LogNorm(lo, hi)
                ticks = np.geomspace(lo, hi, 3 if hi / lo < 10 else 4)
            else:
                norm = None
            for row, name in enumerate(items):
                ax = axes[row, col]
                values, xe, ye = panels[row][col]
                if norm is not None:
                    mesh = ax.pcolormesh(xe, ye, np.ma.masked_less_equal(values.T, 0),
                                         cmap='inferno', norm=norm, rasterized=True)
                if not np.any(values > 0):
                    ax.text(.5, .5, 'No data in range', ha='center', transform=ax.transAxes)
                if col == 2:
                    ax.plot([xe[0], xe[-1]], [ye[0], ye[-1]], '--', color='0.5', lw=1)
                ax.set(xlim=(xe[0], xe[-1]), ylim=(ye[0], ye[-1]),
                       ylabel=(name.replace('_', ' ').title() + '\n' if col == 0 else '') + ylabel[col])
                if row == len(items) - 1:
                    ax.set_xlabel('Cosmic time at birth [Gyr]' if col < 2 else r'$\epsilon_{\rm birth}$')
            if mesh is not None:
                label = r'$M_{\mathrm{formed}}$ [$M_\odot$ / bin pair]' if col == 2 else r'SFR [$M_\odot\,\mathrm{yr}^{-1}$ / bin]'
                colorbar = fig.colorbar(mesh, ax=axes[:, col], location='bottom',
                                       label=label, ticks=ticks, format='%.2g')
                colorbar.minorticks_off()
    return fig


def plot_birth_spatial_history(birth_position, seed_position, birth_time, initial_mass,
                               *, seed_time, lookback_edges, rotation, half_width,
                               virial_history=None, merger_times=None,
                               map_bins=80, radial_bins=90, time_bins=420,
                               show_box_axes=False, redshift_at_time=None):
    """GalaxyPose-style SFH plus XY/XZ maps for one fixed stellar structure.

    Supply aligned, centered positions in pkpc in box axes and initial masses
    in solar masses. `rotation` maps box vectors to the fixed seed frame (one
    axis per row). Lookback time is measured from the seed. `virial_history`,
    if present, gives (cosmic time, physical R200c). An optional legacy third
    element is ignored. `merger_times` gives descendant cosmic times in Gyr;
    coincident events share a single marker. Select the events before plotting.
    Birth and seed maps use the same fixed orientation. Set `show_box_axes`
    to add a second view of the same centered birth positions in box axes.
    The upper panel uses GalaxyPose SFHPlotConfig's uniform 420 time and 90
    radial bins by default, without smoothing, independently of the projection
    columns. Both projections use initial
    stellar mass weights (not surviving mass at the current snapshot).
    Optional `redshift_at_time` maps cosmic ages in Gyr to redshift. It changes
    labels, not the time bins or SFR units.
    Return the figure and coverage of the plotted masses.
    """
    edges = np.asarray(lookback_edges)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(half_width) or half_width <= 0:
        raise ValueError('Invalid time bins or extent')
    if any(not isinstance(k, (int, np.integer)) or k < 1 for k in (map_bins, time_bins, radial_bins)):
        raise ValueError('Bin counts must be positive integers')
    result = compute_birth_spatial_history(
        birth_position, seed_position, birth_time, initial_mass,
        seed_time=seed_time, lookback_edges=edges, rotation=rotation,
        time_edges=np.linspace(edges[0], edges[-1], time_bins + 1),
        radius_edges=np.linspace(0, half_width, radial_bins + 1),
        projection_edges=np.linspace(-half_width, half_width, map_bins + 1),
        show_box_axes=show_box_axes)
    fig = plot_birth_history(result, virial_history=virial_history, merger_times=merger_times,
                             redshift_at_time=redshift_at_time)
    return fig, {'total_mass': result.total_mass, 'window_mass': result.window_mass,
                 'sfh_mass': result.sfh_mass, 'sfh_time_edges': result.time_edges_gyr,
                 'sfh_radius_edges': result.radius_edges_pkpc,
                 'time_bin_mass': result.time_bin_mass.tolist(),
                 'projected_mass': result.projected_mass}


def plot_birth_history(result, *, top_panel='R', circularity_history=None,
                       current_circularity_history=None, virial_history=None,
                       merger_times=None, redshift_at_time=None):
    """Render `compute_birth_spatial_history` output without rebinning.

    `top_panel` selects 'R' (default) or 'jz/jc'. For 'jz/jc', pass one or both
    of `circularity_history` (at birth) and `current_circularity_history`
    (today, for the same cohort); with both, the upper block shows two rows
    sharing one colour scale. Spatial projections are unaffected.
    Overlay times are cosmic ages in Gyr; `redshift_at_time` changes labels
    only. Returns a Matplotlib figure and leaves saving to the caller.
    """
    seed_time, edges = result.seed_time_gyr, result.lookback_edges_gyr
    sfh_t, radii, sfr = result.time_edges_gyr, result.radius_edges_pkpc, result.sfr_surface_density
    grid, maps = result.projection_edges_pkpc, result.stellar_surface_density
    if top_panel not in ('R', 'jz/jc'):
        raise ValueError("top_panel must be 'R' or 'jz/jc'")

    rows = []
    if top_panel == 'R':
        rows.append({'values': sfr, 'edges': radii, 'ylabel': r'$R_{\rm birth}$ [pkpc]',
                     'coverage': None})
        top_label = (r'$\Sigma_{\rm SFR}$ [$M_\odot\,\mathrm{yr}^{-1}\,\mathrm{pkpc}^{-2}$]')
    else:
        same_window = np.array_equal(sfh_t[[0, -1]], edges[[0, -1]])
        for history, suffix in ((circularity_history, 'birth'),
                                (current_circularity_history, 'now')):
            if history is None:
                continue
            if (history.seed_time_gyr != seed_time
                    or not np.array_equal(history.time_edges_gyr, sfh_t)
                    or (same_window and not np.isclose(history.window_mass, result.window_mass))):
                raise ValueError('Circularity and projection histories must use the same cohort and SFH grid')
            rows.append({'values': history.sfr_per_circularity,
                         'edges': history.circularity_edges,
                         'ylabel': rf'$(j_z/j_c)_{{\rm {suffix}}}$'})
        if not rows:
            raise ValueError("top_panel='jz/jc' requires a circularity history")
        if virial_history is not None:
            raise ValueError('R200c cannot be overlaid on a dimensionless circularity axis')
        top_label = r'$d\mathrm{SFR}/d\epsilon$ [$M_\odot\,\mathrm{yr}^{-1}$]'

    widths = np.diff(grid)
    uniform_grid = np.allclose(widths, widths[0], rtol=1e-12, atol=0)
    _, ncols = maps.shape[:2]
    frames = ('Current', 'Birth', 'Birth: box axes')[:len(maps) // 2]
    redshift_edges = None
    if redshift_at_time is not None:
        redshift_edges = np.asarray(redshift_at_time(seed_time - edges), dtype=float)
        if (redshift_edges.shape != edges.shape or not np.isfinite(redshift_edges).all()
                or np.any(np.diff(redshift_edges) <= 0)):
            raise ValueError('Redshift must increase along the lookback bins')

    def norm(values):
        pos = values[values > 0]
        if not pos.size:
            return LogNorm(1, 10)
        lo, hi = np.percentile(pos, [1, 99.5])
        return LogNorm(lo, max(hi, 1.01 * lo))

    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 12}):
        width = 1.30 * ncols + 1.6
        columns = [1] * ncols + [.12, .15]
        panel_size = .74 * width / sum(columns)
        row_height, block_gap = 2.05, .36
        top_height = row_height          # rows share the single-row footprint
        projection_height = len(maps) * panel_size
        fig = plt.figure(figsize=(width, (top_height + block_gap + projection_height) / .88))
        fig.subplots_adjust(left=.13, right=.87, bottom=.06, top=.94)
        outer = fig.add_gridspec(2, 1, height_ratios=[top_height, projection_height],
                                 hspace=2 * block_gap / (top_height + projection_height))
        top_grid = outer[0].subgridspec(len(rows), len(columns), width_ratios=columns,
                                        wspace=0, hspace=.07)
        map_grid = outer[1].subgridspec(len(maps), len(columns), width_ratios=columns,
                                        wspace=0, hspace=0)
        shared_top = norm(np.concatenate([row['values'][np.isfinite(row['values'])].ravel()
                                          for row in rows]))
        top_axes = []
        for i, row in enumerate(rows):
            ax = fig.add_subplot(top_grid[i, 1:-2])
            top_axes.append(ax)
            mesh = ax.pcolormesh(sfh_t, row['edges'], np.ma.masked_less_equal(row['values'].T, 0),
                                 cmap='inferno', norm=shared_top, rasterized=True)
            ax.set_facecolor(mesh.cmap(0))
            for x in edges[1:-1]:
                ax.axvline(x, color='white', alpha=.26, lw=.6, ls=':')
            if row['edges'][0] < 0 < row['edges'][-1]:
                ax.axhline(0, color='white', alpha=.5, lw=.75, ls=':')
            if virial_history is not None and i == 0:
                cosmic, r200 = map(np.asarray, virial_history[:2])
                ax.plot(seed_time - cosmic, r200, color='#58dbe7', lw=1.4,
                        label=r'FoF $R_{200c}$')
            if merger_times is not None:
                times = np.asarray(merger_times, dtype=float)
                if times.ndim != 1 or not np.isfinite(times).all():
                    raise ValueError('Merger times must be a finite 1D array of cosmic ages')
                lookback = np.unique(seed_time - times)
                visible = lookback[(lookback >= edges[0]) & (lookback <= edges[-1])]
                for k, x in enumerate(visible):
                    ax.axvline(x, color='#eac5ff', lw=1.5, ls='--', zorder=4, clip_on=False,
                               label='SubLink merger' if k == 0 and i == 0 else '_nolegend_')
            if i == 0 and ax.get_legend_handles_labels()[0]:
                legend = ax.legend(loc='upper right', frameon=True, facecolor='black',
                                   edgecolor='none', fontsize=9)
                for text in legend.get_texts():
                    text.set_color('white')
            ax.set(xlim=(edges[0], edges[-1]), ylim=(row['edges'][0], row['edges'][-1]))
            ax.set_ylabel(row['ylabel'], fontsize=10)
            ax.tick_params(labelsize=10)
            if i == 0:
                ax.set(xlabel='Lookback from current snapshot [Gyr]')
                if redshift_edges is not None:
                    ax.set(xticks=edges, xticklabels=[f'{z:.2f}' for z in redshift_edges],
                           xlabel='Redshift')
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
            else:
                ax.tick_params(labelbottom=False, bottom=False)
        fig.colorbar(mesh, cax=fig.add_subplot(top_grid[:, -1]), label=top_label)
        shared = norm(maps)
        projections = []
        for i, row in enumerate(maps):
            for j, values in enumerate(row):
                panel = fig.add_subplot(map_grid[i, j])
                masked = np.ma.masked_less_equal(values, 0)
                if uniform_grid:
                    image = panel.imshow(masked, origin='lower',
                                         extent=(grid[0], grid[-1], grid[0], grid[-1]),
                                         cmap='jet', norm=shared, interpolation='nearest')
                else:
                    image = panel.pcolormesh(grid, grid, masked, cmap='jet', norm=shared,
                                             rasterized=True)
                    panel.set_aspect('equal')
                panel.set_facecolor(image.cmap(0))
                mid = (grid[-1] + grid[0]) / 2
                offset = (grid[-1] - grid[0]) / 3
                ticks = [mid - offset, mid, mid + offset]
                panel.set(xticks=ticks, yticks=ticks)
                bottom = i == len(maps) - 1
                panel.tick_params(left=j == 0, labelleft=j == 0, bottom=bottom,
                                  labelbottom=bottom and j % 2 == 0,
                                  labelsize=9, length=2)
                panel.spines['left'].set_visible(j == 0)
                panel.spines['bottom'].set_visible(bottom)
                if i == 0:
                    if not j:
                        title = 'All ages'
                    elif redshift_edges is not None:
                        title = f'z: {redshift_edges[j-1]:.2f}–{redshift_edges[j]:.2f}'
                    else:
                        title = f'{edges[j-1]:.2f}–{edges[j]:.2f} Gyr'
                    panel.set_title(title, fontsize=10)
                if j == 0:
                    panel.set_ylabel('XY' if i % 2 == 0 else 'XZ', fontsize=10)
                    projections.append(panel)
        for i, label in enumerate(frames):
            top, bottom = [p.get_position() for p in projections[2*i:2*i+2]]
            fig.text(top.x0 - .055, (top.y1 + bottom.y0) / 2, label, rotation=90,
                     va='center', ha='center', fontsize=11)
        fig.supxlabel('X [pkpc]', y=.004)
        fig.colorbar(ScalarMappable(norm=shared, cmap='jet'), cax=fig.add_subplot(map_grid[:, -1]),
                     label=r'$\Sigma_{*}$ [$M_\odot\,\mathrm{pkpc}^{-2}$]')
    return fig
