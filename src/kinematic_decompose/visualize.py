import matplotlib
import numpy as np
from dataclasses import dataclass
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d

available_fonts = [f.name for f in fm.fontManager.ttflist]

# Unified Nature-journal figure style (Nature/Science tier): serif
# Times New Roman, restrained sizes, hairline axes, no grid. Applied
# project-wide so every figure (decomposition panels and test figures)
# shares one look.
NATURE_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.style': 'normal',
    'font.weight': 'normal',
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'axes.linewidth': 0.9,
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'axes.grid': False,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}
rcParams.update(NATURE_STYLE)

def _hist_bin_fd(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)


# ============================================================================
# Component specs: the kinematic classification (which GMM components map to
# bulge / halo / disks), the per-component sort order and the color ramp are
# each defined ONCE here and shared by every figure entry point.
# ============================================================================

def _mask_bulge(m, ecut, etacut):
    return (m[:, 0] < ecut) & (np.abs(m[:, 1]) < etacut)


def _mask_halo(m, ecut, etacut):
    return (m[:, 0] > ecut) & (np.abs(m[:, 1]) < etacut)


def _mask_warmdisk(m, ecut, etacut):
    return (m[:, 1] > etacut) & (m[:, 1] < 0.85)


def _mask_colddisk(m, ecut, etacut):
    return m[:, 1] > 0.85


def _mask_counter_rotate(m, ecut, etacut):
    return m[:, 1] < -etacut


# draw order matches the historic rendering order (overlap-sensitive)
_COMPONENT_ORDER = ['Bulge', 'Halo', 'Cold disk', 'Warm disk',
                    'Counter-rotating disk']
_COMPONENT_MASKS = {
    'Bulge': _mask_bulge,
    'Halo': _mask_halo,
    'Cold disk': _mask_colddisk,
    'Warm disk': _mask_warmdisk,
    'Counter-rotating disk': _mask_counter_rotate,
}
_COMPONENT_SORTS = {
    'Bulge': lambda ms: ms[:, 0].argsort(),
    'Halo': lambda ms: ms[:, 0].argsort(),
    'Cold disk': lambda ms: ms[:, 1].argsort()[::-1],
    'Warm disk': lambda ms: ms[:, 1].argsort()[::-1],
    'Counter-rotating disk': lambda ms: np.abs(ms[:, 1]).argsort()[::-1],
}
_COMPONENT_RAMPS = {
    'Bulge': ('darkred', 'mistyrose'),
    'Halo': ('darkorange', 'peachpuff'),
    'Cold disk': ('darkblue', 'lightblue'),
    'Warm disk': ('darkgreen', 'lightgreen'),
    'Counter-rotating disk': None,  # uses the Purples colormap
}


def classify_components(means, covariances, ecut, etacut):
    """Pure function: split (means, covariances) into sorted per-component
    subgroups. Returns {name: (sub_means, sub_covariances)} in draw order."""
    out = {}
    for name in _COMPONENT_ORDER:
        mask = _COMPONENT_MASKS[name](means, ecut, etacut)
        sm, sc = means[mask], covariances[mask]
        idx = _COMPONENT_SORTS[name](sm)
        out[name] = (sm[idx], sc[idx])
    return out


def _component_colors(name, n):
    """Per-component color ramp, identical to the historic gradient logic."""
    n = max(n, 4)
    if _COMPONENT_RAMPS[name] is None:
        return [mcolors.to_hex(plt.cm.Purples(t)) for t in np.linspace(1, 0, n)]
    c0, c1 = _COMPONENT_RAMPS[name]
    a, b = np.array(mcolors.to_rgb(c0)), np.array(mcolors.to_rgb(c1))
    return [mcolors.to_hex(a * (1 - t) + b * t) for t in np.linspace(0, 1, n)]


def _draw_ellipses(ax, comps, colors, proj):
    """Draw one Gaussian ellipse per component per projection, in the
    historic component order."""
    for name in _COMPONENT_ORDER:
        sm, sc = comps[name]
        for j, (mean, covariance) in enumerate(zip(sm, sc)):
            gaussian_ell(ax, mean[proj], covariance[np.ix_(proj, proj)],
                         colors[name][j])


def gaussian_ell(ax, mean, covariance, color):
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.maximum(eigvals, 0)
    widths = 2 * np.sqrt(2) * np.sqrt(eigvals)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    ellipse = matplotlib.patches.Ellipse(
        xy=mean,
        width=widths[0],
        height=widths[1],
        angle=angle,
        edgecolor=color,
        facecolor='none',
        linewidth=4,
        linestyle='solid',
        alpha=0.75,
    )
    ax.add_patch(ellipse)
    ax.scatter(mean[0], mean[1], marker='x', color='k')

def _visualize_residual(X, means, covariances, extent):
    proj = [1,0]
    X_positive = X[X > 0]
    vmin = np.nanpercentile(X_positive, 1)
    vmax = np.nanpercentile(X_positive, 99)
    plt.figure(figsize=(6,5))
    plt.imshow(X, cmap='bwr', 
                extent=extent,
                origin='lower',
                norm=LogNorm(vmin=vmin, vmax=vmax))
    ax = plt.gca()
    for j, (mean, covariance) in enumerate(zip(means, covariances)):
        gaussian_ell(ax, mean[proj], covariance[np.ix_(proj, proj)], color='k')
    plt.show()
    return

def visualize_phase_space(X, means=None, covariances=None, ecut=-0.75, etacut=0.50, threshold_line=False, dims=2, ranges=None): 
    if means is not None and covariances is not None:
        ncs, dims = means.shape
        comps = classify_components(means, covariances, ecut, etacut)
        colors = {name: _component_colors(name, len(ms))
                  for name, (ms, _) in comps.items()}
    else:
        comps = {}
        colors = {name: _component_colors(name, 10) for name in _COMPONENT_ORDER}

    axis_labels = [r'$e/|e|_\mathrm{max}$', r'$j_z/j_c$', r'$j_p/j_c$']
    
    if ranges is None:
        percentile_low, percentile_high = 0.5, 99.5  
        buffer_factor = 0.05
        ranges = []
        for i in range(dims):
            low = np.percentile(X[:, i], percentile_low)
            high = np.percentile(X[:, i], percentile_high)
            span = high - low
            low -= span * buffer_factor
            high += span * buffer_factor
            ranges.append([low, high])
    if dims == 3:
        projects = [[1,0], [1,2], [2,0]]
        width_ratios=[1, 0.25, 1, 0.25, 1, 0.05]
        figsize=(np.sum(width_ratios)*3, 3)
    else:
        projects = [[1,0]]
        width_ratios=[1, 0.05]
        figsize=(np.sum(width_ratios)*3.8, 3)
    fig = plt.figure(figsize=figsize, dpi=150)
    gs = fig.add_gridspec(
        1, len(width_ratios),
        wspace=0,
        hspace=0,
        width_ratios=width_ratios)
    
    N = len(X)
    bins = [min(int(np.ptp(X[:, 0]) / _hist_bin_fd(X[:, 0])), 200), min(int(np.ptp(X[:, 1]) / _hist_bin_fd(X[:, 1])), 200)]
    
    hist_params = {
        'bins': bins,
        'cmap': 'Spectral',
        'cmin': 1,
        'norm': LogNorm(),
    }
    for i, proj in enumerate(projects):
        ax = plt.subplot(gs[2*i])
        im = ax.hist2d(X[:, proj[0]], X[:, proj[1]], range=[ranges[proj[0]],ranges[proj[1]]],**hist_params)
        ax.set_xlabel(f"{axis_labels[proj[0]]}", fontsize=12)
        ax.set_ylabel(f"{axis_labels[proj[1]]}", fontsize=12)
        ax.tick_params(labelsize=8)
        if means is not None and covariances is not None:
            _draw_ellipses(ax, comps, colors, proj)
                
        if threshold_line and i==0:
            ax.axvline(etacut, lw=2, linestyle='--', color='k')
            ax.axhline(ecut, lw=2, linestyle='--', color='k')

    ax = plt.subplot(gs[-1])
    cbar = fig.colorbar(im[3], cax=ax, pad=0)
    cbar.set_label('$N_{*}$', fontsize=12)
    cbar.ax.tick_params(labelsize=12) 
    plt.tight_layout()
    return fig

from scipy.stats import binned_statistic_2d

def _binned_sum_count(x, y, values, x_edges, y_edges):
    """O(N) 2D binning with weighted sum + count (no sort).

    Replacement for scipy.stats.binned_statistic_2d, which internally
    lexsorts the full data (O(N log N)) on every call. Here each particle
    is mapped to its bin once via searchsorted, then bincount accumulates
    the weighted sum and the count in O(N).
    """
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    # Bin mapping matching scipy.binned_statistic_2d edge semantics:
    #  - value == left edge  -> first bin
    #  - value == right edge -> last bin
    #  - value outside range -> discarded
    # searchsorted(side='right')-1 handles the left edge and interior bins,
    # but puts right-edge values one past the last bin -> fix them explicitly.
    ix = np.searchsorted(x_edges, x, side='right') - 1
    iy = np.searchsorted(y_edges, y, side='right') - 1
    ix = np.where(x == x_edges[-1], nx - 1, ix)
    iy = np.where(y == y_edges[-1], ny - 1, iy)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = ix[valid] * ny + iy[valid]
    w = np.asarray(values)[valid]
    counts = np.bincount(flat, minlength=nx * ny).reshape(nx, ny)
    sums = np.bincount(flat, weights=w, minlength=nx * ny).reshape(nx, ny)
    return sums, counts

def plot_surface_density(ax, pos, mass, view='face', size=50, bins=500):
    range_val = (-size, size)
    extent = [range_val[0], range_val[1], range_val[0], range_val[1]]

    cmap = plt.get_cmap('jet').copy() 
    cmap.set_bad('white')
    cmap.set_under('white')

    vmin = 6.5
    vmax = 10.5
    if view == 'face': 
        x,z = pos[:,0], pos[:,1]
        pixel = (2*size)**2/bins**2
        x_edges = np.linspace(range_val[0], range_val[1], bins + 1)
        y_edges = np.linspace(range_val[0], range_val[1], bins + 1)
        stat, _ = _binned_sum_count(x, z, mass, x_edges, y_edges)
    else:
        vmin = 6.5
        vmax = 10.5
        x,z = pos[:,0], pos[:,2]
        x_range = (-size, size) 
        y_range = (-size*0.5, size*0.5)
        bins = [bins, bins//2]
        pixel = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) / (bins[0] * bins[1])
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]] 
        x_edges = np.linspace(x_range[0], x_range[1], bins[0] + 1)
        y_edges = np.linspace(y_range[0], y_range[1], bins[1] + 1)
        stat, _ = _binned_sum_count(x, z, mass, x_edges, y_edges)
    density=np.log10(stat/pixel)
    im = ax.imshow(density.T,extent=extent,origin='lower',cmap=cmap,vmin=vmin,vmax=vmax, interpolation='nearest', rasterized=True)
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    return im

def plot_vlos(ax, pos, vel, mass, size=50, bins=500):
    x_range = (-size, size) 
    y_range = (-size*0.5, size*0.5)
    bins = [bins, bins//2]
    pixel = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) / (bins[0] * bins[1])
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]  
    cmap = plt.get_cmap('turbo').copy() 
    cmap.set_bad('white')
    cmap.set_under('white')
    x,z = pos[:,0], pos[:,2]
    x_edges = np.linspace(x_range[0], x_range[1], bins[0] + 1)
    y_edges = np.linspace(y_range[0], y_range[1], bins[1] + 1)
    # single O(N) binning pass: weighted vlos sum + mass sum + count
    vlos_val = (vel[:,1])/np.sqrt(vel[:,1]**2+3*np.var(vel[:,1]))
    vlos_sum, cnt = _binned_sum_count(x, z, vlos_val, x_edges, y_edges)
    mass_sum, _ = _binned_sum_count(x, z, mass, x_edges, y_edges)
    with np.errstate(divide='ignore', invalid='ignore'):
        vlos = np.where(cnt > 0, vlos_sum / np.maximum(cnt, 1), np.nan)
    density=np.log10(mass_sum/pixel)
    vlos[density < 6.5] = np.nan
    im = ax.imshow(vlos.T,extent=extent,origin='lower',cmap=cmap,vmin=-0.9,vmax=0.9, interpolation='nearest', rasterized=True)
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    return im

# ============================================================================
# Panel specs: declarative description of every panel in the decomposition
# figure. Rendering logic lives in the _render_* functions below, so adding
# a new panel type = one PanelSpec + one render function.
# ============================================================================

@dataclass(frozen=True)
class PanelSpec:
    kind: str          # 'phase_space' | 'density' | 'vlos'
    proj: tuple = ()   # phase_space only: projection axes
    view: str = 'face'  # density only: 'face' | 'edge'


def _render_phase_panel(ax, X, proj, ranges, hist_params, comps, colors,
                        axis_labels, ecut, etacut, threshold_line, fs):
    proj = list(proj)  # tuple would be read as multi-dim index, not columns
    counts, xedges, yedges, im = ax.hist2d(X[:, proj[0]], X[:, proj[1]],
                                           range=[ranges[proj[0]], ranges[proj[1]]],
                                           **hist_params)
    im.set_rasterized(True)
    im.set_clim(vmin=1, vmax=np.nanmax(counts)*1.5)
    _draw_ellipses(ax, comps, colors, proj)
    ax.set_xlabel(f"{axis_labels[proj[0]]}", fontsize=fs['xylabel'])
    ax.set_ylabel(f"{axis_labels[proj[1]]}", fontsize=fs['xylabel'])
    ax.tick_params(labelsize=fs['tick'])
    if threshold_line:
        ax.axvline(etacut, lw=1, linestyle=':', color='k')
        ax.axhline(ecut, lw=1, linestyle=':', color='k')
    return im


def _render_density_panel(ax, particle, view, size, bins, name, color,
                          r50, r90, fs, show_title=True):
    im = plot_surface_density(ax, particle['pos'], particle['mass'],
                              view, size=size, bins=bins)
    if show_title:  # overlay + titles only on the face-on row (historic)
        if name == 'Total':
            circle = plt.Circle((0, 0), r50, fill=False, color='k', linewidth=1,
                                linestyle='--', alpha=0.75)
            ax.add_patch(circle)
            circle = plt.Circle((0, 0), r90, fill=False, color='orange',
                                linewidth=1, linestyle='--', alpha=0.75)
            ax.add_patch(circle)
            fmt = '.2f' if r90 >= 10 and r50 < 10 else '.1f'
            ax.text(0.96, 0.09, f'$r_{{50}} = {r50:{fmt}}$ kpc',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=fs['text'], color='k')
            ax.text(0.96, 0.02, f'$r_{{90}} = {r90:.1f}$ kpc',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=fs['text'], color='orange')
        if name != 'Total':
            ax.set_title(f'{name} ({particle.Mass_frac*100:.0f}%)',
                         color=color, fontsize=fs['title'])
        else:
            ax.set_title(f'{name}', color=color, fontsize=fs['title'])
    return im


def _render_vlos_panel(ax, particle, size, bins):
    return plot_vlos(ax, particle['pos'], particle['vel'], particle['mass'],
                     size=size, bins=bins)


def visualize_decomposition(X, model, galaxy, eoemin_cut, jzojc_cut, ranges=None, threshold_line=False):
    means = model.means_
    covariances = model.covariances_

    _, dims = means.shape 
     
    ecut = eoemin_cut
    etacut = jzojc_cut
    comps = classify_components(means, covariances, ecut, etacut)
    colors = {name: _component_colors(name, len(ms))
              for name, (ms, _) in comps.items()}

    color_map = {
        'Total': 'k',
        'Bulge': 'red', 
        'Halo': 'orange',
        'Cold disk': 'blue',
        'Warm disk': 'green',
        'Counter-rotating disk': 'purple'
    }
    particle_map = {
        'Total': galaxy.s,
        'Bulge': galaxy.bulge,
        'Halo': galaxy.halo,
        'Cold disk': galaxy.colddisk,
        'Warm disk': galaxy.warmdisk,
        'Counter-rotating disk': galaxy.counter_rotating_disk
    }
    names = (['Total']
             + [name for name in _COMPONENT_ORDER if len(comps[name][0]) > 0]
             + ['Color bar'])
    plot_items = []
    for name in names:
        if name == 'Color bar':
            plot_items.append({
            'name': name,
            'color': None,
            'particle': None,
        })
        else:
            color = color_map[name]
            particle = particle_map[name]        
            plot_items.append({
                'name': name,
                'color': color,
                'particle': particle,
            })
    ncol = len(names)

    image_unit = 3
    hsapce = 0.12
    figsize = ((ncol-1)*image_unit+0.05, (3+hsapce)*image_unit)
    axis_labels = [r'$e/|e|_\mathrm{max}$', r'$j_z/j_c$', r'$j_p/j_c$']
    bar_size = 0.05
    bar_label_fontsize = 14
    tick_fontsize = 10
    xylabel_fontsize = 14
    title_fontsize = 14
    text_fontsize = 9
    fig = plt.figure(figsize=figsize)
    outer_gs = fig.add_gridspec(
            2, 1,
            hspace=hsapce*image_unit,
            height_ratios=[0.5,1])  
    
    wspace = 0.33
    if dims == 3: 
        ps_ncol = 8
        width_ratios=[(ncol-0.95-3-2*wspace-bar_size)/2, 1,  wspace, 1,  wspace, 1, bar_size, (ncol-0.95-3-2*wspace-bar_size)/2]
    elif dims == 2:
        ps_ncol = 4
        width_ratios=[(ncol-0.95-1-bar_size)/2, 1, bar_size, (ncol-0.95-1-bar_size)/2]

    ps_gs = outer_gs[0].subgridspec(1, ps_ncol, wspace=0, hspace=0, width_ratios=width_ratios)

    bins = max(min(int(np.ptp(X[:, 0]) / _hist_bin_fd(X[:, 0])), 200), min(int(np.ptp(X[:, 1]) / _hist_bin_fd(X[:, 1])), 200))
    hist_params = {
        'bins': bins,
        'cmap': 'Spectral',
        'cmin': 1,
        'norm': LogNorm(vmin=1),
    }

    if dims == 3: projects = [[1,0], [1,2], [2,0]]
    elif dims == 2: projects = [[1,0]]

    if ranges is None:
        percentile_low, percentile_high = 0.1, 99.9  
        buffer_factor = 0.05
        ranges = []
        for i in range(dims):
            low = np.percentile(X[:, i], percentile_low)
            high = np.percentile(X[:, i], percentile_high)
            span = high - low
            low -= span * buffer_factor
            high += span * buffer_factor
            ranges.append([low, high])

    fs = {'xylabel': xylabel_fontsize, 'tick': tick_fontsize,
          'bar': bar_label_fontsize, 'title': title_fontsize,
          'text': text_fontsize}

    # --- phase-space panels (declarative) ---
    ps_panels = [PanelSpec('phase_space', proj=tuple(p)) for p in projects]
    for k, spec in enumerate(ps_panels):
        ax = plt.subplot(ps_gs[2*k+1])
        im = _render_phase_panel(ax, X, spec.proj, ranges, hist_params, comps,
                                 colors, axis_labels, ecut, etacut,
                                 threshold_line and k == 0, fs)

    ax = plt.subplot(ps_gs[2*k+1+1])
    cbar = fig.colorbar(im, cax=ax, pad=0, extend='max')
    cbar.set_label('$N_{*}$', fontsize=fs['bar'])
    cbar.ax.tick_params(labelsize=fs['tick'])

    # --- surface-density / vlos panels (declarative) ---
    sd_rows = [PanelSpec('density', view='face'),
               PanelSpec('density', view='edge'),
               PanelSpec('vlos')]
    sd_ncol = ncol
    sd_gs = outer_gs[1].subgridspec(3, sd_ncol, wspace=0, hspace=0, width_ratios=[1 for _ in range(sd_ncol-1)]+[0.05], height_ratios=[1,0.5,0.5])
    
    r50 = galaxy.s.r50  # cache: each access re-derives r + full sort (~0.5s)
    r90 = galaxy.s.r90
    size = min(max(np.sqrt(2)*5*r50, 1.2*np.sqrt(2)*r90), 100)
    bin_width = 2*galaxy.properties['eps']
    bins = min(int(2*size/bin_width), 300)
    
    for i, row_spec in enumerate(sd_rows):
        last_im = None  # im of the last non-colorbar panel in this row
        for j, plot_item in enumerate(plot_items):
            name = plot_item['name']
            if name == "Color bar":
                ax = plt.subplot(sd_gs[i, -1])
                cbar = fig.colorbar(last_im, cax=ax, pad=0)
                if i == 0:
                    cbar.set_label(r'$\log_{10} \Sigma_*$/(M$_\odot$ kpc$^{-2}$)', fontsize=fs['bar']-1)
                elif i == 2:
                    cbar.set_label('$v_{los}/\sqrt{v_{los}^{2}+3\sigma_{los}^{2}}$', fontsize=fs['bar']-5)
                cbar.ax.tick_params(labelsize=fs['tick'])
            else:
                ax = plt.subplot(sd_gs[i, j])
                if row_spec.kind == 'density':
                    last_im = _render_density_panel(ax, plot_item['particle'],
                                                    row_spec.view, size, bins,
                                                    name, plot_item['color'],
                                                    r50, r90, fs,
                                                    show_title=(i == 0))
                else:
                    last_im = _render_vlos_panel(ax, plot_item['particle'],
                                                 size, bins)
    del means, covariances
    return fig
