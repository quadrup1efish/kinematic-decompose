import warnings
warnings.filterwarnings('ignore')

import matplotlib
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic_2d

available_fonts = [f.name for f in fm.fontManager.ttflist]

params = {
    'font.family': 'serif',
    'font.serif': 'DejaVu Serif',  
    'font.style': 'normal', 
    'font.weight': 'normal',
    'mathtext.fontset': 'dejavuserif',
    'font.size': 15,
    'legend.frameon': False
}
rcParams.update(params)

def _hist_bin_fd(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)

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

def visualize_residual(X, means, covariances, extent):
    proj = [1,0]
    vmin = np.nanpercentile(X, 1)
    vmax = np.nanpercentile(X, 99)
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
    if means is not None:
        ncs, dims = means.shape
        bulge_index = (means[:, 0] < ecut) & (np.abs(means[:, 1]) < etacut)
        halo_index  = (means[:, 0] > ecut) & (np.abs(means[:, 1]) < etacut)
        warmdisk_index = (means[:, 1] > etacut) & (means[:, 1] < 0.85)
        colddisk_index = (means[:, 1] > 0.85)
        counter_rotate_index = means[:, 1] < -etacut

        bulge_means = means[bulge_index]
        bulge_covariances = covariances[bulge_index]

        halo_means = means[halo_index]
        halo_covariances = covariances[halo_index]

        warmdisk_means = means[warmdisk_index]
        warmdisk_covariances = covariances[warmdisk_index]

        colddisk_means = means[colddisk_index]
        colddisk_covariances = covariances[colddisk_index]

        counter_rotate_means = means[counter_rotate_index]
        counter_rotate_covariances = covariances[counter_rotate_index]

        bulge_sort_idx = bulge_means[:, 0].argsort()
        bulge_means = bulge_means[bulge_sort_idx]
        bulge_covariances = bulge_covariances[bulge_sort_idx]

        halo_sort_idx = halo_means[:, 0].argsort()
        halo_means = halo_means[halo_sort_idx]
        halo_covariances = halo_covariances[halo_sort_idx]

        warmdisk_sort_idx = warmdisk_means[:, 1].argsort()[::-1]
        warmdisk_means = warmdisk_means[warmdisk_sort_idx]
        warmdisk_covariances = warmdisk_covariances[warmdisk_sort_idx]

        colddisk_sort_idx = colddisk_means[:, 1].argsort()[::-1]
        colddisk_means = colddisk_means[colddisk_sort_idx]
        colddisk_covariances = colddisk_covariances[colddisk_sort_idx]

        counter_rotate_sort_idx = np.abs(counter_rotate_means[:, 1]).argsort()[::-1]
        counter_rotate_means = counter_rotate_means[counter_rotate_sort_idx]
        counter_rotate_covariances = counter_rotate_covariances[counter_rotate_sort_idx]

        bulge_ncs = max(len(bulge_means),4) 
        halo_ncs = max(len(halo_means),4) 
        colddisk_ncs = max(len(colddisk_means),4)
        warmdisk_ncs = max(len(warmdisk_means),4)
        counter_rotate_ncs = max(len(counter_rotate_means),4)
    else: 
        bulge_ncs = 10#len(bulge_means)
        halo_ncs = 10#len(halo_means)
        colddisk_ncs = 10#len(colddisk_means)
        warmdisk_ncs = 10#len(warmdisk_means)
        counter_rotate_ncs = 10

    axis_labels = [r'$e/|e|_\mathrm{max}$', r'$j_z/j_c$', r'$j_p/j_c$']

    colors_counter_rotate = [mcolors.to_hex(plt.cm.Purples(t)) 
                         for t in np.linspace(1, 0, counter_rotate_ncs)]

    start_color = np.array(mcolors.to_rgb('darkblue'))
    end_color = np.array(mcolors.to_rgb('lightblue'))
    colors_colddisk = []
    for i in range(colddisk_ncs):
        t = i / (colddisk_ncs - 1)
        color = start_color * (1 - t) + end_color * t
        colors_colddisk.append(mcolors.to_hex(color))

    start_color = np.array(mcolors.to_rgb('darkgreen'))
    end_color = np.array(mcolors.to_rgb('lightgreen'))
    colors_warmdisk = []
    for i in range(warmdisk_ncs):
        t = i / (warmdisk_ncs - 1)
        color = start_color * (1 - t) + end_color * t
        colors_warmdisk.append(mcolors.to_hex(color))

    start_color = np.array(mcolors.to_rgb('darkred'))
    end_color = np.array(mcolors.to_rgb('mistyrose'))
    colors_bulge = []
    for i in range(bulge_ncs):
        t = i / (bulge_ncs-1)#i / (bulge_ncs-1)
        color = start_color * (1 - t) + end_color * t
        colors_bulge.append(mcolors.to_hex(color))

    start_color = np.array(mcolors.to_rgb('darkorange'))
    end_color = np.array(mcolors.to_rgb('peachpuff'))
    colors_halo = []
    for i in range(halo_ncs):
        t = i / (halo_ncs-1)
        color = start_color * (1 - t) + end_color * t
        colors_halo.append(mcolors.to_hex(color))
    
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
            for j, (mean, covariance) in enumerate(zip(bulge_means, bulge_covariances)):
                gaussian_ell(ax, mean[proj], covariance[np.ix_(proj, proj)], colors_bulge[j])
            for j, (mean, covariance) in enumerate(zip(halo_means, halo_covariances)):
                gaussian_ell(ax, mean[proj], covariance[np.ix_(proj, proj)], colors_halo[j])
            for j, (mean, covariance) in enumerate(zip(colddisk_means, colddisk_covariances)):
                gaussian_ell(ax, mean[proj], covariance[np.ix_(proj, proj)], colors_colddisk[j])
            for j, (mean, covariance) in enumerate(zip(warmdisk_means, warmdisk_covariances)):
                gaussian_ell(ax, mean[proj], covariance[np.ix_(proj, proj)], colors_warmdisk[j])
            for j, (mean, covariance) in enumerate(zip(counter_rotate_means, counter_rotate_covariances)):
                gaussian_ell(ax, mean[proj], covariance[np.ix_(proj, proj)], colors_counter_rotate[j])
                
        if threshold_line and i==0:
            ax.axvline(etacut, lw=2, linestyle='--', color='k')
            ax.axhline(ecut, lw=2, linestyle='--', color='k')

    ax = plt.subplot(gs[-1])
    cbar = fig.colorbar(im[3], cax=ax, pad=0)
    cbar.set_label('$N_{*}$', fontsize=12)
    cbar.ax.tick_params(labelsize=12) 
    plt.tight_layout()
    #plt.show()

from scipy.stats import binned_statistic_2d

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
        stat= binned_statistic_2d(x=x,y=z,values=mass,statistic='sum',bins=bins,range=[range_val, range_val])[0]
    else:
        vmin = 7
        vmax = 10.5
        x,z = pos[:,0], pos[:,2]
        x_range = (-size, size) 
        y_range = (-size*0.5, size*0.5)
        bins = [bins, bins//2]
        pixel = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) / (bins[0] * bins[1])
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]] 
        stat= binned_statistic_2d(x=x,y=z,values=mass,statistic='sum',bins=bins,range=[x_range, y_range])[0]
    density=np.log10(stat/pixel)
    im = ax.imshow(density.T,extent=extent,origin='lower',cmap=cmap,vmin=vmin,vmax=vmax, interpolation='nearest')
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
    vlos= binned_statistic_2d(x=x,y=z,values=(vel[:,1])/np.sqrt(vel[:,1]**2+3*np.var(vel[:,1])),statistic='mean',bins=bins, range=[x_range, y_range])[0]
    stat= binned_statistic_2d(x=x,y=z,values=mass,statistic='sum',bins=bins,range=[x_range, y_range])[0]
    density=np.log10(stat/pixel)
    vlos[density < 7] = np.nan
    im = ax.imshow(vlos.T,extent=extent,origin='lower',cmap=cmap,vmin=-1,vmax=1, interpolation='nearest')
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    return im

def visualize_decomposition(X, auto_gmm, galaxy, ranges=None, threshold_line=False):
    means = auto_gmm.best_model.means_
    covariances = auto_gmm.best_model.covariances_

    _, dims = means.shape 
     
    ecut = auto_gmm.ecut 
    etacut= 0.5
    bulge_index = (means[:, 0] < ecut) & (np.abs(means[:, 1]) < etacut)
    halo_index  = (means[:, 0] > ecut) & (np.abs(means[:, 1]) < etacut)
    warmdisk_index = (means[:, 1] > etacut) & (means[:, 1] < 0.85)
    colddisk_index = (means[:, 1] > 0.85)
    counter_rotate_index = means[:, 1] < -etacut

    bulge_means = means[bulge_index]
    bulge_covariances = covariances[bulge_index]

    halo_means = means[halo_index]
    halo_covariances = covariances[halo_index]

    warmdisk_means = means[warmdisk_index]
    warmdisk_covariances = covariances[warmdisk_index]

    colddisk_means = means[colddisk_index]
    colddisk_covariances = covariances[colddisk_index]

    counter_rotate_means = means[counter_rotate_index]
    counter_rotate_covariances = covariances[counter_rotate_index]

    bulge_sort_idx = bulge_means[:, 0].argsort()
    bulge_means = bulge_means[bulge_sort_idx]
    bulge_covariances = bulge_covariances[bulge_sort_idx]

    halo_sort_idx = halo_means[:, 0].argsort()
    halo_means = halo_means[halo_sort_idx]
    halo_covariances = halo_covariances[halo_sort_idx]

    warmdisk_sort_idx = warmdisk_means[:, 1].argsort()[::-1]
    warmdisk_means = warmdisk_means[warmdisk_sort_idx]
    warmdisk_covariances = warmdisk_covariances[warmdisk_sort_idx]

    colddisk_sort_idx = colddisk_means[:, 1].argsort()[::-1]
    colddisk_means = colddisk_means[colddisk_sort_idx]
    colddisk_covariances = colddisk_covariances[colddisk_sort_idx]

    counter_rotate_sort_idx = np.abs(counter_rotate_means[:, 1]).argsort()[::-1]
    counter_rotate_means = counter_rotate_means[counter_rotate_sort_idx]
    counter_rotate_covariances = counter_rotate_covariances[counter_rotate_sort_idx]

    bulge_ncs = max(len(bulge_means),4) 
    halo_ncs = max(len(halo_means),4) 
    colddisk_ncs = max(len(colddisk_means),4)
    warmdisk_ncs = max(len(warmdisk_means),4)
    counter_rotate_ncs = max(len(counter_rotate_means),4)
    
    colors_counter_rotate = [mcolors.to_hex(plt.cm.Purples(t)) 
                         for t in np.linspace(1, 0, counter_rotate_ncs)]

    colors_colddisk = [mcolors.to_hex(np.array(mcolors.to_rgb('darkblue')) * (1 - t) + 
                                    np.array(mcolors.to_rgb('lightblue')) * t)
                    for t in np.linspace(0, 1, colddisk_ncs)]

    colors_warmdisk = [mcolors.to_hex(np.array(mcolors.to_rgb('darkgreen')) * (1 - t) + 
                                    np.array(mcolors.to_rgb('lightgreen')) * t)
                    for t in np.linspace(0, 1, warmdisk_ncs)]

    colors_bulge = [mcolors.to_hex(np.array(mcolors.to_rgb('darkred')) * (1 - t) + 
                                np.array(mcolors.to_rgb('mistyrose')) * t)
                    for t in np.linspace(0, 1, bulge_ncs)]

    colors_halo = [mcolors.to_hex(np.array(mcolors.to_rgb('darkorange')) * (1 - t) + 
                                np.array(mcolors.to_rgb('peachpuff')) * t)
                for t in np.linspace(0, 1, halo_ncs)]

    components = [
        (bulge_means, 'Bulge'), 
        (halo_means, 'Halo'),
        (colddisk_means, 'Cold disk'),
        (warmdisk_means, 'Warm disk'),
        (counter_rotate_means, 'Counter-rotating disk')
    ]
    names = ['Total'] + [name for data, name in components if len(data) > 0] + ['Color bar']
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
    text_fontsize = 10
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

    bins = [min(int(np.ptp(X[:, 0]) / _hist_bin_fd(X[:, 0])), 200), min(int(np.ptp(X[:, 1]) / _hist_bin_fd(X[:, 1])), 200)]
    hist_params = {
        'bins': bins,
        'cmap': 'Spectral',
        'cmin': 1,
        'norm': LogNorm(),
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

    for i, proj in enumerate(projects):
        ax = plt.subplot(ps_gs[2*i+1])
        im = ax.hist2d(X[:, proj[0]], X[:, proj[1]], range=[ranges[proj[0]],ranges[proj[1]]],**hist_params)
        for j, (mean, covariance) in enumerate(zip(bulge_means, bulge_covariances)):
            gaussian_ell(ax, mean[proj], covariance[np.ix_(proj, proj)], colors_bulge[j])
        for j, (mean, covariance) in enumerate(zip(halo_means, halo_covariances)):
            gaussian_ell(ax, mean[proj], covariance[np.ix_(proj, proj)], colors_halo[j])
        for j, (mean, covariance) in enumerate(zip(colddisk_means, colddisk_covariances)):
            gaussian_ell(ax, mean[proj], covariance[np.ix_(proj, proj)], colors_colddisk[j])
        for j, (mean, covariance) in enumerate(zip(warmdisk_means, warmdisk_covariances)):
            gaussian_ell(ax, mean[proj], covariance[np.ix_(proj, proj)], colors_warmdisk[j])
        for j, (mean, covariance) in enumerate(zip(counter_rotate_means, counter_rotate_covariances)):
            gaussian_ell(ax, mean[proj], covariance[np.ix_(proj, proj)], colors_counter_rotate[j])

        ax.set_xlabel(f"{axis_labels[proj[0]]}", fontsize=xylabel_fontsize)
        ax.set_ylabel(f"{axis_labels[proj[1]]}", fontsize=xylabel_fontsize)
        ax.tick_params(labelsize=tick_fontsize)
        if threshold_line and i==0:
            ax.axvline(etacut, lw=1, linestyle=':', color='k')
            ax.axhline(ecut, lw=1, linestyle=':', color='k')

    ax = plt.subplot(ps_gs[2*i+1+1])
    cbar = fig.colorbar(im[3], cax=ax, pad=0)
    cbar.set_label('$N_{*}$', fontsize=bar_label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    sd_ncol = ncol
    sd_gs = outer_gs[1].subgridspec(3, sd_ncol, wspace=0, hspace=0, width_ratios=[1 for _ in range(sd_ncol-1)]+[0.05], height_ratios=[1,0.5,0.5])
    
    map  = {'Total': galaxy.s, 'Bulge': galaxy.bulge, 'Halo': galaxy.halo, 'Cold disk': galaxy.colddisk, 'Warm disk': galaxy.warmdisk, 'Counter-rotating disk': galaxy.counter_rotate_disk}
    colors = ['k', 'red', 'orange', 'blue', 'green', 'purple']
    size = 6.5*galaxy.s.r50
    bin_width = 2*galaxy.properties['eps']
    bins = min(int(2*size/bin_width), 300)
    
    for i in range(3):
        for j, name in enumerate(names):
            if name == "Color bar":
                if i==0:
                    ax = plt.subplot(sd_gs[i,j])
                    cbar = fig.colorbar(im, cax=ax, pad=0)
                    cbar.set_label(r'$\log_{10} \Sigma_*$/(M$_\odot$ kpc$^{-2}$)', fontsize=bar_label_fontsize)
                    cbar.ax.tick_params(labelsize=tick_fontsize)
                elif i==1:
                    ax = plt.subplot(sd_gs[i,j])
                    cbar = fig.colorbar(im, cax=ax, pad=0)
                    #cbar.set_label(r'$\log_{10} \Sigma_*$/(M$_\odot$ kpc$^{-2}$)', fontsize=9)
                    cbar.ax.tick_params(labelsize=tick_fontsize)
                else:
                    ax = plt.subplot(sd_gs[i,-1])
                    cbar = fig.colorbar(im, cax=ax, pad=0)
                    cbar.set_label('$v_{los}/\sqrt{v_{los}^{2}+3\sigma_{los}^{2}}$', fontsize=bar_label_fontsize)
                    cbar.ax.tick_params(labelsize=tick_fontsize)
            else:
                comp = map[name]
                ax = plt.subplot(sd_gs[i,j])
                if i==0:
                    im = plot_surface_density(ax, comp['pos'], comp['mass'], size=size, bins=bins)
                    if j != 0: 
                        ax.set_title(f'{name} ({comp.Mass_frac*100:.0f}%)', color=colors[j], fontsize=title_fontsize)
                    else:
                        ax.set_title(f'{name}', color=colors[j], fontsize=title_fontsize) 
                    """
                    if j != 0:
                        ax.text(0.95, 0.95, f'{comp.Mass_frac:.2f}', 
                        transform=ax.transAxes,
                        ha='right', 
                        va='top',
                        fontsize=text_fontsize,
                        color=colors[j])
                    """
                else:
                    if i == 1:
                        im = plot_surface_density(ax, comp['pos'], comp['mass'], 'edge', size=size, bins=bins)
                    else:
                        im = plot_vlos(ax, comp['pos'], comp['vel'], comp['mass'], size=size, bins=bins) 
        
                    

