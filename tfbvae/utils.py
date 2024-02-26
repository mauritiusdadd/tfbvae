
import sys
import itertools

import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import median_abs_deviation
from scipy.signal import savgol_filter
from scipy.signal.windows import general_gaussian
from astropy.table import Table

import seaborn as sns
import pandas as pd

from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

import tensorflow as tf

tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)


def smoothFFT(flux, m=1, sigma=25, axis=-1, mask=None):
    flux = np.copy(flux)
    if mask is None:
        mask = np.zeros_like(flux, dtype=bool)
    mask |= ~np.isfinite(flux)

    if len(flux.shape) > 1:
        for j in range(flux.shape[0]):
            flux[j, mask[j]] = np.interp(
                np.flatnonzero(mask[j]),
                np.flatnonzero(~mask[j]),
                flux[j, ~mask[j]]
            )
    else:
        flux[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            flux[~mask]
        )

    XX = np.hstack((flux, np.flip(flux, axis=-1)))
    win = np.roll(general_gaussian(XX.shape[-1], m, sigma), XX.shape[-1]//2)
    fXX = np.fft.fft(XX, axis=-1)
    XXf = np.real(np.fft.ifft(fXX*win))[..., :flux.shape[-1]]
    XXf[mask] = np.nan
    return XXf


def separateContinuum(flux, m=1, sigma=10, mask=None):
    continuum = smoothFFT(flux, m, sigma, mask=mask)
    lines = flux - continuum
    return continuum, lines


def getNormedHist(data, n_nins=200, axis=-1):
    mask = np.isnan(data)

    medianf = np.median(data[~mask], axis=-1)
    registered = data - medianf
    maxf = np.max(np.abs(registered[~mask]), axis=-1)
    data = registered / maxf

    data = np.clip(data, -1, 1)
    bins = np.linspace(-1, 1, n_nins)

    if len(data.shape) > 1:
        hist = np.apply_along_axis(
            lambda x: np.histogram(x, bins)[0],
            axis=-1,
            arr=data
        )
        return (hist, bins)
    else:
        return np.histogram(data, bins)


##############################################
#   Convenient functions for plotting data   #
##############################################

def plotConfusionMatrix(true_labels, predicted_labels, margins=False,
                        normalize=False, fmt=None, actual_label=["True Label"],
                        predicted_label=["Predicted Label"]):
    confusion_matrix = pd.crosstab(
        true_labels,
        predicted_labels,
        rownames=actual_label,
        colnames=predicted_label,
        margins=margins,
        normalize=normalize,
    )

    if fmt is None:
        if not normalize:
            fmt = 'd'
        else:
            fmt = '.2f'

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap=plt.cm.Greens,
        square=True
    )


def scatterDensity2d(poits_x, points_y, nbins=50):
    hh, locx, locy = np.histogram2d(poits_x, points_y, bins=[nbins, nbins])
    color_val = np.array(
        [
            hh[np.argmax(a <= locx[1:]), np.argmax(b <= locy[1:])]
            for a, b in zip(poits_x, points_y)
        ]
    )
    color_val_x = savgol_filter(color_val, 5, 2)
    color_val_x -= np.min(color_val_x)
    color_val_x /= np.max(color_val_x)
    idx_x = color_val_x.argsort()
    return color_val_x, idx_x


def plotScatterDensity2d(points_x, points_y, probs=None, cmap='bone_r',
                         density_mode='cmap', density_alpha_mode='normal',
                         density_color_mode='normal', color='',
                         x_label=None, y_label=None, figsize=(10, 10),
                         nbins=200, log_density=False, ax=None,
                         **kargs):
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    plot_args = {
        's': 25,
        'alpha': 0.5,
        'edgecolors': None,
    }
    plot_args.update(kargs)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None

    if len(points_x) > 0:
        density, ixd = scatterDensity2d(points_x, points_y, nbins=nbins)
    else:
        return fig, ax
    if log_density:
        density = np.log10(1e-5 + density)
    density -= np.min(density)
    density /= np.max(density)

    if density_mode == 'alpha' or density_mode=='both':
        if density_alpha_mode == 'inverse':
            plot_args['alpha'] = plot_args['alpha']*(1 - density)[ixd]
        elif density_alpha_mode == 'normal':
            plot_args['alpha'] = plot_args['alpha']*density[ixd]
        else:
            raise ValueError(
                f"Unknown density alpha mode {density_alpha_mode}"
            )

    if density_mode == 'cmap' or density_mode=='both':
        if cmap:
            plot_args['cmap'] = cmap

        if density_color_mode == 'inverse':
            plot_args['c'] = 1 - density[ixd]
        elif density_color_mode == 'normal':
            plot_args['c'] = density[ixd]
        else:
            raise ValueError(
                f"Unknown density cmap mode {density_color_mode}"
            )
    else:
        if color:
            plot_args['c'] = color

    ax.scatter(points_x[ixd], points_y[ixd], **plot_args)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    return fig, ax


def plotScatterAux(points_x, points_y, aux, points_z=None, probs=None,
                   cmap='jet', x_label=None, y_label=None, z_label=None,
                   aux_label=None, color_norm=plt.Normalize,
                   figsize=(10, 10), sort=False, ax=None, **kargs):
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    points_z = None if points_z is None else np.array(points_z)

    aux = np.array(aux)
    plot_args = {
        's': 10,
        'alpha': 0.7,
        'edgecolors': None
    }
    plot_args.update(kargs)

    if ax is None:
        fig = plt.figure(figsize=figsize, tight_layout=True)
        if points_z is None:
            ax = fig.add_subplot(111)
        else:
            ax = Axes3D(fig)
    else:
        fig = ax.figure

    norm=color_norm(
        vmin=np.min(aux),
        vmax=np.max(aux)
    )

    scmap = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(scmap, anchor=(0, 0.4), ax=ax)

    if aux_label is not None:
        cbar.set_label(aux_label, rotation=270, labelpad=20)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    if sort == 'ascending':
        idx_x = np.argsort(aux)
    elif sort == 'descending':
        idx_x = np.argsort(aux)[::-1]
    else:
        idx_x = list(range(len(aux)))

    if points_z is None:
        ax.scatter(
            points_x[idx_x],
            points_y[idx_x],
            c=aux[idx_x],
            cmap=cmap,
            **plot_args,
        )
    else:
        ax.scatter(
            points_x[idx_x],
            points_y[idx_x],
            points_z[idx_x],
            c=aux[idx_x],
            cmap=cmap,
            **plot_args,
        )
    return fig, ax


def plotScatterCluster2d(points_x, points_y, labels, probs=None, cmap='tab10',
                         x_label=None, y_label=None, figsize=(10, 10),
                         cluster_labels={-1: 'Noise',}, density_bins=200,
                         log_density=False, outlier_ids=[-1], ax=None,
                         legend_markerscale=3, **kargs):
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    plot_args = {
        's': 10,
        'alpha': 0.7,
        'edgecolors': None
    }
    plot_args.update(kargs)

    clusters = list(set(labels))
    palette = sns.color_palette(cmap, len(clusters))

    if density_bins is not None:
        density, sort_idx = scatterDensity2d(
            points_x,
            points_y,
            density_bins
        )
        if log_density:
            density = np.log10(1e-5 + density)
            density -= np.min(density)
            density /= np.max(density)
        density = 0.8*density + 0.2

    else:
        density = None

    if probs is not None:
        cluster_colors = np.array([
            sns.desaturate(palette[clusters.index(col)], sat)
            if col not in outlier_ids else (0.5, 0.5, 0.5) for col, sat in
            zip(labels, probs)
        ])
        sort_idx = np.argsort(probs)
    else:
        cluster_colors = np.array([
            palette[clusters.index(col)]
            if col not in outlier_ids else (0.5, 0.5, 0.5)
            for col in labels
        ])
        sort_idx = None

    if ax is None:
        fig = plt.figure(figsize=figsize, tight_layout=True)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    for c in clusters:
        if sort_idx is not None:
            cluster_mask = labels[sort_idx] == c
            z_x = points_x[sort_idx][cluster_mask]
            z_y = points_y[sort_idx][cluster_mask]
            z_col = cluster_colors[sort_idx][cluster_mask]
            if density is not None:
                plot_args['alpha'] = density[sort_idx][cluster_mask]
        else:
            cluster_mask = labels == c
            z_x = points_x[cluster_mask]
            z_y = points_y[cluster_mask]
            z_col = cluster_colors[cluster_mask]
            if density is not None:
                plot_args['alpha'] = density[cluster_mask]

        try:
            c_label = cluster_labels[c]
        except (KeyError, IndexError):
            try:
                c_label = cluster_labels['fallback']
            except (KeyError, IndexError):
                c_label = "Cluster {c_id}"

        c_label = c_label.format(
            c_id=c,
            c_size=np.sum(cluster_mask)
        )

        ax.scatter(z_x, z_y, c=z_col, label=c_label, **plot_args)

    if legend_markerscale:
        leg = ax.legend(markerscale=legend_markerscale)

    return fig, ax


def plotImputed(data, imputed_data, ft_1, ft_2, nan_vals=[-99],
                 common_missing_data=False,figsize=(10, 12), ax=None):

    if isinstance(data, Table):
        data = data.to_pandas()
    else:
        data = pd.DataFrame(data)

    if isinstance(imputed_data, Table):
        imputed_data = imputed_data.to_pandas()
    else:
        imputed_data = pd.DataFrame(imputed_data)

    ft_1_nan_mask = np.zeros(data.shape[0], dtype=bool)
    ft_2_nan_mask = np.zeros(data.shape[0], dtype=bool)
    ft_1_imp_mask = np.zeros(data.shape[0], dtype=bool)
    ft_2_imp_mask = np.zeros(data.shape[0], dtype=bool)
    common_mask = np.ones(data.shape[0], dtype=bool)
    nonans_mask = np.ones(data.shape[0], dtype=bool)

    for nan_val in nan_vals:
        ft_1_nan_mask |= data[ft_1] == nan_val
        ft_2_nan_mask |= data[ft_2] == nan_val

        ft_1_imp_mask |= imputed_data[ft_1] != nan_val
        ft_2_imp_mask |= imputed_data[ft_2] != nan_val

        for feature in list(data.columns):
            nonans_mask &= data[feature] != nan_val

    ft_1_mask = ft_1_nan_mask & ft_1_imp_mask
    ft_2_mask = ft_2_nan_mask & ft_2_imp_mask

    common_mask &= ft_1_mask & ft_2_mask

    x_label = f'{ft_1}'
    y_label = f'{ft_2}'

    fig, ax = plotScatterDensity2d(
        imputed_data[ft_1][nonans_mask],
        imputed_data[ft_2][nonans_mask],
        figsize=figsize,
        x_label=x_label,
        y_label=y_label,
        density_mode='both',
        density_alpha_mode='normal',
        density_color_mode='inverse',
        cmap='bone',
        s=15,
        alpha=1,
        ax=ax,
    )

    _ = plotScatterDensity2d(
        imputed_data[ft_1][ft_1_mask],
        imputed_data[ft_2][ft_1_mask],
        density_mode='alpha',
        density_alpha_mode='inverse',
        color='#378ddd',
        s=15,
        label=f"imputed {ft_1}",
        alpha=1,
        ax=ax
    )

    _ = plotScatterDensity2d(
        imputed_data[ft_1][ft_2_mask],
        imputed_data[ft_2][ft_2_mask],
        density_mode='alpha',
        density_alpha_mode='inverse',
        color='#fc7a00',
        s=15,
        label=f"imputed {ft_2}",
        alpha=1,
        ax=ax
    )

    if common_missing_data:
        _ = plotScatterDensity2d(
            imputed_data[ft_1][common_mask],
            imputed_data[ft_2][common_mask],
            density_mode='alpha',
            density_alpha_mode='inverse',
            color='red',
            s=15,
            label=f"imputed {ft_1} & {ft_2}",
            alpha=0.2,
            ax=ax
        )

    leg = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        mode="expand",
        ncol=2,
        borderaxespad=0,
        markerscale=3
    )

    return fig, ax


def _plot_tmpax(ax, datax, datay, color, s=40):
    def set_size(ax, w, h):
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)

    ax_bbox = ax.get_window_extent().transformed(
        ax.figure.dpi_scale_trans.inverted()
    )
    ax_width_px = ax_bbox.width*ax.figure.dpi
    ax_height_px = ax_bbox.height*ax.figure.dpi

    tmp_fig = plt.figure(frameon=False)
    tmp_fig.set_dpi(ax.figure.get_dpi())
    tmp_ax = tmp_fig.add_axes([0, 0, 1, 1])

    new_w_in = ax_width_px / tmp_fig.dpi
    new_h_in = ax_height_px / tmp_fig.dpi

    set_size(tmp_ax, new_w_in, new_h_in)

    tmp_ax.patch.set_facecolor("none")
    tmp_ax.patch.set_edgecolor("none")
    tmp_ax.axis('off')
    tmp_ax.scatter(
        datax,
        datay,
        color=color,
        s=s,
        alpha=1,
    )

    tmp_ax.set_xlim(ax.get_xlim())
    tmp_ax.set_ylim(ax.get_ylim())
    tmp_fig.canvas.draw()

    tmp_fig.canvas.print_jpg("/tmp/tmp_ax.jpg")

    fig_w, fig_h = tmp_fig.canvas.get_width_height()
    buff_c = np.fromstring(tmp_fig.canvas.tostring_argb(), dtype=np.uint8)
    buff_c = buff_c.reshape(fig_h, fig_w, 4)[..., [1, 2, 3, 0]]
    tmp_ax.clear()
    plt.close(tmp_fig)
    return buff_c


def plotImputed2(data, imputed_data, ft_1, ft_2, nan_vals=[-99],
                  common_missing_data=False, figsize=(10, 12)):

    if isinstance(data, Table):
        data = data.to_pandas()
    else:
        data = pd.DataFrame(data)

    if isinstance(imputed_data, Table):
        imputed_data = imputed_data.to_pandas()
    else:
        imputed_data = pd.DataFrame(imputed_data)

    ft_1_nan_mask = np.zeros(data.shape[0], dtype=bool)
    ft_2_nan_mask = np.zeros(data.shape[0], dtype=bool)
    ft_1_imp_mask = np.zeros(data.shape[0], dtype=bool)
    ft_2_imp_mask = np.zeros(data.shape[0], dtype=bool)
    common_mask = np.ones(data.shape[0], dtype=bool)
    nonans_mask = np.ones(data.shape[0], dtype=bool)

    for nan_val in nan_vals:
        ft_1_nan_mask |= data[ft_1] == nan_val
        ft_2_nan_mask |= data[ft_2] == nan_val

        ft_1_imp_mask |= imputed_data[ft_1] != nan_val
        ft_2_imp_mask |= imputed_data[ft_2] != nan_val

        for feature in list(data.columns):
            nonans_mask &= data[feature] != nan_val

    ft_1_mask = ft_1_nan_mask & ft_1_imp_mask
    ft_2_mask = ft_2_nan_mask & ft_2_imp_mask

    common_mask &= ft_1_mask & ft_2_mask

    x_label = f'{ft_1}'
    y_label = f'{ft_2}'

    fig, ax = plt.subplots(figsize=figsize)

    _, ax = plotScatterDensity2d(
        imputed_data[ft_1][nonans_mask | ft_1_mask | ft_2_mask],
        imputed_data[ft_2][nonans_mask | ft_1_mask | ft_2_mask],
        x_label=x_label,
        y_label=y_label,
        density_mode='both',
        density_alpha_mode='normal',
        density_color_mode='normal',
        cmap='bone',
        s=15,
        alpha=0.3,
        ax=ax
    )

    ax.set_aspect('equal')
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    buff_1 = _plot_tmpax(
        ax,
        datax=imputed_data[ft_1][ft_1_mask],
        datay=imputed_data[ft_2][ft_1_mask],
        color='#0083ff',
        s=40
    )

    buff_2 = _plot_tmpax(
        ax,
        datax=imputed_data[ft_1][ft_2_mask],
        datay=imputed_data[ft_2][ft_2_mask],
        color='#fc7a00',
        s=40
    )

    legend_patches = [
        mpatches.Patch(color='#717f9a', label='No missing data'),
        mpatches.Patch(color='#0083ff', label=f'Imputed {ft_1}'),
        mpatches.Patch(color='#fc7a00', label=f'Imputed {ft_2}'),
    ]

    if common_missing_data:
        buff_c = _plot_tmpax(
            ax,
            datax=imputed_data[ft_1][common_mask],
            datay=imputed_data[ft_2][common_mask],
            color='red',
            s=40
        )
        legend_patches.append(
            mpatches.Patch(color='red', label=f'Imputed {ft_1} & {ft_2}')
        )

    extent = [x_lim[0], x_lim[1], y_lim[0], y_lim[1]]

    blended = np.zeros_like(buff_1)
    blended[..., 0] = np.minimum(buff_1[..., 0], buff_2[..., 0])
    blended[..., 1] = np.minimum(buff_1[..., 1], buff_2[..., 1])
    blended[..., 2] = np.minimum(buff_1[..., 2], buff_2[..., 2])
    blended[..., 3] = np.minimum(buff_1[..., 3], buff_2[..., 3])

    ax.imshow(buff_1, alpha=0.7, extent=extent, zorder=2)
    ax.imshow(buff_2, alpha=0.7, extent=extent, zorder=3)
    ax.imshow(blended, alpha=0.3, extent=extent, zorder=4)

    if common_missing_data:
        ax.imshow(buff_c, alpha=0.5, extent=extent, zorder=5)

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        handles=legend_patches,
        mode="expand",
        ncol=2,
        borderaxespad=0
    )

    plt.tight_layout()
    return fig, ax


def plotImputed2Color(data, imputed_data, ft_1, ft_2, ft_3, nan_vals=[-99],
                        common_missing_data=False, figsize=(10, 12),
                        ft_colors=['#0083ff', '#fc7a00', '#7f00ff', 'red']):

    if isinstance(data, Table):
        data = data.to_pandas()
    else:
        data = pd.DataFrame(data)

    if isinstance(imputed_data, Table):
        imputed_data = imputed_data.to_pandas()
    else:
        imputed_data = pd.DataFrame(imputed_data)

    ft_1_nan_mask = np.zeros(data.shape[0], dtype=bool)
    ft_2_nan_mask = np.zeros(data.shape[0], dtype=bool)
    ft_3_nan_mask = np.zeros(data.shape[0], dtype=bool)
    ft_1_imp_mask = np.zeros(data.shape[0], dtype=bool)
    ft_2_imp_mask = np.zeros(data.shape[0], dtype=bool)
    ft_3_imp_mask = np.zeros(data.shape[0], dtype=bool)
    common_mask = np.ones(data.shape[0], dtype=bool)
    nonans_mask = np.ones(data.shape[0], dtype=bool)

    for nan_val in nan_vals:
        ft_1_nan_mask |= data[ft_1] == nan_val
        ft_2_nan_mask |= data[ft_2] == nan_val
        ft_3_nan_mask |= data[ft_3] == nan_val

        ft_1_imp_mask |= imputed_data[ft_1] != nan_val
        ft_2_imp_mask |= imputed_data[ft_2] != nan_val
        ft_3_imp_mask |= imputed_data[ft_3] != nan_val

        for feature in list(data.columns):
            nonans_mask &= data[feature] != nan_val

    ft_1_mask = ft_1_nan_mask & ft_1_imp_mask
    ft_2_mask = ft_2_nan_mask & ft_2_imp_mask
    ft_3_mask = ft_3_nan_mask & ft_3_imp_mask

    masks = {
        ft_1: (ft_1_mask, ft_1_imp_mask, ft_1_nan_mask),
        ft_2: (ft_2_mask, ft_2_imp_mask, ft_2_nan_mask),
        ft_3: (ft_3_mask, ft_3_imp_mask, ft_3_nan_mask),
    }

    x_label = f'{ft_1} - {ft_2}'
    y_label = f'{ft_2} - {ft_3}'

    fig, ax = plt.subplots(figsize=figsize)

    common_mask = nonans_mask
    for x in masks.values():
        common_mask |= x[0]

    _, ax = plotScatterDensity2d(
        (imputed_data[ft_1]-imputed_data[ft_2])[common_mask],
        (imputed_data[ft_2]-imputed_data[ft_3])[common_mask],
        x_label=x_label,
        y_label=y_label,
        density_mode='both',
        density_alpha_mode='normal',
        density_color_mode='normal',
        cmap='bone',
        s=30,
        alpha=0.5,
        ax=ax
    )

    legend_patches = [
        mpatches.Patch(color='#717f9a', label='No missing data')
    ]

    buffs = []
    for i, ft in enumerate([ft_1, ft_2, ft_3]):
        buff = _plot_tmpax(
            ax,
            datax=(imputed_data[ft_1]-imputed_data[ft_2])[masks[ft][0]],
            datay=(imputed_data[ft_2]-imputed_data[ft_3])[masks[ft][0]],
            color=ft_colors[i],
            s=40
        )
        buffs.append(buff)
        perc = np.sum(masks[ft][0])/data.shape[0]
        legend_patches.append(
            mpatches.Patch(
                color=ft_colors[i],
                label=f'Imputed {ft} ({perc:.2%})'
            )
        )

    if common_missing_data:
        common_mask = None
        for x in masks.values():
            if common_mask is None:
                common_mask = x[0]
            else:
                common_mask &= x[0]

        buff_c = _plot_tmpax(
            ax,
            datax=(imputed_data[ft_1]-imputed_data[ft_2])[common_mask],
            datay=(imputed_data[ft_2]-imputed_data[ft_3])[common_mask],
            color=ft_colors[i],
            s=40
        )

        legend_patches.append(
            mpatches.Patch(
                color=ft_colors[3],
                label=f'Imputed {ft_1} & {ft_2} & {ft_3}'
            )
        )

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    extent = [x_lim[0], x_lim[1], y_lim[0], y_lim[1]]

    z_order = 2
    for buff in buffs:
        ax.imshow(buff, alpha=0.6, extent=extent, zorder=z_order)
        z_order += 1

    for buff_i, buff_j in itertools.combinations(buffs, 2):
        blended = np.zeros_like(buff_i)
        blended[..., 0] = np.add(buff_i[..., 0], buff_j[..., 0]).clip(0, 255)
        blended[..., 1] = np.add(buff_i[..., 1], buff_j[..., 1]).clip(0, 255)
        blended[..., 2] = np.add(buff_i[..., 2], buff_j[..., 2]).clip(0, 255)
        # NOTE: we use minimum as blending function for the alpha channel
        #       so that only common pionts have null transparency
        blended[..., 3] = np.minimum(buff_i[..., 3], buff_j[..., 3])
        ax.imshow(blended, alpha=0.4, extent=extent, zorder=z_order)
        z_order += 1

    if common_missing_data:
        ax.imshow(buff_c, alpha=0.2, extent=extent, zorder=z_order)

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        handles=legend_patches,
        mode="expand",
        ncol=2,
        borderaxespad=0
    )
    plt.tight_layout()

    return fig, ax


def plotFeatureImportance(importances, std, labels):

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.title("Feature importances")
    plt.bar(
        range(importances.shape[0]),
        importances[indices],
        color="#5eb6ac",
        align="center",
        yerr=std[indices]
    )
    plt.xticks(
        range(len(indices)),
        [labels[i] for i in indices],
        rotation=60,
        ha='right'
    )
    plt.xlim(-1, len(indices))
    plt.ylabel('Relative Importance')
    plt.tight_layout()
    plt.savefig("featureImportance.png")
    plt.show()
    return indices, importances


##############################################
#        Other TF2 utility functions         #
##############################################


def getDatasets(xdata, ydata, train_split=0.6, val_split=0,
                categorical=True, as_tf_dataset=True):
    # Splitting dataset
    ids = np.arange(xdata.shape[0])

    datasets = train_test_split(
        ids,
        xdata,
        ydata,
        test_size=(1-train_split),
        shuffle=True,
    )

    n_input_features = xdata.shape[-1]

    ids_train = datasets[0]
    ids_test = datasets[1]

    X_train = datasets[2]
    X_test = datasets[3]

    y_train = datasets[4]
    y_test = datasets[5]

    if val_split > 0:
        sub_datasets = train_test_split(
            X_train,
            y_train,
            test_size=val_split,
            shuffle=True
        )

        X_train = sub_datasets[0]
        X_val = sub_datasets[1]
        y_train = sub_datasets[2]
        y_val = sub_datasets[3]
        if categorical:
            Y_val_categorical = tf.keras.utils.to_categorical(y_val)
        else:
            Y_val_categorical = y_val

        if as_tf_dataset:
            val_data = tf.data.Dataset.from_tensor_slices(
                (X_val, Y_val_categorical)
            )
        else:
            val_data = (X_val, Y_val_categorical)
    else:
        val_data = None

    if categorical:
        Y_train_categorical = tf.keras.utils.to_categorical(y_train)
        Y_test_categorical = tf.keras.utils.to_categorical(y_test)
        n_noutputs = Y_train_categorical.shape[-1]
    else:
        Y_train_categorical = y_train
        Y_test_categorical = y_test
        n_noutputs = y_train.shape[-1]

    if as_tf_dataset:
        train_data = tf.data.Dataset.from_tensor_slices(
            (X_train, Y_train_categorical)
        )

        test_data = tf.data.Dataset.from_tensor_slices(
            (X_test, Y_test_categorical)
        )
    else:
        train_data = (X_train, Y_train_categorical)
        test_data = (X_test, Y_test_categorical)

    if val_data:
        return (
            (n_input_features, n_noutputs),
            train_data,
            test_data,
            val_data,
            (ids_train, ids_test)
        )
    else:
        return (
            (n_input_features, n_noutputs),
            train_data,
            test_data,
            (ids_train, ids_test)
        )

def permutationImportanceDataset(model, test_dataset, shuffle_index=1,
                                   n_repeats=5, default_metric='accuracy'):
    # NOTE: there is no easy way to shuffle a single column of a
    #       TF2 Dataset, so the only way to do this is to convert
    #       back the datasets to numpy arrays
    xx = np.array([i[0] for i in test_dataset.as_numpy_iterator()])
    yy = np.array([i[1] for i in test_dataset.as_numpy_iterator()])
    return permutationImportance(
        model,
        xx, yy,
        shuffle_index=shuffle_index,
        n_repeats=n_repeats,
        default_metric=default_metric
    )


def permutationImportance(model, x_data, y_data=None, n_repeats=5,
                           use_metric='accuracy', batch_size=32):
    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x

        # Should work for Pandas DataFrame
        try:
            x_numpy = x.ro_numpy()
        except AttributeError:
            pass
        else:
            return x_numpy

        # Should work for Tensorflow Tensor
        try:
            x_numpy = x.numpy()
        except AttributeError:
            pass
        else:
            return x_numpy

        # Let's hope that numpy is able to cast the input to an array
        return np.array(x)

    n_features = x_data.shape[1]

    x_data = to_numpy(x_data)

    if y_data is None:
        y_data = x_data
        use_x_as_y = True
    else:
        y_data = to_numpy(y_data)

    try:
        metric_index = model.metrics_names.index(use_metric)
    except (ValueError, TypeError):
        try:
            metric_index = int(use_metric)
        except (ValueError, TypeError):
            metric_index = 0

    baseline_score = model.evaluate(
        x_data,
        y_data,
        batch_size=batch_size,
        verbose=0,
    )[metric_index]

    importances_mean = np.zeros(n_features)
    importances_std = np.zeros(n_features)

    for i in range(n_features):
        scores = []
        for j in range(n_repeats):
            _progress = (i*n_repeats + j)/(n_features*n_repeats)
            sys.stdout.write(
                f"Computing feature importances {_progress:.2%}\r"
            )
            sys.stdout.flush

            shuffled_x_data = x_data.copy()
            p = np.random.permutation(shuffled_x_data.shape[0])
            shuffled_x_data[:, i] = x_data[p, i]

            if use_x_as_y:
                shuffled_y_data = shuffled_x_data
            else:
                shuffled_y_data = y_data.copy()
                shuffled_y_data[:, i] = y_data[p, i]

            result = model.evaluate(
                shuffled_x_data,
                shuffled_y_data,
                batch_size=batch_size,
                verbose=0,
            )
            scores.append(baseline_score - result[metric_index])
        importances_mean[i] = np.mean(scores)
        importances_std[i] = np.std(scores)
    print("Computing feature importances 100.00%")
    return {
        'importances_mean': importances_mean,
        'importances_std': importances_std
    }
