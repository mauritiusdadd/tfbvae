
import sys
import typing

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.signal import savgol_filter
from scipy.signal.windows import general_gaussian
from astropy.table import Table

import seaborn as sns
import pandas as pd

from matplotlib.cm import ScalarMappable

import tensorflow as tf

tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)


def smooth_fft(
    data: np.array,
    m: float = 1.0,
    sigma: float = 25.0,
    axis: int = -1,
    mask: typing.Optional[np.array] = None
) -> np.array:
    """
    Return a smoothed version of an array.

    Parameters
    ----------
    data : numpy.array
        The input array to be smoothed.
    m : float, optional
        parameter to be passed to the function general_gaussian().
        The default value is 1.0.
    sigma : float, optional
        Parameter to be passed to the function general_gaussian().
        The default value is 25.0.
    axis : int, optional
        The axis along with perform the smoothing. The default value is -1.
    mask : numpy.array, optional
        An optional array containing a boolean mask of values that should be
        masked during the smoothing process, were a True means that the
        corresponding value in the input array is masked.
    Returns
    -------
    numpy.array
        The smoothed array.
    """
    data = np.copy(data)
    if mask is None:
        mask = np.zeros_like(data, dtype=bool)
    mask |= ~np.isfinite(data)

    if len(data.shape) > 1:
        for j in range(data.shape[0]):
            data[j, mask[j]] = np.interp(
                np.flatnonzero(mask[j]),
                np.flatnonzero(~mask[j]),
                data[j, ~mask[j]]
            )
    else:
        data[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            data[~mask]
        )

    xx = np.hstack((data, np.flip(data, axis=axis)))
    win = np.roll(
        general_gaussian(xx.shape[axis], m, sigma),
        xx.shape[axis]//2
    )
    fxx = np.fft.fft(xx, axis=axis)
    xxf = np.real(np.fft.ifft(fxx*win))[..., :data.shape[axis]]
    xxf[mask] = np.nan
    return xxf


def separate_continuum(
    data: np.array,
    m: float = 1.0,
    sigma: float = 10.0,
    mask: typing.Optional[np.array] = None
) -> typing.Tuple[np.array, np.array]:
    """
    Split a numpy array in a smoothed continuum and a residual.

    Parameters
    ----------
    data : numpy.array
        The input array to be smoothed.
    m : float, optional
        parameter to be passed to the function general_gaussian().
        The default value is 1.0.
    sigma : float, optional
        Parameter to be passed to the function general_gaussian().
        The default value is 25.0.
    axis : int, optional
        The axis along with perform the smoothing. The default value is -1.
    mask : numpy.array, optional
        An optional array containing a boolean mask of values that should be
        masked during the smoothing process, were a True means that the
        corresponding value in the input array is masked.

    Returns
    -------
    continuum : np.array
    residuals : np.array
    """
    continuum = smooth_fft(data, m, sigma, mask=mask)
    residuals = data - continuum
    return continuum, residuals


def get_normed_hist(
    data: np.array,
    n_bins: int = 200,
    axis: int = -1
) -> typing.Tuple[np.array, np.array]:
    """
    Compute the histogram of a dataset after normalizing it.
    Parameters
    ----------
    data : numpy.array
        The input array
    n_bins : int, optional
        Number of bins in the histogram. The default value is 200.
    axis : int, optional
        Axis along with compute the histograms. The default value is -1.

    Returns
    -------
    histogram: numpy.array
        The histogram
    bin_edges: numpy.array
        The bin edges.
    """
    medianf = np.nanmedian(data, axis=axis)
    registered = (data.T - medianf)
    maxf = np.nanmax(np.abs(registered), axis=1 - axis)
    data = (registered / maxf).T

    data = np.clip(data, -1, 1)
    bins = np.linspace(-1, 1, n_bins)

    if len(data.shape) > 1:
        hist = np.apply_along_axis(
            lambda x: np.histogram(x, bins)[0],
            axis=axis,
            arr=data
        )
        return (hist.T / np.max(hist, axis=axis)).T, bins
    else:
        return np.histogram(data, bins)


##############################################
#   Convenient functions for plotting data   #
##############################################

def plot_confusion_matrix(true_labels, predicted_labels, margins=False,
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


def scatter_density_2d(poits_x, points_y, nbins=50):
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


def plotscatter_density_2d(points_x, points_y, probs=None, cmap='bone_r',
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
        density, ixd = scatter_density_2d(points_x, points_y, nbins=nbins)
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


def plot_scatter_aux(points_x, points_y, aux, points_z=None, probs=None,
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


def plot_ccatter_cluster_2d(points_x, points_y, labels, probs=None, cmap='tab10',
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
        density, sort_idx = scatter_density_2d(
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


def plo_imputed(data, imputed_data, ft_1, ft_2, nan_vals=[-99],
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

    fig, ax = plotscatter_density_2d(
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

    _ = plotscatter_density_2d(
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

    _ = plotscatter_density_2d(
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
        _ = plotscatter_density_2d(
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


def plot_feature_impotrance(importances, std, labels):

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

def get_permutation_importance_dataset(model, test_dataset, shuffle_index=1,
                                   n_repeats=5, default_metric='accuracy'):
    # NOTE: there is no easy way to shuffle a single column of a
    #       TF2 Dataset, so the only way to do this is to convert
    #       back the datasets to numpy arrays
    xx = np.array([i[0] for i in test_dataset.as_numpy_iterator()])
    yy = np.array([i[1] for i in test_dataset.as_numpy_iterator()])
    return permutation_importance(
        model,
        xx, yy,
        shuffle_index=shuffle_index,
        n_repeats=n_repeats,
        default_metric=default_metric
    )


def permutation_importance(model, x_data, y_data=None, n_repeats=5,
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
