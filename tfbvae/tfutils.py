#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 10:19:58 2021

@author: daddona
"""
import os
import time
import datetime
import threading
from pathlib import Path

from tqdm.auto import tqdm

import h5py
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation

import tensorflow as tf
from tensorflow.python.client import device_lib

from . import fakecurses

try:
    from IPython import display
except Exception:
    HAS_IPYTHON = False
else:
    HAS_IPYTHON = True

_do_update_lock = threading.Condition()
_last_update = 0

def getAvailableGpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

##############################################
# An eye candy for showing training progress #
##############################################

def printBar(partial, total=None, wid=32):
    """
    Returns a nice text/unicode progress bar showing
    partial and total progress

    Parameters
    ----------
    partial : float
        Partial progress expressed as decimal value.
    total : float, optional
        Total progress expresses as decimal value.
        If it is not provided or it is None, than
        partial progress will be shown as total progress.
    wid : TYPE, optional
        Width in charachters of the progress bar.
        The default is 32.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    wid -= 2
    prog = int((wid)*partial)
    if total is None:
        total_prog = prog
        common_prog = prog
    else:
        total_prog = int((wid)*total)
        common_prog = min(total_prog, prog)
    pbar_full = '\u2588'*common_prog
    pbar_full += '\u2584'*(total_prog - common_prog)
    pbar_full += '\u2580'*(prog - common_prog)
    return (f"\u2595{{:<{wid}}}\u258F").format(pbar_full)


class BasicPBarCallback(tf.keras.callbacks.Callback):
    """
    Keras callback that shows train progress as a nice progress bar
    """
    TRAIN_HDR_FMT = 'Epoch   Partial/Total Progress            Steps/Total'\
              '   {0:s}     ETA'

    PREDICT_HDR_FMT = '        Progress            Steps/Total'\
              '   {0:s}     ETA'

    TRAIN_BAR_FMT = '\r{0: 5d}  {3: <32s}  {1: 6d}/{2: 6d}  {4}  {5:<10s}'
    PREDICT_BAR_FMT = '\r       {3: <32s}  {1: 6d}/{2: 6d}  {4}  {5:<10s}'

    NULL_VALUE_STRING = '       --       '
    VALUE_FMT = " {:^14.4e} "

    def __init__(self, show_metrics=['loss', 'val_loss']):
        super(BasicPBarCallback, self).__init__()
        self.curr_epoch = 0
        self.t_begin = 0
        self._t_per_batch = None
        self._eta = 0
        self.show_metrics = show_metrics
        self.metrics = {}

    def on_epoch_begin(self, epoch, logs=None):
        self.curr_epoch = epoch + 1

    def on_train_batch_end(self, batch, logs=None):
        total_batches = (self.params['epochs']+1)*self.params['steps']
        elapsed_batches = self.curr_epoch * self.params['steps'] + batch

        if elapsed_batches == 0:
            self._eta = '--'
        else:
            t_per_batch =  (time.time() - self.t_begin) / elapsed_batches
            remaining_batches = total_batches - elapsed_batches
            t_remaining = remaining_batches*t_per_batch

            self._eta = str(datetime.timedelta(seconds=int(t_remaining)))

        if self.params['steps'] == 0:
            pbar_full = printBar(
                0,
                self.curr_epoch/self.params['epochs']
            )
        else:
            pbar_full = printBar(
                (batch+1)/self.params['steps'],
                self.curr_epoch/self.params['epochs']
            )

        loss_str = ''
        for metric in self.show_metrics:
            if logs and metric in logs:
                self.metrics[metric] = logs[metric]
                loss_str += self.VALUE_FMT.format(logs[metric])
            else:
                try:
                    m_val = self.metrics[metric]
                except KeyError:
                    loss_str += self.NULL_VALUE_STRING
                else:
                    if m_val is not None:
                        loss_str += self.VALUE_FMT.format(m_val)
                    else:
                        loss_str += self.NULL_VALUE_STRING

        pbar = self.TRAIN_BAR_FMT.format(
            self.curr_epoch,
            batch+1,
            self.params['steps'],
            pbar_full,
            loss_str,
            self._eta)
        print(pbar, end='\r')

    def on_epoch_end(self, epoch, logs=None):
        for metric in self.show_metrics:
            try:
                self.metrics[metric] = logs[metric]
            except KeyError:
                self.metrics[metric] = None

    def on_train_begin(self, *args):
        self.t_begin = time.time()

        metrics_header = "".join(
            [
                f'{m: ^16s}' if len(m) <=16 else f'{m[:6] +"..."+ m[-7:]: ^16s}'
                for m in self.show_metrics
            ]
        )


        pbar = self.TRAIN_BAR_FMT.format(
            self.curr_epoch,
            1,
            self.params['steps'],
            '',
            '',
            '')
        print("")
        print(self.TRAIN_HDR_FMT.format(metrics_header))
        print(pbar, end='\r')

    def on_train_end(self, *args):
        self.t_end = time.time()
        delta = datetime.timedelta(seconds=int(self.t_end - self.t_begin))
        print(f"\nElapsed time : {delta}")


class AdvancedPBarCallback(tf.keras.callbacks.Callback):
    """
    Keras callback that shows train progress in a curses interface
    """
    TRAIN_HDR_FMT = 'Epoch   Partial/Total Progress            Steps/Total'\
              '   {0:s}     ETA'

    PREDICT_HDR_FMT = '        Progress            Steps/Total'\
              '   {0:s}     ETA'

    TRAIN_BAR_FMT = '\r{0: 5d}  {3: <32s}  {1: 6d}/{2: 6d}  {4}  {5:<10s}'
    PREDICT_BAR_FMT = '\r       {2: <32s}  {0: 6d}/{1: 6d}  {3}  {4:<10s}'

    NULL_VALUE_STRING = '       --       '
    VALUE_FMT = " {:^14.4e} "

    UPDATE_INTERVAL_SEC = 0.1

    def __init__(self, show_metrics=None,
                 stdscr=None, title="", row=1):
        super(AdvancedPBarCallback, self).__init__()
        self.curr_epoch = 0
        self.t_begin = 0
        self._t_per_batch = None
        self._eta = 0
        if show_metrics is None:
            self.show_metrics = []
        else:
            self.show_metrics = show_metrics
        self.metrics = {}
        self.title = title
        self.row = row
        self._called_in_fit = False

        # Curses interface
        if stdscr is None:
            self.stdscr = fakecurses.ncurseinit()
            self._restore_screen = True
        else:
            self.stdscr = stdscr
            self._restore_screen = False

    def __del__(self):
        if self._restore_screen:
            fakecurses.ncursereset(self.stdscr)

    def _printinfo(self, metrics_header, pbar, clear=True):
        global _last_update
        global _do_update_lock
        row = self.row*4
        self.stdscr.addstr(row, 0, f"--- [ {self.title} ] -------------------")
        self.stdscr.addstr(
            row + 1, 0, self.TRAIN_HDR_FMT.format(metrics_header)
        )
        self.stdscr.addstr(row + 2, 0, pbar)
        self.stdscr.noutrefresh()

        if time.time() - _last_update >= self.UPDATE_INTERVAL_SEC:
            _do_update_lock.acquire()
            _last_update = time.time()
            fakecurses.ncursedoupdate(self.stdscr)
            if clear:
                self.stdscr.clear()
            _do_update_lock.release()

    def _get_metrics_header(self):
        metrics_header = "".join([
            f'{m: ^16s}' if len(m) <=16 else f'{m[:6] +"..."+ m[-6:]: ^16s}'
            for m in self.show_metrics
        ])
        return metrics_header

    def _update_show_metrics(self):
        for metric in self.model.metrics_names:
            self.show_metrics.append(metric)
            self.show_metrics.append(f"val_{metric}")

    def _get_loss_str(self, logs):
        loss_str = ''
        for metric in self.show_metrics:
            if logs and metric in logs:
                self.metrics[metric] = logs[metric]
                loss_str += self.VALUE_FMT.format(logs[metric])
            else:
                try:
                    m_val = self.metrics[metric]
                except KeyError:
                    loss_str += self.NULL_VALUE_STRING
                else:
                    if m_val is not None:
                        loss_str += self.VALUE_FMT.format(m_val)
                    else:
                        loss_str += self.NULL_VALUE_STRING
        return loss_str

    def on_epoch_begin(self, epoch, logs=None):
        if not self.show_metrics:
            self._update_show_metrics()
        self.curr_epoch = epoch + 1

    def on_train_batch_end(self, batch, logs=None):
        total_batches = (self.params['epochs']+1)*self.params['steps']
        elapsed_batches = self.curr_epoch * self.params['steps'] + batch
        metrics_header = self._get_metrics_header()

        if elapsed_batches == 0:
            self._eta = '--'
        else:
            t_per_batch =  (time.time() - self.t_begin) / elapsed_batches
            remaining_batches = total_batches - elapsed_batches
            t_remaining = remaining_batches*t_per_batch

            self._eta = str(datetime.timedelta(seconds=int(t_remaining)))

        if self.params['steps'] == 0:
            pbar_full = printBar(
                0,
                self.curr_epoch/self.params['epochs']
            )
        else:
            pbar_full = printBar(
                (batch+1)/self.params['steps'],
                self.curr_epoch/self.params['epochs']
            )

        loss_str = self._get_loss_str(logs)
        pbar = self.TRAIN_BAR_FMT.format(
            self.curr_epoch,
            batch+1,
            self.params['steps'],
            pbar_full,
            loss_str,
            self._eta
        )
        self._printinfo(metrics_header, pbar)

    def on_test_batch_end(self, batch, logs=None):
        if self._called_in_fit:
            return

        if batch == 0:
            self._eta = '--'
        else:
            t_per_batch =  (time.time() - self.t_begin) / batch
            remaining_batches = self.params['steps'] - batch
            t_remaining = remaining_batches*t_per_batch

            self._eta = str(datetime.timedelta(seconds=int(t_remaining)))


        metrics_header = self._get_metrics_header()

        if self.params['steps'] == 0:
            pbar_full = printBar(0)
        else:
            pbar_full = printBar((batch+1)/self.params['steps'])
        loss_str = self._get_loss_str(logs)
        pbar = self.PREDICT_BAR_FMT.format(
            batch+1,
            self.params['steps'],
            pbar_full,
            loss_str,
            self._eta
        )
        self._printinfo(metrics_header, pbar)

    def on_predict_batch_end(self, batch, logs=None):
        self.on_test_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for metric in self.show_metrics:
            try:
                self.metrics[metric] = logs[metric]
            except KeyError:
                self.metrics[metric] = None

    def on_train_begin(self, logs):
        self._called_in_fit = True
        self.t_begin = time.time()
        metrics_header = self._get_metrics_header()
        pbar = self.TRAIN_BAR_FMT.format(
            self.curr_epoch,
            1,
            self.params['steps'],
            '',
            '',
            '')
        self._printinfo(metrics_header, pbar)

    def on_test_begin(self, logs):
        if self._called_in_fit:
            return
        self.t_begin = time.time()
        metrics_header = self._get_metrics_header()
        pbar = self.PREDICT_BAR_FMT.format(
            1,
            self.params['steps'],
            '',
            '',
            '')
        self._printinfo(metrics_header, pbar)

    def on_predict_begin(self, logs):
        self.on_test_begin(self, logs)

    def on_train_end(self, logs):
        self.t_end = time.time()
        metrics_header = "".join([
            f'{m: ^16s}' if len(m) <=16 else f'{m[:6] +"..."+ m[-7:]: ^16s}'
            for m in self.show_metrics
        ])

        pbar_full = printBar(
            self.params['steps']/self.params['steps'],
            self.params['epochs']/self.params['epochs']
        )

        loss_str = self._get_loss_str(logs)
        pbar = self.TRAIN_BAR_FMT.format(
            self.curr_epoch,
            self.params['steps'],
            self.params['steps'],
            pbar_full,
            loss_str,
            self._eta
        )
        self._printinfo(metrics_header, pbar, clear=False)
        self._called_in_fit = False


class AdvancedEarlyStopCallback(tf.keras.callbacks.Callback):
    """
    Advanced verdsion of tf.keras.callbacks.EarlyStopping
    that can monitor multiple losses/metrics
    """

    VALID_MODES = ['min', 'max', 'auto']
    STOP_REASON_STEADY = 0
    STOP_REASON_EXPIRED = 1
    STOP_REASON_FAILED = -1

    def __init__(self, monitor=['loss',], patience=50,
                 mode='auto', min_delta_percent=0.01,
                 tol_treshold_percent=0.01, average_interval=10):
        super(AdvancedEarlyStopCallback, self).__init__()
        self.monitor = monitor
        self.patience = patience

        if not hasattr(monitor, '__iter__') or isinstance(monitor, str):
            raise TypeError(
                "'monitor' param should be a list or an iterable "
                "of metrics to monitor."
            )

        invalid_mode = False
        if hasattr(mode, '__iter__') and not isinstance(mode, str):
            if len(mode) != len(monitor):
                invalid_mode = True
            else:
                for m in mode:
                    if m not in self.VALID_MODES:
                        invalid_mode = True
                        break
        else:
            if mode not in self.VALID_MODES:
                invalid_mode = True
            else:
                mode = [mode, ] * len(monitor)

        if invalid_mode:
            raise ValueError(
                f"'mode' should be {', '.join(self.VALID_MODES)} or a "
                "list of modes of the same lenght of monitor."
            )

        self.monitor_op = []
        self.monitor_tol_op = []
        for i, m in enumerate(mode):
            if mode == 'min':
                self.monitor_op.append(np.less)
                self.monitor_tol_op.append(np.add)
            elif mode == 'max':
                self.monitor_op.append(np.greater)
                self.monitor_tol_op.append(np.subtract)
            else:
                if monitor[i].endswith('loss'):
                    self.monitor_op.append(np.less)
                    self.monitor_tol_op.append(np.add)
                else:
                    self.monitor_op.append(np.greater)
                    self.monitor_tol_op.append(np.subtract)

        self.min_delta_percent = min_delta_percent
        self.tol_treshold_percent = tol_treshold_percent
        self._baseline_epoch = 0
        self._baseline = None
        self._monitor_values = {}
        self._reset_history()
        self._first_update_epoch = 0
        self._delta_epochs = average_interval
        self.stop_reason = None

    def _reset_history(self):
        self._first_update_epoch = 0
        for x in self.monitor:
            self._monitor_values[x] = []

    def _update_history(self, logs):
        if logs is None:
            return
        for metric in self.monitor:
            try:
                metric_val = logs[metric]
            except KeyError:
                continue
            else:
                self._monitor_values[metric].append(metric_val)

    def on_train_begin(self, logs=None):
        self._reset_history()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        if epoch - self._first_update_epoch < self._delta_epochs:
            self._update_history(logs)
            return

        average = [
            np.mean(self._monitor_values[x]) for x in self.monitor
        ]

        if self._baseline is None:
            self._baseline = average
            self._baseline_epoch = epoch
            return

        lo_tresholds = np.multiply(
            self._baseline,
            1 - self.tol_treshold_percent
        )

        hi_tresholds = np.multiply(
            self._baseline,
            1 + self.tol_treshold_percent
        )

        within_tresholds = np.less(average, hi_tresholds)
        within_tresholds &= np.greater(average, lo_tresholds)

        converging = np.zeros_like(within_tresholds, dtype=int)

        for i, m in enumerate(self.monitor):
            min_delta = self.min_delta_percent * self._baseline[i]
            if (
                self.monitor_op[i](average[i], self._baseline[i])
                and abs(self._baseline[i]-average[i]) > min_delta
            ):
                self._baseline[i] = average[i]
                self._baseline_epoch = epoch
                converging[i] = 1
            elif within_tresholds[i] is False:
                converging[i] = -1
            else:
                converging[i] = 0

        self.stop_reason = self.STOP_REASON_EXPIRED
        if np.sum(converging) < 0:
            self.model.stop_training = True
            self.stop_reason = self.STOP_REASON_FAILED

        if epoch > self._baseline_epoch + self.patience:
            self.model.stop_training = True
            if np.sum(converging) == 0:
                self.stop_reason = self.STOP_REASON_STEADY

        self._reset_history()
        self._first_update_epoch = epoch


class SaveLatentSpaceCallback(tf.keras.callbacks.Callback):
    """Save the latetnt space at each epoch."""

    def __init__(self, out_file, data, out_dir='latent_space_history',
                 overwrite=True, notebook_liveview=False, plot_kwargs={}):

        if not out_file.endswith('.h5'):
            out_file = out_file + '.h5'

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.out_dir = out_dir
        self.out_file = os.path.join(out_dir, out_file)
        self.data = data
        self.h5f = None
        self.epoch_record = {}
        self.overwrite = overwrite
        self.notebook_liveview = notebook_liveview
        self._lw = None
        self.plot_kwargs = plot_kwargs

    def __del__(self):
        """
        Close the hdf5 dataset file.

        Returns
        -------
        None.

        """
        if self.h5f:
            self.h5f.close()

    def on_train_begin(self, *args):
        if self.overwrite and os.path.isfile(self.out_file):
            Path.unlink(self.out_file)

        self.h5f = h5py.File(self.out_file, 'w')

    def on_epoch_end(self, epoch, logs=None):
        epoch_label = f'epoch_{epoch}'

        try:
            z_mean, _, _ = self.model.encoder.predict(self.data, verbose=0)
        except AttributeError:
            return
        else:
            self.epoch_record[epoch] = epoch_label
            self.h5f.create_dataset(epoch_label, data=z_mean)

            do_live_update = HAS_IPYTHON and self.notebook_liveview

            if do_live_update:
                if self._lw is None:
                    self._lw = {}
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                    ax.set_axis_off()
                    self._lw['fig'] = fig
                    self._lw['ax'] = ax
                    self._lw['hd'] = display.display("", display_id=True)
                    self._lw['sc'] = ax.scatter(
                        z_mean.T[0],
                        z_mean.T[1],
                        **self.plot_kwargs
                    )
                else:
                    self._lw['ax'].set_xlim(
                        np.min(z_mean.T[0]), np.max(z_mean.T[0])
                    )
                    self._lw['ax'].set_ylim(
                        np.min(z_mean.T[1]), np.max(z_mean.T[1])
                    )
                    self._lw['sc'].set_offsets(z_mean)

                self._lw['hd'].update(self._lw['fig'])


    def on_train_end(self, *args):
        if self.h5f:
            self.h5f.close()

    def saveLatentAnimation(self, dataset_file=None,
                            figsize=(8, 6), static=False, outdir=None,
                            fps=15, metadata={}):
        if outdir is None:
            outdir = self.out_dir

        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        if dataset_file is None:
            dataset_file = self.out_file

        with h5py.File(dataset_file, 'r') as h5f:
            epoch_record = {int(k.split('_')[-1]): k for k in h5f.keys()}

            xmin = None
            xmax = None
            ymin = None
            ymax = None

            writer = animation.writers['ffmpeg'](fps=fps, metadata=metadata)

            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.set_axis_off()

            if static:
                for k in (pbar := tqdm(sorted(epoch_record.keys()))):
                    pbar.set_description("Processing")
                    dataset_key = epoch_record[k]
                    dataset = h5f[dataset_key][()]
                    dsxmin, dsymin = dataset.min(axis=0)
                    dsxmax, dsymax = dataset.max(axis=0)
                    xmin = dsxmin if xmin is None else np.minimum(xmin, dsxmin)
                    ymin = dsymin if ymin is None else np.minimum(ymin, dsymin)
                    xmax = dsxmax if xmax is None else np.maximum(xmax, dsxmax)
                    ymax = dsymax if ymax is None else np.maximum(ymax, dsymax)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

            try:
                with writer.saving(fig, f"{dataset_file}.mp4", 100):
                    sc_plot = None

                    for k in (pbar := tqdm(sorted(epoch_record.keys()))):
                        pbar.set_description("Rendering")
                        dataset_key = epoch_record[k]
                        dataset = h5f[dataset_key][()]

                        if not static:
                            dsxmin, dsymin = dataset.min(axis=0)
                            dsxmax, dsymax = dataset.max(axis=0)
                            ax.set_xlim(dsxmin, dsxmax)
                            ax.set_ylim(dsymin, dsymax)

                        if sc_plot is None:
                            sc_plot = ax.scatter(
                                dataset.T[0],
                                dataset.T[1],
                                **self.plot_kwargs
                            )
                        else:
                            sc_plot.set_offsets(dataset)
                        writer.grab_frame()

            except (Exception, KeyboardInterrupt):
                print("Aborted")
            plt.close(fig)


##############################################
#             Distance Functions             #
##############################################

@tf.function
def tf_pdist_cosine(X):
    """
    Cosine distance.

    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    Parameters
    ----------
    X : TensorFlow Tensor or numpy array
        list of points

    Returns
    -------
    TensorFlow Tensor or numpy array
        The distance matrix.

    """
    # flatten
    X1 = tf.reshape(X, [tf.shape(X)[0], -1])
    # Compute distances
    normalized = tf.nn.l2_normalize(X1, axis=1)
    prod = tf.matmul(normalized, normalized, transpose_b=True)
    D = 1 - prod
    D_shape = tf.shape(D)
    D *= 1 - tf.eye(D_shape[0], D_shape[1])
    return D


@tf.function
def tf_pdist_correlation(X):
    """
    Correlation distance.

    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    Parameters
    ----------
    X : TensorFlow Tensor or numpy array
        list of points

    Returns
    -------
    TensorFlow Tensor or numpy array
        The distance matrix.

    """
    # flatten
    X1 = tf.reshape(X, [tf.shape(X)[0], -1])
    # Compute distances
    X2 =  tf.transpose(tf.transpose(X1) - tf.reduce_mean(X1, axis=1))
    normalized = tf.nn.l2_normalize(X2, axis=1)
    prod = tf.matmul(normalized, normalized, transpose_b=True)
    D = 1 - prod
    D_shape = tf.shape(D)
    D *= 1 - tf.eye(D_shape[0], D_shape[1])
    return D


def tf_pdist_euclidean(X):
    """
    Euclidean distance.

    Parameters
    ----------
    X : TensorFlow Tensor or numpy array
        list of points

    Returns
    -------
    TensorFlow Tensor or numpy array
        The distance matrix.

    """
    # flatten
    X1 = tf.reshape(X, [tf.shape(X)[0], -1])

    r = tf.reduce_sum(X1*X1, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])

    D = r - 2*tf.matmul(X1, X1, transpose_b=True) + tf.transpose(r)
    D_shape = tf.shape(D)
    D *= 1 - tf.eye(D_shape[0], D_shape[1])

    # This are squared distances, no negative number allowed!
    D = tf.maximum(D, 0.0)

    return tf.math.sqrt(D)
