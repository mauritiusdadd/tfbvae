#!/usr/bin/env python3
"""
Thi is a simple implementation of a Beta-Variational Autoencoder.
You can choose among tree different loss functions.

This is still a prototype so it still has poor documentation
"""

__author__ = "Maurizio D'Addona"
__copyright__ = "Copyright 2021, Maurizio D'Addona"
__credits__ = ["Maurizio D'Addona"]
__license__ = "GPL"
__version__ = "0.1.1"
__maintainer__ = "Maurizio D'Addona"
__email__ = "mauritiusdadd@gmail.com"
__status__ = "Prototype"

import os
import sys
import time
import typing
import concurrent.futures

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from tensorflow.python.client import device_lib

from . import tfutils

# Local directory to save module testing results
TEST_OUT_DIR = "tfvae_test_out"


class Sampling(layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """

    def __init__(self, mu=0, std=1):
        super(Sampling, self).__init__()

        self._mu = mu
        self._std = std

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(
            shape=(batch, dim),
            mean=self._mu,
            stddev=self._std
        )
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def get_test_encoder(
        n: int,
        l1: typing.Optional[float] = None,
        l2: typing.Optional[float] = None
) -> typing.List[layers.Dense]:
    """
    Returns a simple encoder layer

    Parameters
    ----------
    n : integer
        Regulates the number of neurons in eache layer:
            - The first layer will have 2*N neurons
            - The second layer will have N + 1 neurons
    l1 : float, optional
        The default is None. Regulates the l1 regularization
    l2 : float, optional
        The default is None. Regulates the l2 regularization

    Returns
    -------
    enc_layers : tuple
        A tuple of layers representing the encoder.

    """
    enc_layers = [
        layers.Dense(
            2 * n,
            activation="relu",
            name="enc_layer_1",
            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            activity_regularizer=regularizers.l1_l2(l1=l1, l2=l2)
        ),
        layers.Dense(
            n + 1,
            activation="relu",
            name="enc_layer_2",
            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            activity_regularizer=regularizers.l1_l2(l1=l1, l2=l2)
        )
    ]
    return enc_layers


def get_test_decoder(
        n: int,
        l1: typing.Optional[float] = None,
        l2: typing.Optional[float] = None
) -> typing.List[layers.Dense]:
    """
    Returns a simple encoder layer.

    Parameters
    ----------
    n : integer
        Regulates the number of neurons in eache layer:
            - The first layer will have N + 1 neurons
            - The second layer will have 2*N neurons
    l1 : float, optional
        The default is None. Regulates the l1 regularization
    l2 : float, optional
        The default is None. Regulates the l2 regularization

    Returns
    -------
    dec_layers : tuple
        Tuple of layers representing the decoder.

    """
    dec_layers = [
        layers.Dense(
            n + 1,
            activation="relu",
            name="dec_layer_1",
            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            activity_regularizer=regularizers.l1_l2(l1=l1, l2=l2)
        ),
        layers.Dense(
            2 * n,
            activation="relu",
            name="dec_layer_2",
            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            activity_regularizer=regularizers.l1_l2(l1=l1, l2=l2)
        ),
        layers.Dense(
            n,
            activation=None,
            name="dec_layer_3",
        )
    ]
    return dec_layers


def get_test_encoder_cnn(
        n: int,
        l1: typing.Optional[float] = None,
        l2: typing.Optional[float] = None
) -> typing.List[typing.Union[layers.Layers, ...]]:
    """
    Returns a simple CNN encoder layer

    Parameters
    ----------
    n : integer
        Regulates the number of neurons in the fully connected section:
            - The first layer will have 2*N neurons
            - The second layer will have N + 1 neurons
    l1 : float, optional
        The default is None. Regulates the l1 regularization
    l2 : float, optional
        The default is None. Regulates the l2 regularization

    Returns
    -------
    enc_layers : list
        List of layers representing the encoder.

    """
    enc_layers = [
        layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            activation='relu',
            name='cnn_enc_layer_1'
        ),
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            activation='relu',
            name='cnn_enc_layer_2'
        ),
        layers.Flatten(),
        layers.Dense(
            2 * n,
            activation="relu",
            name='cnn_enc_layer_3',
            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            activity_regularizer=regularizers.l1_l2(l1=l1, l2=l2)
        ),
        layers.Dense(
            n + 1,
            activation="relu",
            name='cnn_enc_layer_4',
            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            bias_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            activity_regularizer=regularizers.l1_l2(l1=l1, l2=l2)
        )
    ]
    return enc_layers


def get_test_decoder_cnn(
        n: int,
        l1: typing.Optional[float] = None,
        l2: typing.Optional[float] = None
) -> typing.List[typing.Union[layers.Layers, ...]]:
    """
    Returns a simple CNN decoder layer

    Parameters
    ----------
    n : integer
        Regulates the size of output image that will be N x N.
        For MNIST data use N=28
    l1 : float, optional
        The default is None. Regulates the l1 regularization
    l2 : float, optional
        The default is None. Regulates the l2 regularization

    Returns
    -------
    dec_layers : list
        List of layers representing the decoder.

    """
    n = int(n / 4)

    dec_layers = [
        layers.Dense(
            units=n * n * 32,
            activation=tf.nn.relu,
            name='cnn_dec_layer_1'
        ),
        layers.Reshape(
            target_shape=(n, n, 32),
            name='cnn_dec_layer_2'
        ),
        layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name='cnn_dec_layer_3'
        ),
        layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu',
            name='cnn_dec_layer_4'
        ),
        layers.Conv2DTranspose(
            filters=1,
            kernel_size=3,
            strides=1,
            padding='same',
            name='cnn_dec_layer_5'
        ),
    ]
    return dec_layers


class BetaVariationalAutoencoder(tf.keras.Model):
    """
    A simple Beta-Variational Autoencoder

    Three different loss functions are available:

        - classic:
            The classic KL loss function used in Variational Autoencoders
            (See: [1])
        - tsne:
            The KL divergence used in t-SNE (See: [2])
        - distance:
            A classic KL loss function but with an additional term that
            is minimum when objects in latent space have a similar distance
            distribution to the distances in the original feature space (i.e.
            objetcs that are near in the feature space are near also in the
            latent space)

    For losses that depends on distribution of distances (like 'tsne' and
    'distance') the following metrics are available to compute pairwise
    distances between points:

        - euclidean:
            The standard euclidean distance
        - cosine:
            The coside distance (See [3])
        - correlation:
            The correlation distance (See [3])

    References
    ----------
    [1] https://keras.io/examples/generative/vae/
    [2] https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
    [3] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    """

    METHODS = [
        "classic",
        "tsne",
        "distance"
    ]

    METRICS = {
        "euclidean": tfutils.tf_pdist_euclidean,
        "correlation": tfutils.tf_pdist_correlation,
        "cosine": tfutils.tf_pdist_cosine,
    }

    def __init__(
            self,
            encoder_layers: typing.List[layers.Layer, ...],
            decoder_layers: typing.List[layers.Layer, ...],
            input_shape: typing.Union[typing.List[int, ...], np.array],
            latent_dim: int = 2,
            beta: float = 0.7,
            verbose: bool = False,
            flatten_input: bool = False
    ):
        """
        Parameters
        ----------
        encoder_layers : list
            A list of keras layers representing the encoder.
        decoder_layers : list
            A list of keras layers representing the decoder.
        input_shape : list
            The shape of input data.
        latent_dim : int
            Number of dimensions of the latent space. The default is 2.
        beta : float, optional
            Beta value. This value multipy the KL loss making this term more
            relevant in respect with the reconstruction term.
            The default is 0.7.
        verbose : bool, optional
            Indicates wheter to be verbose or not. The default is False.
        flatten_input : bool, optional
            Indicates wheter to flatten the input berfore passing it
            to the encoder. The default is False.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(BetaVariationalAutoencoder, self).__init__()

        self._inp_shape = input_shape
        self._beta = beta

        self.latent_dim = latent_dim
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.rec_loss_tracker = keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        # Encoder
        encoder_inputs = keras.Input(shape=input_shape)

        if len(input_shape) > 1 and flatten_input:
            x = layers.Flatten()(encoder_inputs)
        else:
            x = encoder_inputs

        for enc_layer in encoder_layers:
            x = enc_layer(x)

        x = layers.Flatten()(x)
        x = layers.BatchNormalization()(x)

        z_mean = layers.Dense(
            latent_dim,
            activation=None,
            name="z_mean"
        )(x)
        z_log_var = layers.Dense(
            latent_dim,
            activation=None,
            name="z_log_var"
        )(x)
        z = Sampling()([z_mean, z_log_var])

        self.encoder = keras.Model(
            encoder_inputs,
            [z_mean, z_log_var, z],
            name="encoder"
        )

        # Decoder
        latent_inputs = keras.Input(shape=(latent_dim,))

        x = latent_inputs
        for dec_layer in decoder_layers:
            x = dec_layer(x)

        x = layers.Flatten()(x)

        x = layers.Dense(
            np.prod(input_shape),
            activation='linear'
        )(x)

        decoder_outputs = layers.Reshape(
            input_shape,
            name="dec_output_reshape",
        )(x)

        self.decoder = keras.Model(
            latent_inputs,
            decoder_outputs,
            name="decoder"
        )
        if verbose:
            self.encoder.summary()
            self.decoder.summary()

    def summary(self) -> None:
        """
        Get the summaries of the encoder and the decoder.

        Returns
        -------
        None
        """
        self.encoder.summary()
        self.decoder.summary()

    def call(
            self,
            inputs: typing.Union[tf.data.Dataset, np.array]
    ) -> np.array:
        z_mean, z_log_var, z = self.encoder(inputs, training=False)
        decoder_out = self.decoder(z, training=False)
        return decoder_out

    @tf.function
    def train_step(
            self,
            data: typing.Union[tf.data.Dataset, np.array]
    ) -> typing.Dict[str, typing.Union[tf.Tensor, np.array, float]]:
        """
        This function is called within fit() method.

        Parameters
        ----------
        data : ndarray
            The input data used to train the model.

        Returns
        -------
        metrics_dict : dict
            Dictionary containing the test metrics.

        """
        dims = list(range(1, len(self._inp_shape)))

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            y_pred = self.decoder(z, training=True)

            if not dims:
                rec_loss = keras.losses.mean_squared_error(data, y_pred)
            else:
                rec_loss = tf.reduce_sum(
                    keras.losses.mean_squared_error(data, y_pred),
                    axis=dims
                )

            kl_loss = (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=1)

            total_loss = 2 * (
                        (1 - self._beta) * rec_loss + self._beta * kl_loss)
            total_loss += sum(self.losses)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        metrics_dict = {}

        for metric in self.metrics:
            metric.update_state(data, y_pred)
            metrics_dict[metric.name] = metric.result()

        metrics_dict['loss'] = self.total_loss_tracker.result()
        metrics_dict['rec_loss'] = self.rec_loss_tracker.result()
        metrics_dict['kl_loss'] = self.kl_loss_tracker.result()

        return metrics_dict

    @tf.function
    def test_step(self, data):
        """
        This function is called within evaluate() method.

        Parameters
        ----------
        data : tuple
            A tuple containing the input data and target data in the form
            [input_data, target_data].

        Returns
        -------
        metrics_dict : dict
            Dictionary containing the test metrics.

        """
        dims = list(range(1, len(self._inp_shape)))

        z_mean, z_log_var, z = self.encoder(data, training=False)
        y_pred = self.decoder(z_mean, training=False)

        if not dims:
            rec_loss = keras.losses.mean_squared_error(data, y_pred)
        else:
            rec_loss = tf.reduce_sum(
                keras.losses.mean_squared_error(data, y_pred),
                axis=dims
            )

        kl_loss = (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = -0.5 * tf.reduce_sum(kl_loss, axis=1)

        total_loss = 2 * ((1 - self._beta) * rec_loss + self._beta * kl_loss)
        total_loss += sum(self.losses)

        self.total_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        metrics_dict = {}

        for metric in self.metrics:
            metric.update_state(data, y_pred)
            metrics_dict[metric.name] = metric.result()

        metrics_dict['loss'] = self.total_loss_tracker.result()
        metrics_dict['rec_loss'] = self.rec_loss_tracker.result()
        metrics_dict['kl_loss'] = self.kl_loss_tracker.result()

        return metrics_dict

    @tf.function
    def predict_step(self, data):
        """
        This function is called within predict() method.

        Parameters
        ----------
        data : ndarray
            The input data.

        Returns
        -------
        z_mean : ndarray
            Encoder prediction.

        z_log_var: ndarray
            Variance in the latent space associated to z_mean

        z: ndarray
            Random sample taken from the gaussian distributions described
            by z_mean, z_log_var.
        """
        z_mean, z_log_var, z = self.encoder(data, training=False)
        return z_mean, z_log_var, z


def plot_history(history):
    """
    Plot the train history

    Parameters
    ----------
    history : keras train history
        The train history returned by the fit() method.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object of the plot.

    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(history.history['loss'], label='training loss')
    ax.plot(history.history['rec_loss'], label='reconstruction loss')
    ax.plot(history.history['kl_loss'], label='kl loss')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_label_clusters(data, labels, title="", cmap='tab10'):
    """
    Make a scatter with points colormapperd according to their
    respective labels.

    Parameters
    ----------
    data : numpy array
        The inpud data.
    labels : list or numpy array
        The labels.
    title : string, optional
        Title of the plot. The default is "".
    cmap : string or colormap, optional
        The colormap used to color the points.
        The default is 'tab10'.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    # display a 2D plot of the digit classes in the latent space
    labels = np.array(labels)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    z_x = data.T[0]
    z_y = data.T[1]

    target_labels = list(set(labels))
    palette = sns.color_palette(cmap, len(target_labels))
    cluster_colors = np.array([palette[col] for col in labels])

    for c in target_labels:
        label_mask = labels == c

        ax.scatter(
            z_x[label_mask],
            z_y[label_mask],
            s=5,
            c=cluster_colors[label_mask],
            alpha=0.9,
            cmap=cmap,
            label=f"{c}"
        )

    ax.set_xlabel("latent space x")
    ax.set_ylabel("latent space y")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def gen_latent_tiles(vae, n=64, scale=3.0):
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    xv, yv = np.meshgrid(grid_x, grid_y)

    grid_coord = np.empty((n, n, 2))

    grid_coord[..., 0] = xv
    grid_coord[..., 1] = yv

    z_sample = grid_coord.reshape((n * n, 2))

    x_decoded = vae.decoder.predict(z_sample, batch_size=128)
    x_decoded = np.mean(x_decoded, axis=-1)
    tiles = x_decoded.reshape((n, n, x_decoded.shape[1], x_decoded.shape[2]))

    return tiles, grid_coord


def plot_latent_space_from_tiles(tiles, grid_coord,
                                 figsize=25, cmap="Greys_r", title=""):
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))

    tile_x_size = tiles.shape[2]
    tile_y_size = tiles.shape[3]

    flat_img = np.hstack(np.hstack(tiles))

    start_x_range = tile_x_size // 2
    end_x_range = grid_coord.shape[0] * tile_x_size + start_x_range
    pixel_range_x = np.arange(start_x_range, end_x_range, tile_x_size)

    start_y_range = tile_y_size // 2
    end_y_range = grid_coord.shape[1] * tile_y_size + start_y_range
    pixel_range_y = np.arange(start_y_range, end_y_range, tile_y_size)

    sample_range_x = np.round(grid_coord[:, 0, 0], 1)
    sample_range_y = np.round(grid_coord[:, 1, 1], 1)

    ax.set_xticks(pixel_range_x, sample_range_x)
    ax.set_yticks(pixel_range_y, sample_range_y)

    ax.set_xlabel("latent space x")
    ax.set_ylabel("latent space y")
    ax.set_title(title)
    ax.imshow(flat_img, cmap=cmap)
    plt.tight_layout()
    return fig


def plot_latent_space(vae, n=64, figsize=25, scale=3.0,
                      cmap="Greys_r", title=""):
    tiles, grid = gen_latent_tiles(vae, n, scale)

    return plot_latent_space_from_tiles(
        tiles,
        grid,
        figsize,
        cmap,
        title
    )


def test_bvae(data, labels, experiment_name="Unknown",
              epochs=500, batch_size=128, beta=1, device=None,
              pb_callback_row=0, pb_callback_title=""):
    test_params = {
        'experiment_name': experiment_name,
        'beta': beta,
        'device': str(device),
        'epochs': epochs,
        'batch_size': batch_size,
        'labels': labels,
    }

    with tf.device(device):
        print("Building vae model...")
        autoencoder = BetaVariationalAutoencoder(
            get_test_encoder_cnn(32, l1=1e-5, l2=1e-4, ),
            get_test_decoder_cnn(data.shape[1], l1=1e-5, l2=1e-4, ),
            input_shape=data.shape[1:],
            latent_dim=2,
            beta=beta,
            flatten_input=False,
            verbose=False,
        )

        print("Compiling model...")
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            metrics=['mae']
        )

        print("Training model...")
        callback_es = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=50,
            mode='min',
            min_delta=1e-4
        )

        callback_pb = tfutils.AdvancedPBarCallback(
            show_metrics=[
                'loss',
                'rec_loss',
                'kl_loss',
            ],
            row=pb_callback_row,
            title=pb_callback_title
        )

        history = autoencoder.fit(
            data,
            epochs=epochs,
            shuffle=True,
            batch_size=batch_size,
            callbacks=[callback_es, callback_pb],
            verbose=0,
        )

        print("Testing model")
        eval_results = autoencoder.evaluate(
            data,
            batch_size=batch_size,
        )

        test_params['eval_results'] = eval_results

        z_mean, _, _ = autoencoder.predict(
            data,
            batch_size=batch_size
        )
        print(np.max(np.abs(z_mean)))
        latent_tiles = gen_latent_tiles(
            autoencoder,
            scale=np.max(np.abs(z_mean))
        )

        return history, z_mean, test_params, latent_tiles


def test_mnist():
    def preprocess_images(images):
        images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
        return np.where(images > .5, 1.0, 0.0).astype('float32')

    def generate_jobs(datasets, beta_values, epochs=30):
        jobs = []
        for dataset_name, dataset in datasets.items():
            print(f"Loading {dataset_name} dataset...")
            (train_im, train_l), (test_im, test_l) = dataset.load_data()

            images = np.concatenate([train_im, test_im], axis=0)
            labels = np.concatenate([train_l, test_l], axis=0)
            images = preprocess_images(images)

            for beta in beta_values:
                jobs.append(
                    {
                        'data': images,
                        'labels': labels,
                        'dataset_name': dataset_name,
                        'epochs': epochs,
                        'beta': beta
                    }
                )
        return jobs

    datasets = {
        "MNIST": tf.keras.datasets.mnist,
        "Fashion MNIST": tf.keras.datasets.fashion_mnist,
    }

    tf_devices = tfutils.get_available_gpus()

    threads = [None, ] * len(tf_devices)

    print("Generating tests...")
    jobs = generate_jobs(
        datasets,
        [0.3, 0.5, 0.9],
        epochs=10
    )

    if not os.path.isdir(TEST_OUT_DIR):
        os.mkdir(TEST_OUT_DIR)

    # Mulithreading scheduler for parallel execution on multiple GPUs
    job_count = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while jobs:
            try:
                # check if there is a free device
                job_device_id = -1
                for thread_id, thread in enumerate(threads):
                    if thread is None:
                        job_device_id = thread_id
                        break
                    elif thread.done():
                        history, pred, job_params, tiles = thread.result()

                        threads[thread_id] = None
                        # NOTE: matplotlib is not thread safe so plotting must
                        #       be done here in the main thread!

                        fig = plot_history(history)
                        fig_name = str(job_params['experiment_name'])
                        fig_name += f"_history_{job_params['beta']}.jpg"
                        plt.savefig(
                            os.path.join(TEST_OUT_DIR, fig_name),
                            dpi=300
                        )
                        plt.close(fig)

                        fig_title = str(job_params['experiment_name'])
                        fig_title += f" (beta={job_params['beta']:.2f})"

                        fig = plot_label_clusters(
                            pred,
                            job_params['labels'],
                            title=fig_title
                        )

                        fig_name = str(job_params['experiment_name'])
                        fig_name += f"_latentspace_{job_params['beta']}.jpg"
                        plt.savefig(
                            os.path.join(TEST_OUT_DIR, fig_name),
                            dpi=300
                        )
                        plt.close(fig)

                        fig = plot_latent_space_from_tiles(tiles[0], tiles[1])
                        fig_name = str(job_params['experiment_name'])
                        fig_name += f"_tiles_{job_params['beta']}.jpg"
                        plt.savefig(
                            os.path.join(TEST_OUT_DIR, fig_name),
                            dpi=300
                        )
                        plt.close(fig)

                time.sleep(0.5)

                if job_device_id >= 0:
                    job = jobs.pop()
                    exp_name = f"bvae_test_{job['dataset_name']}"
                    threads[job_device_id] = executor.submit(
                        test_bvae,
                        job['data'],
                        job['labels'],
                        experiment_name=exp_name,
                        epochs=job['epochs'],
                        beta=job['beta'],
                        device=tf_devices[job_device_id],
                        pb_callback_row=job_device_id,
                        pb_callback_title=(
                            f"job {job_count} on {job['dataset_name']}"
                        )
                    )
                    job_count += 1
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\nAbort signal received: waiting for running "
                      "threads to complete!\n\n")
                for thread_id, thread in enumerate(threads):
                    if thread is None or thread.done():
                        continue
                    elif thread.cancel():
                        continue
                    else:
                        while not thread.done():
                            time.sleep(1)
                sys.exit()


def test():
    test_mnist()


if __name__ == '__main__':
    test()
