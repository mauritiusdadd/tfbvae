#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 13:55:04 2022.

@author: daddona
"""
import copy
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import callbacks as callbacks_module


class LBFGWrapper():
    """
    Wrapper for tfp.optimizer.lbfgs_minimize.

    A wrapper class that permits to use TF keras models
    with LBFG optimizer from tfp.
    """

    def __init__(self, model):
        self.model = model
        self._x_train = None
        self._y_train = None
        self._callbacks = None
        self._verbose = True

        self.shapes = tf.shape_n(model.trainable_variables)
        self.n_tensors = len(self.shapes)

        # we'll use tf.dynamic_stitch and tf.dynamic_partition later,
        # so we need to prepare required information first
        count = 0
        self.idx = []  # stitch indices
        self.part = []  # partition indices

        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            self.idx.append(
                tf.reshape(
                    tf.range(count, count+n, dtype=tf.int32),
                    shape
                )
            )
            self.part.extend([i]*n)
            count += n

        self.part = tf.constant(self.part)

        self._current_epoch = tf.Variable(0)
        self.history = []

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @tf.function
    def assign_new_model_parameters(self, params_1d):
        """
        Update the model's parameters with a 1D tf.Tensor.

        Parameters
        ----------
            params_1d [in]: a 1D tf.Tensor representing the model's trainable
            parameters.
        """
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, params)):
            self.model.trainable_variables[i].assign(
                tf.reshape(param, shape)
            )

    @tf.function
    def train_step(self, params_1d):
        """
        Execute one training step.

        Parameters
        ----------
        params_1d : a 1D tf.Tensor
            Parameter passed from lbfg_minimize.

        Returns
        -------
        loss_value : TYPE
            DESCRIPTION.
        grads : TYPE
            DESCRIPTION.

        """
        # update the parameters in the model
        self.assign_new_model_parameters(params_1d)

        # use GradientTape so that we can calculate the gradient
        # of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # calculate the loss
            reconstruction = self.model(self._x_train, training=True)
            loss_value = self.model.compiled_loss._losses(
                reconstruction, self._y_train
            )

        # NOTE: this must be outside the with-as section
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        grads = tf.dynamic_stitch(self.idx, grads)

        self.loss_tracker.update_state(loss_value)

        # Saving history
        tf.py_function(self.history.append, inp=[loss_value], Tout=[])

        # print out iteration & loss
        self._current_epoch.assign_add(1)

        return loss_value, grads

    def fit(self, x_train, y_train, epochs, callbacks=[], verbose=True,
            **args):
        self._x_train = x_train
        self._y_train = y_train
        self._verbose = verbose

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(
            self.idx,
            self.model.trainable_variables
        )

        zero_mask = tf.cast(init_params != 0, tf.keras.backend.floatx())

        self._logs = tf_utils.sync_to_numpy_or_python_type({})

        self._callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self.model,
            verbose=verbose,
            epochs=epochs,
            steps=1
        )

        self._stop_train = False
        self._current_epoch = tf.Variable(0)

        self._callbacks.model.stop_training = False
        self._callbacks.on_train_begin()

        while not self._stop_train:
            # train the model with L-BFGS solver
            results = tfp.optimizer.lbfgs_minimize(
                value_and_gradients_function=self.train_step,
                initial_position=init_params,
                max_iterations=epochs,
                **args
            )
            if results.failed:
                print("FAILED!")
                print("Restart with from a new random position...")
                init_params = 1.0 - 2.0 * tf.random.uniform(
                    shape=zero_mask.shape,
                    dtype=tf.keras.backend.floatx()
                )
            elif results.converged:
                print("Converged!")
                self._stop_train = True
            else:
                print("NOT converged, continuing...")
                init_params = results.position

        self._x_train = None
        self._y_train = None

        # after training, the final optimized parameters are still in
        # results.position so we have to manually put them back to the model
        self.assign_new_model_parameters(results.position)

        self._callbacks.on_train_end(logs=self._logs)
        self._callbacks = []
        self._logs = None


def test_mnist():
    """
    Test LBFG on MNIST data.

    Returns
    -------
    None.

    """
    float_x = 'float64'
    from sklearn.metrics import confusion_matrix, f1_score
    mnist = tf.keras.datasets.mnist

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    x_train, x_test = X_train / 255.0, X_test / 255.0

    y_train = tf.keras.utils.to_categorical(
        Y_train, num_classes=None, dtype=float_x
    )

    y_test = tf.keras.utils.to_categorical(
        Y_test, num_classes=None, dtype=float_x
    )

    # use float64 by default
    tf.keras.backend.set_floatx(float_x)

    # prepare prediction model, loss function,
    # and the function passed to L-BFGS solver
    pred_model = tf.keras.models.Sequential([
      layers.Flatten(input_shape=(28, 28)),
      layers.Dense(32, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(10),
      layers.Softmax()
    ])

    pred_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['mae'],
        #loss=tf.keras.losses.CategoricalCrossentropy()
        loss=tf.keras.losses.MeanAbsoluteError()
    )

    func = LBFGWrapper(pred_model)

    func.fit(
        x_train=x_train,
        y_train=y_train,
        epochs=1000,
        tolerance=1.0e-14,
        f_relative_tolerance=1e-8
    )

    # do some prediction
    pred_outs = pred_model.predict(x_test)
    Y_pred = np.argmax(pred_outs, axis=-1)

    cm = confusion_matrix(Y_test, Y_pred)

    print(f"\nF1 score {f1_score(Y_test, Y_pred, average='micro')}")
    print(cm)


if __name__ == "__main__":
    test_mnist()