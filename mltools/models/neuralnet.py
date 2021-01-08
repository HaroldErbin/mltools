# -*- coding: utf-8 -*-


"""
Class to represent a neural network model.

Models are created in Keras: a building function is passed as argument to the
`NeuralNet` class. The function must return the model, and a dict of all
the submodels involved. The dict should always include the complete model
under the key `model`.
Example: for a GAN, the first argument is the complete GAN model,
the dictionary is made of the generator and discriminator parts.
"""

import os
import tempfile
import time

import numpy as np

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from .model import Model

from mltools.data.structure import DataStructure
import mltools.data.datatools as dt

from mltools.analysis.metrics import MetricLookup
from mltools.analysis.logger import Logger


# TODO: create a more primitive NeuralNet class used only for predictions?
# useful if one wants to use an already trained network (like a submodel)
class NeuralNetPredictor():

    pass


class NeuralNet(Model):

    model_name = "Neural Network"

    # model_fn must be provided
    _model_fn_required = True

    def reset_model(self, model_params):
        """
        Reset model using given parameters.

        This method is called at initialization if model parameters are given. It is useful to
        have a separate methods when the model must be created several times, for example, when
        computing learning curves or performing hyperparameter tuning.
        """

        if model_params is None:
            self.model_params = {}

            self.model = None
            self.submodels = None
        else:
            self.model_params = model_params

        if self.n > 1:
            self.submodels = [self.create_model() for i in range(self.n)]
            self.model = [m["main"] for m in self.submodels]
            model = self.model[0]
        else:
            # dict of models
            self.submodels = self.create_model()
            # main model
            self.model = self.submodels["main"]
            model = self.model

        # TODO: improve the definition of losses, make contact with module metrics.py
        # TODO: metric names in Keras are not abbreviated

        # read loss and metric names from model parameters, otherwise read from Keras model
        self.loss = self.model_params.get('loss', model.loss)
        self.metrics = self.model_params.get('metrics',
                                             list({self.loss} | set(model.metrics_names[1:])))

    def create_model(self):
        return self.model_fn(self.inputs, self.outputs, **self.model_params)

    def get_model(self, model=None):
        if model is None:
            return self.model
        else:
            if self.n_models > 1:
                return [m[model] for m in self.submodels]
            else:
                return self.submodels[model]

    def fit(self, X, y=None, val_data=None, train_params=None, scaling=True):

        # TODO: missing early stopping in summary

        # TODO: define default train parameters
        if train_params is None:
            train_params = {}

        verbose = train_params.get("verbose", 0)

        self.train_params_history.append(train_params)

        # prepare callbacks: early stopping and save best model
        callbacks = []

        # copy params to remove callbacks
        params = train_params.copy()

        early_stopping = params.pop("early_stopping", None)

        if early_stopping is not None:
            if isinstance(early_stopping, (list, tuple)):
                early_stopping = dict(zip(['min_delta', 'patience'],
                                          early_stopping))
                early_stopping['restore_best_weights'] = True

            if not isinstance(early_stopping, dict):
                raise TypeError("`early_stopping` must be a dictionary.")

            es_call = EarlyStopping(**early_stopping)
            callbacks.append(es_call)

            # use hash to prevent name collision when running several codes
            hashtime = hex(hash(time.time()))
            modelfile = os.path.join(tempfile.gettempdir(),
                                     'best_model_%s.h5' % hashtime)

            # name idea: 'model.{epoch:02d}-{val_loss:.2f}.h5'

            mc_call = ModelCheckpoint(modelfile, save_best_only=True)
            callbacks.append(mc_call)

        reduce_lr = params.pop("reduce_lr", None)

        if reduce_lr is not None:
            if isinstance(reduce_lr, (list, tuple)):
                reduce_lr = dict(zip(['factor', 'patience'], reduce_lr))

            if not isinstance(reduce_lr, dict):
                raise TypeError("`early_stopping` must be a dictionary.")

            rl_call = ReduceLROnPlateau(**reduce_lr)
            callbacks.append(rl_call)

        if y is None:
            y = X

        if val_data is None:
            if "val" in X and "val" in y:
                X_val = X['val']
                y_val = y['val']
            else:
                X_val = None
                y_val = None
        elif isinstance(val_data, (tuple, list)):
            X_val, y_val = val_data
        else:
            X_val = y_val

        if "train" in X and "train" in y:
            X = X['train']
            y = y['train']

        begin_preprocess = time.monotonic()

        # add tests in the function?
        X = self.transform_data(X, self.inputs, scaling=scaling)
        y = self.transform_data(y, self.outputs, scaling=scaling)

        if X_val is not None and y_val is not None:
            X_val = self.transform_data(X_val, self.inputs, scaling=scaling)
            y_val = self.transform_data(y_val, self.outputs, scaling=scaling)

            val_data = (X_val, y_val)
        else:
            val_data = None

        preprocess_time = time.monotonic() - begin_preprocess

        # TODO: add method to prefix keys with inputs/outputs/aux, etc.

        # train Keras model which returns history
        if self.n_models > 1:
            train_time = []
            metrics = []

            for i, m in enumerate(self.model):
                if verbose > 0:
                    print(f"\n# Training model {i+1}/{len(self.model)}\n")

                begin_train = time.monotonic()

                history = m.fit(X, y, validation_data=val_data, callbacks=callbacks, **params)

                metrics.append(history.history)
                train_time.append(time.monotonic() - begin_train)

            if early_stopping is not None:
                try:
                    # self.model[i] = keras.models.load_model(modelfile)
                    os.remove(modelfile)
                except OSError:
                    pass

        else:
            begin_train = time.monotonic()

            history = self.model.fit(X, y, validation_data=val_data, callbacks=callbacks, **params)

            train_time = time.monotonic() - begin_train

            metrics = history.history

            if early_stopping is not None:
                try:
                    # self.model = keras.models.load_model(modelfile)
                    os.remove(modelfile)
                except OSError:
                    pass

        history = self._update_history(metrics, train_time, preprocess_time)

        return history

    def predict(self, X, scaling=True, return_all=False):

        # note: predict() resets model.history

        X = self.transform_data(X, self.inputs)

        if self.n_models > 1:
            y = [self.transform_pred(m.predict(X), scaling=scaling) for m in self.model]

            if return_all is True:
                # return all predictions if explicitly requested

                # TODO: put in a single array for each feature
                return y
            else:
                # average predictions
                if self.outputs is not None:
                    return self.outputs.average(y)
                else:
                    # if no data structure is defined, try simple average
                    return Logger.average(y)
        else:
            y = self.transform_pred(self.model.predict(X), scaling=scaling)

            return y

    def transform_data(self, data, features, scaling=False):
        """
        Prepare data to feed to the network.

        This selects the appropriate transformation mode from the data
        structure. If the model is sequential and if there is a single feature,
        it selects the corresponding tensor from the mode `col`. If there
        are multiple features, it selects the mode `flat`. If the model is
        functional, then it uses the mode `col` and return the complete
        `dict.
        """

        # features is a DataStructure

        if features is None or data is None:
            return data

        # TODO: allow different types of network?
        #       (this would require to build inputs for each independently)
        if self.n_models > 1:
            nn_type = set(map(type, self.model))
            model = self.model[0]

            if len(nn_type) > 1:
                raise ValueError("Cannot handle ensemble of neural networks "
                                 "of different types.")
            nn_type = nn_type.pop()
        else:
            nn_type = type(self.model)
            model = self.model

        if nn_type == keras.models.Sequential:
            if len(features) == 1:
                # single feature -> return tensor data (preserving shape)
                return list(features(data, mode="col", scaling=scaling).values())[0]
            else:
                # multiple features -> return a matrix
                return features(data, mode="flat", scaling=scaling)
        else:
            data = features(data, mode="col", scaling=scaling, trivial_dim=True)

            # prefix names if necessary
            if features == self.inputs:
                data = {k: v for k, v
                        in dt.affix_keys(data, suffix="_input").items()
                        if k in model.input_names}
            if features == self.outputs:
                data = {k: v for k, v
                        in dt.affix_keys(data, suffix="_output").items()
                        if k in model.output_names}

            # TODO: take into account multiple auxiliary inputs/outputs
            # TODO: more generic name matching (don't rely on _input, etc.)

            return data

    def transform_pred(self, data, scaling=False):

        if self.n_models > 1:
            nn_type = set(map(type, self.model))
            model = self.model[0]

            if len(nn_type) > 1:
                raise ValueError("Cannot handle ensemble of neural networks "
                                 "of different types.")
            nn_type = nn_type.pop()
        else:
            nn_type = type(self.model)
            model = self.model

        if nn_type != keras.models.Sequential:
            if len(model.output_names) > 1:
                data = dict(zip(model.output_names, data))
            # else:
            #     data = {model.output_names[0]: data}

            data = {k.rstrip('_output'): v for k, v in data.items()}

        if self.outputs is not None:
            return self.outputs.inverse_transform(data, scaling=scaling)
        else:
            return data

    def _update_metric_history(self, metrics=None):
        """
        Update metric history during training.

        When training a bag of neural networks, the number of epochs can be different if using
        early stopping. To fix this issue, each loss array is padded with its last value until they
        all have the same length.
        """

        metrics = metrics.copy()

        nonmetric = {}
        history = {}

        # for a single metric (= self.loss), the history returned from training is a list
        # convert to dict
        if isinstance(metrics, list) and self.n_models == 1:
            metrics = {"loss": np.array(metrics)}

        if self.n_models > 1:

            if isinstance(metrics[0], list):
                # convert history lists to dict
                metrics = [{"loss": m} for m in metrics]

            metrics = dt.exchange_list_dict(metrics)

            # losses can be of different lengths (e.g. due to early stopping)
            # to solve this problem, we pad with the last data of each array
            for metric, hist in metrics.items():
                # compute length of longest history to be used for padding
                epochs = np.reshape([len(h) for h in hist], (1, -1))
                max_length = np.max(epochs)

                # padding function: trivial when all histories have the same length
                pad_data = lambda x: np.pad(x, (0, max_length - len(x)), constant_values=x[-1])

                # TODO: check if transpose is optimal
                history[metric] = np.c_[tuple(pad_data(h) for h in hist)].T
        else:
            epochs = np.array(len(metrics["loss"]))
            history.update({k: np.array(v) for k, v in metrics.items()})

        # update metric names (not needed)
        # history = dict(zip(self.metrics, history.values()))

        nonmetric["epochs"] = epochs

        if "lr" in history:
            nonmetric["lr"] = history.pop("lr")

        if len(self.metric_history) == 0:
            # if no training has happened yet, use directly current history
            self.metric_history.update(history)
        else:
            # if training has already happened, merge history with previous
            # history from previous runs
            for key, val in history.items():
                self.metric_history[key] = np.r_[self.metric_history[key], val]

        return history, nonmetric


def deep_model():
    """
    Generic deep neural network.
    """

    model = None

    return model, {'model': model}

# layers: tuple of int, list or dict
# if int: number of units
# if list: (number of units, kernel size)
# if dict: arguments (include all elements of layers)


#def conv_block(x, layers, d=2, pooling_kernel)
#    # each parameter can be passed as a global parameter and overriden
#    # as a parameter in layers
#
#    layers = [(layer,) if isinstance(layer, int) else layer for layer in layers]
#    layers = [dict(zip(('units', 'pooling'), *layer))
#              if isinstance(layer, (tuple, list)) else layer
#              for layer in layers]
#
#    if d == 1:
#        ConvLayer = layers.Conv1d
#        MaxPoolLayer = layers.MaxPooling1d
#
#    for layer in layers:
#
#
#    return x
