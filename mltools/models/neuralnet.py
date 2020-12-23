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
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)

from .model import Model

from mltools.data.structure import DataStructure
from mltools.data.datatools import affix_keys
from mltools.analysis.logger import Logger


# TODO: create a more primitive NeuralNet class used only for predictions?
# useful if one wants to use an already trained network (like a submodel)
class NeuralNetPredictor():

    pass


class NeuralNet(Model):

    def __init__(self, model_fn, inputs=None, outputs=None, model_params=None,
                 n=1, method=None, name=""):

        Model.__init__(self, inputs, outputs, model_params, n, method, name)

        self.model_fn = model_fn

        if n > 1:
            self.submodels = [self.create_model() for n in range(self.n)]
            self.model = [m["model"] for m in self.submodels]
            model = self.model[0]
        else:
            # dict of models
            self.submodels = self.create_model()
            # main model
            self.model = self.submodels["model"]
            model = self.model

        self.model_name = "Neural Network"

        # read loss and metric names from model parameters, otherwise
        # read from Keras model

        # TODO: metric names in Keras are not abbreviated
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

    def fit(self, X, y=None, val_data=None, train_params=None):

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
        X = self.transform_data(X, self.inputs)
        y = self.transform_data(y, self.outputs)

        if X_val is not None and y_val is not None:
            X_val = self.transform_data(X_val, self.inputs)
            y_val = self.transform_data(y_val, self.outputs)

            val_data = (X_val, y_val)
        else:
            val_data = None

        preprocess_time = time.monotonic() - begin_preprocess

        # TODO: add method to prefix keys with inputs/outputs/aux, etc.

        # train Keras model which returns history
        if self.n_models > 1:
            train_time = []
            losses = []

            for i, m in enumerate(self.model):
                if verbose > 0:
                    print(f"\n# Training model {i+1}/{len(self.model)}\n")

                begin_train = time.monotonic()

                history = m.fit(X, y, validation_data=val_data, callbacks=callbacks, **params)

                losses.append(history.history)
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

            losses = history.history

            if early_stopping is not None:
                try:
                    # self.model = keras.models.load_model(modelfile)
                    os.remove(modelfile)
                except OSError:
                    pass

        history = self.update_train_history(losses, train_time, preprocess_time)

        return history

    def predict(self, X, return_all=False):

        # note: predict() resets model.history

        X = self.transform_data(X, self.inputs)

        if self.n_models > 1:
            y = [self.transform_pred(m.predict(X)) for m in self.model]

            if self.outputs is not None:
                y = [self.outputs.inverse_transform(v) for v in y]

            if return_all is True:
                # return all predictions if explicitly requested
                return y
            else:
                # average predictions
                if self.outputs is not None:
                    return self.outputs.average(y)
                else:
                    # if no data structure is defined, try simple average
                    return Logger.average(y)
        else:
            y = self.transform_pred(self.model.predict(X))

            if self.outputs is not None:
                y = self.outputs.inverse_transform(y)

            return y

    def transform_data(self, data, features):
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
                return list(features(data, mode="col").values())[0]
            else:
                # multiple features -> return a matrix
                return features(data, mode="flat")
        else:
            data = features(data, mode="col", trivial_dim=True)

            # prefix names if necessary
            if features == self.inputs:
                data = {k: v for k, v
                        in affix_keys(data, suffix="_input").items()
                        if k in model.input_names}
            if features == self.outputs:
                data = {k: v for k, v
                        in affix_keys(data, suffix="_output").items()
                        if k in model.output_names}

            # TODO: take into account multiple auxiliary inputs/outputs
            # TODO: more generic name matching (don't rely on _input, etc.)

            return data

    def transform_pred(self, data):

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
            return data
        else:
            if len(model.output_names) > 1:
                data = dict(zip(model.output_names, data))
            # else:
            #     data = {model.output_names[0]: data}

            return {k.rstrip('_output'): v for k, v in data.items()}


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
