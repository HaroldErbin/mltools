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

from tensorflow import keras

from .model import Model


# TODO: create a more primitive NeuralNet class used only for predictions?
# useful if one wants to use an already trained network (like a submodel)
class NeuralNetPredictor():

    pass


class NeuralNet(Model):

    def __init__(self, model_fn, inputs=None, outputs=None, model_params=None,
                 n=1, name=""):

        Model.__init__(self, inputs, outputs, model_params, n, name)

        self.model_fn = model_fn

        # dict of models
        self.submodels = self.create_model()

        # model instance
        self.model = self.submodels["model"]

        self.model_name = "Neural Network"

    def create_model(self):
        return self.model_fn(self.inputs, self.outputs, **self.model_params)

    def get_model(self, model=None):
        if model is None:
            return self.model
        else:
            return self.submodels[model]

    def fit(self, X, y=None, train_params=None, fit_fn=None):
        # fit_fn: fine-tuned fit function (useful for GAN)

        # TODO: check model type, if sequential, use flat mode,
        #   if functional, use col

        # TODO: define default train parameters
        if train_params is None:
            train_params = {}

        self.train_params_history.append(train_params)

        if y is None:
            y = X

        # add tests in the function?
        X = self.transform_data(X, self.inputs)
        y = self.transform_data(y, self.outputs)

        # TODO: add method to prefix keys with inputs/outputs/aux, etc.

        # TODO: create attribute history, which is mean value over the
        #   different models
        # self.history = {}

        return self.model.fit(X, y, **train_params)

    def predict(self, X):

        X = self.transform_data(X, self.inputs)

        y = self.model.predict(X)

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
        # features = DataStructure

        if features is None:
            return data

        if isinstance(self.model, keras.models.Sequential):
            if len(features) == 1:
                # single feature -> return tensor data (preserving shape)
                return list(features(data, mode="col").values())[0]
            else:
                # multiple features -> return a matrix
                return features(data, mode="flat")
        else:
            return features(data, mode="col")


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
