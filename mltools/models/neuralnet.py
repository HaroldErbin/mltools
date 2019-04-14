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


from .model import Model


# TODO: create a more primitive NeuralNet class used only for predictions?
# useful if one wants to use an already trained network (like a submodel)
class NeuralNetPredictor():

    pass


class NeuralNet(Model):

    def __init__(self, model_fn, model_params=None, train_params=None,
                 model=None, name=''):
        self.model_fn = model_fn
        self.model_params = model_params
        self.train_params = train_params

        # if model is not None: instantiate with it (recover its weights)

        # model instance
        self.model = None
        # dict of models
        self.submodels = None

        self.name = name

    def __str__(self):
        return self.name or ('NeuralNet %s' % hex(id(self)))

    def __repr__(self):
        return '<%s>' % str(self)

    def create_model(self):
        pass

    def get_model(self, model=None, inputs=None, outputs=None):
        # TODO: if inputs and outputs are not None and a datastructure,
        # create a NeuralNet instance (for predictions)

        if model is None:
            return self.model
        else:
            return self.submodels[model]

    def train(self, train_fn=None):
        # fine-tuned train function (useful for GAN)
        if train_fn is not None:
            return train_fn()


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
