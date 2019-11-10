"""
Generic model class.
"""


from sklearn.pipeline import Pipeline

from ..data.structure import DataStructure


# TODO: a model can be passed a DataStructure or a Pipeline
# TODO: method to create new model with different parameters from existing
#      (keep inputs/outputs)


class Model:
    """
    Metaclass for ML models

    The optional `inputs` and `outputs` arguments during initialisation
    can be `DataStructure` or `Pipeline`. If defined, they will be call
    before each method of the models to the inputs and outputs data to
    convert them. If they are not define, the inputs and outputs feed
    to the methods should already have the correct format. In this way,
    the models are also compatible as a part of a pipeline.
    """

    def __init__(self, inputs=None, outputs=None, model_params=None, name=""):
        """
        :param inputs: how input data to the various function should be
            converted before feeding to the model
        :type inputs: `DataStructure` or `Pipeline`
        :param outputs: how output data to the various function should be
            converted before feeding to the model
        :type outputs: `DataStructure` or `Pipeline`
        """

        self.model = None

        if model_params is None:
            self.model_params = {}
        else:
            self.model_params = model_params

        self.name = name or 'Model {}'.format(hex(id(self)))

        if (inputs is not None
                and not isinstance(inputs, (Pipeline, DataStructure))):

            error = ("`inputs` can only be None, a Pipeline "
                     "or a DataStructure, not `{}`.")
            raise TypeError(error.format(type(inputs)))

        if (outputs is not None
                and not isinstance(outputs, (Pipeline, DataStructure))):

            error = ("`outputs` can only be None, a Pipeline "
                     "or a DataStructure, not `{}`.")
            raise TypeError(error.format(type(outputs)))

        self.inputs = inputs
        self.outputs = outputs

    def __str__(self):
        return self.name

    def __repr__(self):
        if self.model_params == {}:
            return '<{}>'.format(str(self))
        else:
            return '<{}: {}>'.format(str(self), self.model_params)

    def fit(self, X, y=None):

        if y is None:
            y = X

        if self.inputs is not None:
            X = self.inputs(X, mode='flat')
        if self.outputs is not None:
            y = self.outputs(y, mode='flat')

        return self.model.fit(X, y)

    def predict(self, X):

        if self.inputs is not None:
            X = self.inputs(X, mode='flat')

        y = self.model.predict(X)

        if self.outputs is not None:
            y = self.outputs.inverse_transform(y)

        return y

#    def create_model(self, **model_params):
#        self.model(**model_params)

    def save_model(self, file):
        # save weights
        # save parameters (name...)

        raise NotImplementedError

    def load_model(self, file):

        raise NotImplementedError
