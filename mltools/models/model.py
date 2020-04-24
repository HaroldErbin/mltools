"""
Generic model class.
"""

import json

import numpy as np

from sklearn.pipeline import Pipeline

from ..data.structure import DataStructure


# TODO: a model can be passed a DataStructure or a Pipeline
# TODO: method to create new model with different parameters from existing
#      (keep inputs/outputs)

# TODO: pass parameters n (default n=1):
#   - if validation set, train n models
#   - else, train with cross-validation
#   then, return all results
#   question: compute mean and std here or in predictions?


class Model:
    """
    Metaclass for ML models

    The optional `inputs` and `outputs` arguments during initialisation
    can be `DataStructure` or `Pipeline`. If defined, they will be call
    before each method of the models to the inputs and outputs data to
    convert them. If they are not define, the inputs and outputs feed
    to the methods should already have the correct format. In this way,
    the models are also compatible as a part of a pipeline.

    By default a model must return a single predictions. Hence, if there are
    several models, an average must be returned by the model. This is the
    expected input for the class `Predictions`. Average for identical data
    can be handled in the `DataStructure` class, but more complex average
    must be done through the model.
    """

    def __init__(self, inputs=None, outputs=None, model_params=None, n=1,
                 name=""):
        """
        :param inputs: how input data to the various function should be
            converted before feeding to the model
        :type inputs: `DataStructure` or `Pipeline`
        :param outputs: how output data to the various function should be
            converted before feeding to the model
        :type outputs: `DataStructure` or `Pipeline`
        :param n: number of models to train
        :type outputs: `int`
        """

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

        self.n = n

        # keep all train parameters used
        # not used by all models
        self.train_params_history = []

        # training history
        # not used by all models
        self.history = {}

        # method used (in general classification or regression)
        # not used by all models
        self.method = None

        if model_params is None:
            self.model_params = {}
        else:
            self.model_params = model_params

        self.submodels = None
        # this must be instantiated for each model class because it may
        # depend on additional parameters not defined in the general class
        # (and, conversely, defining these parameters need model_params to
        # have been defined just before)
        self.model = None

        # define base for model name
        # this will be changed in each model
        self.model_name = "Model"
        # instance based name to distinguish the models used
        # this is combined with model_name to give the full name
        self.name = name

    def __str__(self):
        return "{}: {}".format(self.model_name, self.name or hex(id(self)))

    def __repr__(self):
        if self.model_params == {}:
            return '<{}>'.format(str(self))
        else:
            return '<{}: {}>'.format(str(self), self.model_params)

    @property
    def n_models(self):
        """
        Number of internal models.

        For simple bagging it is equal to `n`, but it may be different when
        using stacking.
        """

        if isinstance(self.model, (list, tuple)):
            return len(self.model)
        else:
            return 1

    @property
    def get_train_params(self):
        if len(self.train_params_history) == 1:
            return self.train_params_history[0]
        else:
            return {"Run %d" % (i+1): v for i, v
                    in enumerate(self.train_params_history)}

    def save_params(self, filename="", logtime=True, logger=None):

        model_params = {"model_name": self.model_name, "name": self.name,
                        "n": self.n, "method": self.method}

        model_params.update(self.model_params)

        if logger is not None:
            logger.save_json(model_params, filename, logtime)
        else:
            with open(filename, 'w') as f:
                json.dump(model_params, f, indent=4)

    def fit(self, X, y=None, cv=False):

        if y is None:
            y = X

        if self.inputs is not None:
            X = self.inputs(X, mode='flat')
        if self.outputs is not None:
            y = self.outputs(y, mode='flat')

        # TODO: if verbose is in train_params, write which model is trained

        if self.n_models > 1:
            return [m.fit(X, y) for m in self.model]
        else:
            return self.model.fit(X, y)

    def predict(self, X, return_all=False):

        if self.inputs is not None:
            X = self.inputs(X, mode='flat')

        # test if the model define an ensemble
        if self.n_models > 1:
            y = [m.predict(X) for m in self.model]

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
                    # if no data structure is defined, try brutal average
                    return np.mean(y, axis=0), np.std(y, axis=0)
        else:
            y = self.model.predict(X)

            if self.outputs is not None:
                y = self.outputs.inverse_transform(y)

            return y

    def create_model(self):
        # useful for creating several models (for bagging, cross-validation...)

        raise NotImplementedError("Trying to call abstract `Model` class. ")

    def save_model(self, file):
        # save weights
        # save parameters (name...)

        raise NotImplementedError

    def load_model(self, file):

        raise NotImplementedError
