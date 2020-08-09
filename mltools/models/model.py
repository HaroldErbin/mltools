"""
Generic model class.
"""

import json

import numpy as np

from sklearn.pipeline import Pipeline

from mltools.data import datatools
from mltools.data.structure import DataStructure


# TODO: method `evaluate`, `score` with metric (or list of metrics) as argument

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
                 method=None, name=""):
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

        # TODO: weird bug where DataStructure is no recognized
        # if (inputs is not None
        #         and not isinstance(inputs, (Pipeline, DataStructure))):

        #     error = ("`inputs` can only be None, a Pipeline "
        #              "or a DataStructure, not `{}`.")
        #     raise TypeError(error.format(type(inputs)))

        # if (outputs is not None
        #         and not isinstance(outputs, (Pipeline, DataStructure))):

        #     error = ("`outputs` can only be None, a Pipeline "
        #              "or a DataStructure, not `{}`.")
        #     raise TypeError(error.format(type(outputs)))

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
        self.method = self._get_method(method)

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

    @staticmethod
    def _get_method(method):
        """
        Normalize the method name.

        Choices are:
        - classification (clf)
        - regression (reg)

        """

        if method in ("clf", "classification"):
            return "classification"
        elif method in ("reg", "regression"):
            return "regression"
        elif method is None:
            return None
        else:
            raise ValueError("Method `%s` not permitted." % method)

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

    def fit(self, X, y=None, cv=False, train_params=None, filtering=True):

        # TODO: define method to fit one model, then apply for each model
        # if n > 1: this would allow to have models with different structures

        # filtering → apply filter (for outliers), by default yes

        # TODO: count model number trained

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

    def predict(self, X, return_all=False, filtering=False):

        # filtering → apply filter (for outliers), by default no

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
                    # if no data structure is defined, try simple average
                    return datatools.average(y)
        else:
            y = self.model.predict(X)

            if self.outputs is not None:
                y = self.outputs.inverse_transform(y)

            return y

    def update_history(self, history):
        """
        Update training history.
        """

        # TODO: if early stopping, truncate to value for best model?

        if self.n_models > 1:
            # history can be of different lengths (for example due to early
            # stopping)
            # to solve this problem, we pad with the last data of each array

            if isinstance(history[0], dict):
                lengths = [len(list(h.values())[0]) for h in history]
            else:
                lengths = [len(h) for h in history]

            if len(lengths) > 1:
                max_length = max(lengths)

                if isinstance(history[0], dict):
                    history = [{k: np.pad(v, (0, max_length - len(v)),
                                          constant_values=v[-1])
                                for k, v in h.items()}
                               for h in history]
                else:
                    history = [np.pad(h, (0, max_length - len(h)),
                                      constant_values=h[-1])
                               for h in history]

            hist, hist_std = datatools.average([h for h in history])
        else:
            hist = history
            hist_std = {}

        for metric, values in hist.items():
            self.history[metric] = np.r_[self.history.get(metric, ()), values]

        for metric, values in hist_std.items():
            mstd = metric + "_std"
            self.history[mstd] = np.r_[self.history.get(mstd, ()), values]

        return {}

    def model_representation(self):
        """
        Return a representation of the model.

        This shows how the model makes the computations.
        """

        raise NotImplementedError

    def create_model(self):
        # useful for creating several models (for bagging, cross-validation...)

        raise NotImplementedError("Trying to call abstract `Model` class. ")

    def save_model(self, file):
        # save weights
        # save parameters (name...)

        raise NotImplementedError

    def load_model(self, file):

        raise NotImplementedError

    def summary(self, save_params=True, filename=None, logtime=True,
                logger=None):

        # TODO: make model summary (here instead of Predictions)

        raise NotImplementedError
