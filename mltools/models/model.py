"""
Generic model class.
"""

import json

import numpy as np

from sklearn.pipeline import Pipeline

from mltools.data import datatools as dt
from mltools.data.structure import DataStructure
from mltools.analysis.logger import Logger


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

        # keep all train parameters used (updated only by some models)
        self.train_params_history = []

        # training history (updated only by some models)
        self.history = {}

        # method used (in general classification or regression)
        # defined only for some models
        self.method = self._get_method(method)

        if model_params is None:
            self.model_params = {}
        else:
            self.model_params = model_params

        # define loss and metrics name by looking in model parameters
        # if no loss is given, use 'loss' for the loss function
        # if no metric is given, use the name of the loss
        self.loss = self.model_params.get('loss', 'loss')
        self.metrics = self.model_params.get('metrics', [self.loss])

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

    def fit(self, X, y=None, cv=False, train_params=None, filtering=True):
        """
        Train a model.

        In Scikit, `fit()` erases any previous training. To train by batch,
        it is necessary to use `partial_fit()` instead, but this method is
        not defined for many models.
        """

        # TODO: define method to fit one model, then apply for each model
        # if n > 1: this would allow to have models with different structures

        # filtering → apply filter (for outliers), by default yes
        if train_params is None:
            train_params = {}

        verbose = train_params.get("verbose", 0)

        if y is None:
            y = X

        if self.inputs is not None:
            X = self.inputs(X, mode='flat')
        if self.outputs is not None:
            y = self.outputs(y, mode='flat')

        # TODO: update history

        if self.n_models > 1:
            results = []
            for i, m in enumerate(self.model):
                if verbose > 0:
                    print(f"\n# Training model {i+1}/{len(self.model)}\n")

                results.append(m.fit(X, y))
            return results
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
                    return dt.average(y)
        else:
            y = self.model.predict(X)

            if self.outputs is not None:
                y = self.outputs.inverse_transform(y)

            return y

    def update_train_history(self, history):
        """
        Update training history.

        History is updated after each training. Values are simply concatenated
        to each other, without distinguishing between the different trainings.
        The number of epochs is also stored.
        """

        # if there is a single metric (= loss), history returned from
        #  training is a list; convert to dict
        if len(self.metrics) == 1:
            if self.n_models > 1:
                history = [{self.loss: h} for h in history]
            else:
                history = {self.loss: history}

        if self.n_models > 1:
            history = dt.exchange_list_dict(history)

            # history can be of different lengths (e.g. due to early stopping)
            # to solve this problem, we pad with the last data of each array

            for metric, hist in history.items():
                # compute length of longest history to be used for padding
                epochs = np.array([len(h) for h in hist]).reshape(1, -1)
                max_length = np.max(epochs)

                # padding function: note that it is trivial when all histories
                # have the same length
                pad_data = lambda x: np.pad(x, (0, max_length - len(x)),
                                            constant_values=x[-1])

                history[metric] = np.c_[tuple(pad_data(h) for h in hist)]
        else:
            epochs = len(history)

        # update metric names
        history = dict(zip(self.metrics, history.values()))
        history['epochs'] = epochs

        if self.history == {}:
            # if no training has happened yet, use directly current history
            self.history.update(history)
        else:
            # if training has already happened, merge history with previous
            # history from previous runs
            for metric, values in history.items():
                self.history[metric] = np.r_[self.history[metric], values]

    def model_representation(self):
        """
        Return a representation of the model.

        This shows how the model makes the computations.
        """

        raise NotImplementedError

    def create_model(self):
        # useful for creating several models (for bagging, cross-validation...)

        raise NotImplementedError("Trying to call abstract `Model` class.")

    def save_model(self, file):
        # save weights
        # save parameters

        raise NotImplementedError

    def load_model(self, file):

        raise NotImplementedError

    def get_model_params(self, filename="", logtime=False, logger=None):

        model_params = {"model_name": self.model_name,
                        "n": self.n, "method": self.method}

        if self.name != "":
            model_params["name"] = self.name

        model_params.update(self.model_params)

        if filename != "":
            if logger is None:
                logger = Logger(logtime="filename")

            logger.save_json(model_params, filename=filename, logtime=logtime)

        return model_params

    def get_train_params(self, filename="", logtime=False, logger=None):
        """
        Return dict of train parameters.

        If training has been made by steps, organize `dict` accordingly.

        If filename is given, save in json file.
        """

        if len(self.train_params_history) == 1:
            train_params = self.train_params_history[0]
        else:
            train_params = {"Run %d" % (i+1): v for i, v
                            in enumerate(self.train_params_history)}

        if filename != "":
            if logger is None:
                logger = Logger(logtime="filename")

            logger.save_json(train_params, filename=filename, logtime=logtime)

        return train_params

    def get_train_history(self, average=True, filename="", logtime=False,
                          logger=None):

        # TODO: if early stopping, truncate to value for best model?

        hist_std = {}

        if self.n_models > 1 and average is True:
            hist, hist_std = dt.average(self.history, axis=1)
            hist.update((k + "_std", v) for k, v in hist_std.items())
        else:
            hist = self.history.copy()

        if filename != "" and len(hist) > 0:
            if logger is None:
                logger = Logger(logtime="filename")

            hist_json = {k: v.tolist() for k, v in hist.items()}
            logger.save_json(hist_json, filename=filename, logtime=logtime)

        return hist

    def training_curve(self):

        pass

    def summary(self, save_model_params=True, save_train_params=True,
                save_history=True, filename="", logtime=False, logger=None,
                show=False):

        # TODO: save weights

        if logger is None:
            logger = Logger(logtime="filename")

        text = ""
        params = {}

        model_params = self.get_model_params(filename="")

        if save_model_params is True:
            params["model_params"] = model_params

        train_params = self.get_train_params()

        if save_model_params is True:
            params["train_params"] = train_params

        if len(params) > 0 and filename != "":
            logger.save_json(params, filename=filename + "_params.json",
                             logtime=logtime)

        if save_history is True and filename != "":
            fname = filename + "_history.json"
        else:
            fname = ""
        history = self.get_train_history(average=False, filename=fname,
                                         logtime=logtime, logger=logger)

        text += f"## Model - {self}\n\n"

        if len(model_params) == 0:
            text += "No model parameters"
        else:
            text += "Model parameters:\n"
            text += logger.dict_to_text(model_params)

        if len(train_params) > 0:
            text += "\n\nTrain parameters:\n"
            text += logger.dict_to_text(train_params)

        if show is True:
            print(text)

        return text
