"""
Generic model class.
"""

import time
import json

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid, ParameterSampler

from mltools.data import datatools as dt
from mltools.data.sample import RatioSample
from mltools.data.structure import DataStructure

from mltools.analysis import describe as descr
from mltools.analysis.metrics import MetricLookup
from mltools.analysis.logger import Logger
from mltools.analysis.predictions import Predictions


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

    _per_run_keys = ["epochs", "train_time", "preprocess_time"]

    submodels = None
    # this must be instantiated for each model class because it may
    # depend on additional parameters not defined in the general class
    # (and, conversely, defining these parameters need model_params to
    # have been defined just before)
    model = None

    loss = None
    metrics = None

    # define base for model name
    # this must be changed in each model subclass (as attribute or property)
    model_name = "Model"

    # if true, argument `model_fn` must be given when creating an instance
    _model_fn_required = False

    def __init__(self, inputs=None, outputs=None, model_params=None, model_fn=None, n=1,
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

        if self._model_fn_required is True and model_fn is None:
            raise ValueError("`model_fn` argument must be provided.")

        # keep all train parameters used (updated only by some models)
        self.train_params_history = []

        # training history: training times, etc.
        self.history = {}
        # training history: metrics for each epoch (neural network)
        self.metric_history = {}

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

        # method used (in general classification or regression)
        # defined only for some models
        self.method = self._get_method(method)

        self.model_fn = model_fn

        # instance based name to distinguish the models used
        # this is combined with model_name to give the full name
        self.name = name

        # create model if model_param is given
        if model_params is not None:
            self.reset_model(model_params)
        else:
            self.model_params = {}

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

    def fit(self, X, y=None, cv=False, train_params=None, scaling=True):
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

        if "train" in X and "train" in y:
            X = X['train']
            y = y['train']

        begin_preprocess = time.monotonic()

        if self.inputs is not None:
            X = self.inputs(X, mode='flat', scaling=scaling)
        if self.outputs is not None:
            y = self.outputs(y, mode='flat', scaling=scaling)

        preprocess_time = time.monotonic() - begin_preprocess

        # TODO: update history

        if self.n_models > 1:
            # TODO: if using scikit to train on multi-core, not sure that it will be possible
            #       to time each model individually (if not, do a mean)

            train_time = []
            results = []

            for i, m in enumerate(self.model):
                if verbose > 0:
                    print(f"\n# Training model {i+1}/{len(self.model)}\n")

                begin_train = time.monotonic()

                results.append(m.fit(X, y))

                train_time.append(time.monotonic() - begin_train)
        else:
            begin_train = time.monotonic()

            results = self.model.fit(X, y)

            train_time = time.monotonic() - begin_train

        history = self._update_history(train_time=train_time, preprocess_time=preprocess_time)

        return history

    def predict(self, X, scaling=True, return_all=False):
        """
        Make predictions using model.

        `X` is the input. It can be a dict consisting of several split (such as train and test),
        in which case predictions are done for each split separately.
        """

        # filtering → apply filter (for outliers), by default no

        # TODO: better checking, in particular, using self.inputs
        if "train" in X:
            return {k: self.predict(X[k], return_all) for k in X}

        if self.inputs is not None:
            X = self.inputs(X, mode='flat')

        # test if the model defines an ensemble
        if self.n_models > 1:
            y = [m.predict(X) for m in self.model]

            if self.outputs is not None:
                y = [self.outputs.inverse_transform(v, scaling=scaling) for v in y]

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
                y = self.outputs.inverse_transform(y, scaling=scaling)

            return y

    def fit_predict(self, X, y=None, cv=False, train_params=None, scaling=True, return_all=False):

        history = self.fit(X, y, cv, train_params, scaling)

        return self.predict(X, scaling, return_all)

    def evaluate(self, X, y=None, metric=None, scaling=True, return_all=True):

        # TODO: loop over train, test...

        if y is None:
            y = X

        if self.outputs is not None:
            y = self.outputs(y, mode='col')

        pred = self.predict(X, scaling=scaling, return_all=True)

        if self.n_models > 1:
            # TODO: this assumes that y is a dict, which may not be true
            pred = {f: np.array([p[f] for p in pred]) for f in self.outputs}

        # TODO: statistics if return_all is False
        if return_all is False:
            raise NotImplementedError

        return MetricLookup.evaluate(y, pred, metric=None, n=self.n_models)

    def _update_metric_history(self, metrics=None):
        """
        Update metric history during training.

        Sometimes, the `metrics` argument may contain information unrelated to the metrics (for
        example, for a Keras neural network) or need additional processing. This can be done here.
        It must returns two `dict`: one containing the metrics, another containing information
        unrelated to the metrics.
        """

        nonmetric = {}

        return metrics.copy(), nonmetric

    def _update_history(self, metrics=None, train_time=None, preprocess_time=None):
        """
        Update training history.

        History is updated after each training. Values are simply concatenated to each other,
        without distinguishing between the different trainings.

        Information which can be stored:
        - number of epochs (neural network)
        - preprocessing time (all): time needed to convert the data before training the algorithm
        - training time (all): time needed to train the algorithm
        - metrics (neural network): metrics evaluated at each epoch for both the train and
            validation sets
        - learning rate (neural network): learning rate at each epoch
        """

        history = {}

        if train_time is not None:
            history['train_time'] = np.array(train_time)
            if self.n_models > 1:
                history['train_time'] = history['train_time'].reshape(1, -1)

        if preprocess_time is not None:
            history['preprocess_time'] = np.array(preprocess_time)
            if self.n_models > 1:
                # needed to ensure consistent computation of average
                history['preprocess_time'] = history['preprocess_time'].reshape(1, -1)

        if metrics is not None:
            # process `metrics` parameter in dedicated method: when training from Keras,
            # `metrics` is obtained from `history.history` which contains non-metric information
            #  (e.g. learning rate)
            # moreover, we want to save the number of epochs of each training period
            metric_history, nonmetric_history = self._update_metric_history(metrics=metrics)
            history.update(nonmetric_history)
        else:
            metric_history = None

        if len(self.history) == 0:
            # if no training has happened yet, use directly current history
            self.history.update(history)
        else:
            # if training has already happened, merge history with previous
            # history from previous runs
            for key, val in history.items():
                self.history[key] = np.r_[self.history[key], val]

        if metric_history is not None:
            history["metrics"] = metric_history

        return history

    def model_representation(self):
        """
        Return a representation of the model.

        This shows how the model makes the computations.
        """

        raise NotImplementedError

    def reset_model(self, model_params=None):
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

        # define loss and metrics name by looking in model parameters
        # if no loss is given, use 'loss' for the loss function
        self.loss = self.model_params.get('loss', 'loss')
        # if no metric is given, use the name of the loss
        self.metrics = self.model_params.get('metrics', [self.loss])

        if self.n > 1:
            self.model = [self.create_model() for i in range(self.n)]
        else:
            self.model = self.create_model()

        self.submodels = None

    def create_model(self):
        """
        Create a model with given parameters.

        This method must be implemented independently for each model subclass. It must selects the
        appropriate class (and package) depending on the arguments provided.

        Since the argument contains parameters for both the class and for the algorithm, the
        method in the subclass must take care of cleaning the model parameters before
        sending them to the algorithm.

        This method returns a model instance from the chosen algorithm.

        This method must also update the `model_params` attribute with default parameters if they
        are missing.
        """

        raise NotImplementedError("Trying to call abstract `Model` class.")

    def copy_model(self):
        return self.__class__(self.inputs, self.outputs, self.model_params, self.model_fn,
                              n=self.n_models, method=self.method, name=self.name)

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

    def get_history(self, average=True, filename="", logtime=False, logger=None):
        """
        Return train history.

        This includes also the metric history (per epoch).
        """

        hist_std = {}

        if self.n_models > 1 and average is True:
            hist, hist_std = dt.average(self.history, axis=1)
            hist.update((k + "_std", v) for k, v in hist_std.items())

            if len(self.metric_history) > 0:
                metric_hist, metric_hist_std = dt.average(self.metric_history, axis=1)
                hist["metrics"] = metric_hist
                hist["metrics"].update((k + "_std", v) for k, v in metric_hist_std.items())
        else:
            hist = self.history.copy()

            if len(self.metric_history) > 0:
                hist["metrics"] = self.metric_history

        if filename != "" and len(hist) > 0:
            if logger is None:
                logger = Logger(logtime="filename")

            hist_json = {k: v.tolist() for k, v in hist.items() if k != "metrics"}
            if "metrics" in hist:
                hist_json["metrics"] = {k: v.tolist() for k, v in hist["metrics"].items()}

            logger.save_json(hist_json, filename=filename, logtime=logtime)

        return hist

    def get_training_time(self, include_preprocess=True, mode=None):
        """
        Return the total training time.

        For a bag of models, return all the mean training time and standard deviation.
        """

        times = {}

        if self.n_models > 1:
            train_time, train_time_std = dt.average(self.history["train_time"], axis=1)
            times["train_time"] = np.sum(train_time)
            times["train_time_std"] = np.sum(train_time_std)

            times["total_train_time"] = np.sum(self.history["train_time"])

            if include_preprocess is True:
                preprocess_time, preprocess_time_std = dt.average(self.history["preprocess_time"],
                                                                  axis=1)
                times["preprocess_time"] = np.sum(preprocess_time)
                times["preprocess_time_std"] = np.sum(preprocess_time_std)

                times["total_train_time"] += np.sum(self.history["preprocess_time"])

        else:
            times["train_time"] = np.sum(self.history["train_time"])

            times["total_train_time"] = times["train_time"]

            if include_preprocess is True:
                times["preprocess_time"] = self.history["preprocess_time"]

                times["total_train_time"] += times["preprocess_time"]

        if mode == "text":
            times = {k: Logger.format_time(v) for k, v in times.items()}

            # get length of longest time string to align
            val_maxlen = max(map(len, (times["total_train_time"], times["train_time"],
                                       times.get("preprocessing_time", ""))))
            val_fmt = "- {:<14} {:>%d}\n" % val_maxlen

            if self.n_models > 1:
                std_maxlen = max(map(len, (times["train_time_std"],
                                           times.get("preprocessing_time_std", ""))))
                valstd_fmt = "- {:<14} {:>%d} ± {:>%d}\n" % (val_maxlen, std_maxlen)

                text = val_fmt.format("total", times["total_train_time"])

                text += valstd_fmt.format("training", times["train_time"], times["train_time_std"])

                if include_preprocess is True:
                    text += valstd_fmt.format("preprocessing", times["preprocess_time"],
                                              times["preprocess_time_std"])
            else:
                if include_preprocess is True:
                    text = val_fmt.format("total", times["total_train_time"])
                    text += val_fmt.format("training", times["train_time"])
                    text += val_fmt.format("preprocessing", times["preprocess_time"])
                else:
                    text = times["total_train_time"]

            return text
        else:
            return times

    def training_curve(self, metric=None, sigma=1, log=True, marker=None, filename="",
                       logtime=False, logger=None):
        """
        Evolution of a metric during training.

        `metric` can be a single metric or a list of metrics. It can also be used to plot other
        quantities which change during training (for example, learning rate).
        """

        # TODO: plot position of best model if early stopping
        #       it is located at total_step - wait_step

        if metric == "lr":
            history = self.history.get("lr")
        else:
            history = self.metric_history.copy()
            history["lr"] = self.history.get("lr")

        # if self.n_models > 1:
        #     history, history_std = dt.average(history, axis=0)

        return descr.training_curve(history, None, metric, sigma, log, marker,
                                    filename, logtime, logger)

    def learning_curve(self, X, y=None, train_params=None, ratios=None, val_ratio=0., metric=None,
                       scaling=True, filename="", logtime=False, logger=None):
        """
        Compute learning curve.

        The ratios are given for the training set. `val_ratio` can be set to specify which size
        of the latter is kept for validation. However, evaluation is performed for train and
        validation together.
        """

        logger = logger or Logger()

        if train_params is None:
            train_params = {}

        verbose = train_params.get("verbose", 0)

        if ratios is None:
            ratios = np.arange(0.1, 1.0, 0.1)

        # store results at each ratio
        scores = {f: {"train": {}, "test": {}} for f in self.outputs}
        scores["ratios"] = ratios

        if "train" in X and "val" in X:
            # if X is already split and contains both train and validation data, combine them
            # TODO: merge sets
            raise NotImplementedError("Cannot deal with training data split in train and "
                                      "validation sets.")
        elif "train" in X:
            X = X['train']

        if y is None:
            y = X

        for x in ratios:
            x = np.round(x, 10)

            if verbose > 0:
                print(f"\n# Training ratio: {x}\n")

            sampling = RatioSample({"trainval": x, "test": 1 - x})

            X_eval, y_eval = sampling([X, y], shuffle=True)

            if val_ratio > 0:
                trainval_sampling = RatioSample({"train": 1 - val_ratio, "val": val_ratio})
                X_train, y_train = trainval_sampling([X_eval["trainval"], y_eval["trainval"]])
            else:
                X_train, y_train = X_eval["trainval"], y_eval["trainval"]

            model = self.copy_model()

            model.fit(X_train, y_train, train_params=train_params, scaling=scaling)

            scores_train = self.evaluate(X_eval["trainval"], y_eval["trainval"], metric=metric,
                                         scaling=scaling)
            scores_test = self.evaluate(X_eval["test"], y_eval["test"], metric=metric,
                                        scaling=scaling)

            for f in self.outputs:
                scores[f]["train"] = dt.update_dict_array(scores[f]["train"], scores_train[f])
                scores[f]["test"] = dt.update_dict_array(scores[f]["test"], scores_test[f])

        if self.n_models > 1:
            for f in self.outputs:
                scores[f]["train"].update(dt.average(scores[f]["train"], axis=1, mergedict=True))
                scores[f]["test"].update(dt.average(scores[f]["test"], axis=1, mergedict=True))

        if filename != "":
            logger.save_json(scores, filename=filename + '.json', logtime=logtime)
            pdfname = filename + '.pdf'
        else:
            pdfname = ""

        fig = descr.learning_curve_plot(scores, metric, pdfname, logtime, logger)

        return scores, fig

    def hyperparameter_tuning(self, X, y=None, method='grid', model_params=None, train_params=None,
                              val_ratio=0., metric=None, n_iter=20, scaling=True, filename="",
                              logtime=False, logger=None):
        """
        Perform hyperparameter tuning.

        This evaluates sequentially a model with different model and train parameters. If possible,
        it is recommended to use Scikit (or derived) API to parallelize the search. However, this
        allows to optimize models with several outputs (which cannot be written as an array, for
        example, a non-sequential neural network).

        TODO: implement CV search with scikit and tune
        define automatically a scorer function which can uses DataStructure and Metric
        https://docs.ray.io/en/master/tune/api_docs/sklearn.html
        https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring
        """

        # TODO: check what is done in Scikit to do the same
        # TODO: define single metric to use for finding best model
        #       (if multi-outputs or several metrics)

        # parameter dict are generated with Scikit, which can work only with a single set of
        # parameters
        model_keys = model_params.keys()

        if train_params is None:
            train_params = {}
            train_keys = {}
        else:
            train_keys = train_params.keys()

        verbose = train_params.get("verbose", 0)

        logger = logger or Logger()

        if y is None:
            y = X

        if "train" in X and "val" in X:
            X = {"train": X["train"], "val": X["val"]}
            y = {"train": y["train"], "val": y["val"]}
        else:
            if "train" in X:
                X = X['train']
                X = X['train']

            if val_ratio == 0:
                raise ValueError("If validation data is not provided, `val_ratio` cannot be 0.")

            sampling = RatioSample({"train": val_ratio, "val": 1 - val_ratio})
            X, y = sampling([X, y], shuffle=True)

        if not isinstance(train_params, dict):
            raise TypeError(f"`train_params` must be a dict, found `{type(train_params)}`.")

        if isinstance(model_params, (tuple, list)):
            param_set = [{**train_params, **p} for p in model_params]
        elif isinstance(model_params, dict):
            param_set = {**train_params, **model_params}
        else:
            raise TypeError(f"`model_params` must be a dict or list, found `{type(train_params)}`.")

        if method == "grid":
            all_params = ParameterGrid(param_set)
        elif method == "random":
            all_params = ParameterSampler(param_set, n_iter=n_iter)
            # TODO: allow to change rounding
            all_params = [dict((k, round(v, 6)) for (k, v) in d.items()) for d in all_params]
        else:
            raise ValueError(f"Method '{method}' unknown.")

        # store results for each parameter
        scores = []

        for i, params in enumerate(all_params):

            if verbose > 0:
                print(f"\n# Parameter set: {i}/{len(all_params)}\n")

            t_params = {k: params[k] for k in train_keys}
            m_params = {k: params[k] for k in model_keys}

            param_scores = {f: {"train": {}, "val": {}} for f in self.outputs}

            self.reset_model(m_params)
            self.fit(X, y, train_params=t_params, scaling=scaling)

            scores_train = self.evaluate(X["train"], y["train"], metric=metric, scaling=scaling)
            scores_val = self.evaluate(X["val"], y["val"], metric=metric, scaling=scaling)

            # if self.n_models > 1:
            #     for f in self.outputs:
            #         param_scores[f]["train"].update(dt.average(scores[f]["train"], axis=1, mergedict=True))
            #         param_scores[f]["val"].update(dt.average(scores[f]["val"], axis=1, mergedict=True))

            scores.append({"model_params": m_params, "train_params": t_params,
                           "train_scores": scores_train, "val_scores": scores_val})

        # if filename != "":
        #     logger.save_json(scores, filename=filename + '.json', logtime=logtime)

        return scores

    def summary(self, save_model_params=True, save_train_params=True,
                save_history=True, save_io=True,
                filename="", logtime=False, logger=None, show=False):

        # TODO: save weights

        logger = logger or Logger()

        text = ""
        params = {}

        text += f"## Model - {self}\n\n"

        # model parameters
        model_params = self.get_model_params(filename="")

        if save_model_params is True:
            params["model_params"] = model_params

        if len(model_params) == 0:
            text += "No model parameters"
        else:
            text += "Model parameters:\n"
            text += logger.dict_to_text(model_params)

        # train parameters
        train_params = self.get_train_params()

        if save_model_params is True:
            params["train_params"] = train_params

        if len(train_params) > 0:
            text += "\n\nTrain parameters:\n"
            text += logger.dict_to_text(train_params)

        # inputs and outputs
        if self.inputs is not None:
            name = self.inputs.name or "Inputs"

            if save_io is True:
                params["inputs"] = self.inputs.summary(name=name)

            text += "\n\n"
            text += self.inputs.summary(name=name, mode='text')

        if self.outputs is not None:
            name = self.outputs.name or "Outputs"

            if save_io is True:
                params["outputs"] = self.outputs.summary(name=name)

            text += "\n\n"
            text += self.outputs.summary(name=name, mode='text')

        # history
        if save_history is True and filename != "":
            fname = filename + "_history.json"
        else:
            fname = ""
        history = self.get_history(average=False, filename=fname, logtime=logtime, logger=logger)

        # display text and save
        if show is True:
            print(text)

        if len(params) > 0 and filename != "":
            logger.save_json(params, filename=filename + "_params.json", logtime=logtime)

        return text
