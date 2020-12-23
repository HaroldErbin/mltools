# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from mltools.analysis.logger import Logger
from mltools.analysis.evaluation import TensorEval
from mltools.analysis import describe

from mltools.data import datatools
from mltools.data.features import CategoricalFeatures
from mltools.data.structure import DataStructure


# TODO:
# - for tensors of integers, histogram must have bin length = 1 minimum
# - combine predictions for training and test set
# - write function to convert from dict to dataframe (taking into account
#   tensors, etc.)


class Predictions:

    def __init__(self, X, y_pred=None, y_true=None, y_std=None,
                 model=None, inputs=None, outputs=None, metrics=None,
                 categories_fn=None, integers_fn=None, postprocessing_fn=None,
                 mode='dict', logger=None):
        """
        Inits Predictions class.

        The goal of this class is to first convert the ML output to meaningful
        results. Then, to compare with the real data if they are known.

        The inputs, targets and predictions as given in outputs (or deduced
        from the model) are stored in `X`, `y_true` and `y_pred`. They are
        represented as dict. Other formats can be accessed through the `get_*`
        methods. Available formats are `dict` and `dataframe`, and the default
        is defined in `mode`. The reason for not using arrays is that some
        features need more information (probability distribution...) which
        would be more difficult to represent as arrays.

        The predictions can be given from an ensemble, in which case they are
        stored in `all_y_pred`, and the mean value and standard deviation are
        computed and stored in `y_pred` and `y_std`.
        It is necessary to keep track of all predictions for computing the
        standard deviation of derived quantities (metrics, errors...).

        `metrics` is a dict mapping each feature to a default metric for each
        feature. If a feature has no default metric, this will use the
        metric defined in the correspondng evaluation class.

        `categories` and `integers` are dict mapping features to decision
        functions (to be applied to each sample). They can also be given as
        a list, in which case the default decision is applied:
        - integers: round
        - categories: 1 if p > 0.5 else 0 (or max p is multi-class)

        The `postprocessing_fn` is applied after everything else and can be
        used to perform additional processing of the predictions.

        The predictions for each feature are stored in `predictions`, which
        uses a different class for each type of data. These classes implement
        a set of common methods, but whose arguments may change. Arguments
        not common to all classes must be passed by name.

        Format indicates if data is stored as a dict or as a dataframe.
        """

        # TODO: add CategoricalFeatures as argument, convert data

        # TODO: update to take into account ensemble (keep all results
        #   in another variable)

        self.logger = logger
        self.model = model

        if mode not in ('dataframe', 'dict'):
            raise ValueError("Format `{}` is not supported. Available "
                             "data format are: dict, dataframe."
                             .format(mode))
        else:
            self.mode = mode

        if metrics is None:
            self.metrics = {}
        else:
            self.metrics = metrics
        # TODO: check that metrics is a dict

        # check inputs, outputs, and put in form the data
        self._check_io_data(inputs, outputs, X, y_pred, y_true, y_std)

        # apply processing to predictions
        self._process_predictions(categories_fn, integers_fn,
                                  postprocessing_fn)

        # evaluate predictions for each feature based on its type
        self.predictions = self._typed_predictions()

    def __getitem__(self, key):
        return self.get_feature(key)

    def _check_io_data(self, inputs, outputs, X, y_pred, y_true, y_std):

        # if inputs/outputs is not given, check if they can be deduced from
        # the model
        if inputs is None:
            if self.model is not None and self.model.inputs is not None:
                self.inputs = self.model.inputs
            else:
                self.inputs = None
        else:
            self.inputs = inputs

        if outputs is None:
            if self.model is not None and self.model.outputs is not None:
                self.outputs = self.model.outputs
            else:
                self.outputs = None
        else:
            self.outputs = outputs

        # if targets are not given, extract them from X if `outputs` is given
        if y_true is None:
            if self.outputs is not None:
                self.y_true = self.outputs(X, mode='col')
            else:
                #  TODO: for unsupervised, y_true can remain None
                raise ValueError("Targets `y_true` must be given or one of "
                                 "`outputs` or `model.outputs` must not be"
                                 "`None` to extract the targets from `X`.")
        else:
            if self.outputs is not None:
                self.y_true = self.outputs(self.y_true, mode='col')
            else:
                self.y_true = y_true

        if not isinstance(self.y_true, dict):
            raise TypeError("`y_true` must be a `dict`, found {}."
                            .format(type(self.y_true)))

        if y_pred is None:
            # if predictions are not given, compute them from the model
            # and input data
            self.y_pred = self.model.predict(X, return_all=True)
        else:
            self.y_pred = y_pred

        # if predictions are a list, assume it comes from an ensemble
        # store mean value in y_pred, keep all predictions in all_y_pred
        # note that the latter is a list of dict, instead of a dict of list
        # TODO: change this?
        if isinstance(self.y_pred, list):
            self.all_y_pred = self.y_pred

            if self.outputs is not None:
                self.y_pred, self.y_std = self.outputs.average(self.all_y_pred)
            else:
                self.y_pred, self.y_std = datatools.average(self.all_y_pred)
        else:
            self.y_std = None
            self.all_y_pred = None

        if self.outputs is not None:
            self.y_pred = self.outputs(self.y_pred, mode='col')

        # override y_std if given
        if y_std is not None:
            self.y_std = y_std

        if not isinstance(self.y_pred, dict):
            raise TypeError("`y_pred` must be a `dict`, found {}."
                            .format(type(self.y_pred)))

        # get list of id
        if "id" in X:
            self.idx = X["id"]
        elif isinstance(X, pd.DataFrame):
            self.idx = X.index.to_numpy()
        else:
            # if no id is defined (in a column or as an index), then use
            # the list index?
            # list(range(len(X)))
            self.idx = None

        # after having extracted targets and predictions, we can filter
        # the inputs
        if self.inputs is not None:
            self.X = self.inputs(X, mode='col')
        else:
            self.X = X

        if not isinstance(self.X, dict):
            raise TypeError("`X` must be a `dict`, found {}."
                            .format(type(self.X)))

        # TODO: sort X and y by id if defined?

        # get target feature list
        if self.outputs is not None:
            self.features = self.outputs.features
        else:
            self.features = list(self.y_pred.keys())

    def _process_predictions(self, categories_fn, integers_fn,
                             postprocessing_fn):

        # store processing functions
        if categories_fn is None:
            self.categories_fn = {}
        elif isinstance(categories_fn, list):
            # TODO: default decision function:
            #   1 if p > 0.5 else 0 (or max p is multi-class)
            raise NotImplementedError
        else:
            self.categories_fn = categories_fn

        if integers_fn is None:
            self.integers_fn = {}
        elif isinstance(integers_fn, list):
            # default decision for integers: round
            self.integers_fn = {col: np.round for col in integers_fn}
        else:
            self.integers_fn = integers_fn

        self.postprocessing_fn = postprocessing_fn

        # process data
        for col, fn in self.categories_fn.items():
            if col not in self.y_pred:
                continue

            self.y_pred[col] = fn(self.y_pred[col])

            if self.all_y_pred is not None:
                for i, y in enumerate(self.all_y_pred):
                    self.all_y_pred[i][col] = fn(y[col])

        for col, fn in self.integers_fn.items():
            if col not in self.y_pred:
                continue

            self.y_pred[col] = fn(self.y_pred[col]).astype(int)

            if self.all_y_pred is not None:
                for i, y in enumerate(self.all_y_pred):
                    self.all_y_pred[i][col] = fn(y[col]).astype(int)

        if self.postprocessing_fn is not None:
            self.y_pred = self.postprocessing_fn(self.y_pred)

    def _typed_predictions(self):
        """
        Built dict of predictions based on data type.
        """

        dic = {}

        for k in self.features:
            metric = self.metrics.get(k, None)
            eval_method = self._prediction_type(k)

            std = None if self.y_std is None else self.y_std[k]
            if self.all_y_pred is None:
                all_pred = None
            else:
                all_pred = [p[k] for p in self.all_y_pred]

            dic[k] = eval_method(k, self.y_pred[k], self.y_true[k],
                                 all_pred, std, self.idx, metric, self.logger)

        return dic

    def _prediction_type(self, feature):
        """
        Find the appropriate class to evaluate the feature.
        """

        # TODO: update to handle other types
        # check in CategoricalFeatures for classification

        return TensorPredictions

    def get_X(self, mode=""):
        mode = mode or self.mode

        if mode == 'dataframe':
            return pd.DataFrame(self.X)
        else:
            return self.X

    def get_y_pred(self, mode=""):
        mode = mode or self.mode

        if mode == 'dataframe':
            return pd.DataFrame(self.y_pred)
        else:
            return self.y_pred

    def get_y_true(self, mode=""):
        mode = mode or self.mode

        if mode == 'dataframe':
            return pd.DataFrame(self.y_true)
        else:
            return self.y_true

    def get_errors(self, mode="", relative=False, norm=False):
        # TODO: arguments signed?

        mode = mode or self.mode

        pred_items = self.predictions.items()

        errors = {f: p.get_errors(relative=relative, norm=norm)
                  for f, p in pred_items}

        if mode == 'dataframe':
            return pd.DataFrame(errors)
        else:
            return errors

    def describe_errors(self, relative=False, percentiles=None):

        if percentiles is None:
            percentiles = [0.25, 0.5, 0.75, 0.95]

        return self.get_errors(mode="dataframe", relative=relative)\
                   .abs().describe(percentiles=percentiles)

    def get_feature(self, feature, mode="", filename="", logtime=True):

        return self.predictions[feature].get_feature(mode, filename, logtime)

    def get_all_features(self, mode="", filename="", logtime=True):

        if self.idx is not None:
            dic = {"id": self.idx}
        else:
            dic = {}

        for feature in self.features:
            dic.update(datatools.affix_keys(self.get_feature(feature),
                                            prefix="%s_" % feature))

        # remove duplicate id columns
        dic = {k: v for k, v in dic.items() if not k.endswith("_id")}

        mode = mode or self.mode

        df = pd.DataFrame(dic)

        if self.idx is not None:
            df = df.set_index("id").sort_values(by=['id'])
        if self.logger is not None:
            self.logger.save_csv(df, filename=filename, logtime=logtime)

        if mode == "dataframe":
            return df

        else:
            return dic

    def feature_metric(self, feature, metric=None, mode=None):

        return self.predictions[feature].feature_metric(metric, mode)

    def feature_all_metrics(self, feature, metrics=None, mode=None):

        return self.predictions[feature].feature_all_metrics(metrics, mode)

    def all_feature_metric(self, mode=None):

        # TODO: add dataframe mode?

        if mode == "text":
            results = {f: self.feature_metric(f, mode="text")
                       for f in self.features}

            return Logger.dict_to_text(results, sep=":\n ")
        else:
            return {f: self.feature_metric(f) for f in self.features}

    def all_feature_all_metrics(self, mode=None, filename="", logtime=True):

        # TODO: add dataframe mode?

        # TODO: don't compute if not needed (in text mode without filename)
        results = {f: self.feature_all_metrics(f) for f in self.features}

        if self.logger is not None:
            if self.all_y_pred is not None:
                json = {}
                for f, v in results.items():
                    json[f] = [v[0], {k + "_std": m for k, m in v[1].items()}]
            else:
                json = results

            self.logger.save_json(json, filename=filename, logtime=logtime)

        if mode == "text":
            results = {f: self.feature_all_metrics(f, mode="dict_text")
                       for f in self.features}
            results = Logger.dict_to_text(results, sep="\n")

        return results

    def plot_feature(self, feature, plottype="step", sigma=1, density=True,
                     bins=None, log=False, filename="", logtime=True,
                     **kwargs):

        return self.predictions[feature]\
                   .plot_feature(plottype, density, sigma, bins, log,
                                 filename, logtime, **kwargs)

    def plot_all_features(self, plottype="step", sigma=1, density=True,
                          bins=None, log=False, filename="", logtime=True):

        figs = []

        for feature in self.features:
            figs.append(self.plot_feature(feature, plottype, sigma, density,
                                          bins, log))

        if self.logger is not None:
            self.logger.save_figs(figs, filename, logtime)

        return figs

    def plot_errors(self, feature, relative=False, signed=True,
                    density=True, bins=None, log=False,
                    filename="", logtime=True, **kwargs):
        """
        Plot feature errors.

        Non-universal arguments:
        - `norm` (valid for tensor).
        """

        return self.predictions[feature]\
                   .plot_errors(relative, signed, density, bins, log,
                                filename, logtime, **kwargs)

    def plot_all_errors(self, relative=False, signed=True,
                        density=True, bins=None, log=False,
                        filename="", logtime=True, **kwargs):

        figs = []

        for feature in self.features:
            figs.append(self.plot_errors(feature, relative, signed,
                                         density, bins, log, **kwargs))

        if self.logger is not None:
            self.logger.save_figs(figs, filename, logtime)

        return figs

    def plot_feature_errors(self, feature, sigma=1, signed_errors=True,
                            density=True, bins=None, log=False,
                            filename="", logtime=True, **kwargs):

        return self.predictions[feature]\
                   .plot_feature_errors(sigma, signed_errors, density, bins,
                                        log, filename, logtime, **kwargs)

    def training_curve(self, metric='loss', history=None, history_std=None,
                       log=True, marker=None, filename="", logtime=False):
        """
        Loss evolution during training.

        If the training is performed over an ensemble, history can be of
        different lengths in case of early stopping. In that case, it is
        necessary to pad each history such that they have the same length.
        We choose to pad with the last value (which may not correspond to
        the best model).
        """

        # TODO: improve and make more generic

        # TODO: in Model, add as private attribute
        non_metric_keys = {"epochs", "train_time", "process_time"}

        if history is None:
            history = self.model.history.copy()

            if self.model.n_models > 1:
                history, history_std = datatools.average(history)

        if history is None or len(set(history.keys()) - non_metric_keys) == 0:
            return

        if isinstance(history, dict):
            val_history = history.get(f"val_{metric}", None)
            history = history[metric]
        else:
            val_history = None

        if isinstance(history_std, dict):
            val_history_std = history_std.get(f"val_{metric}", None)
            history_std = history_std.get(metric, None)
        else:
            val_history_std = None
            history_std = None

        if log is True and history_std is not None:
            # error bar in log scale is given by rescaled relative error
            # TODO: create function to compute it
            history_std *= np.log10(np.e) / history
            if val_history_std is not None:
                val_history_std *= np.log10(np.e) / val_history

        fig, ax = plt.subplots()

        logger = self.logger or Logger
        styles = logger.styles

        steps = np.arange(1, len(history) + 1, dtype=int)

        ax.plot(steps, history[:], linestyle='solid', marker=marker,
                color=styles["color:train"], label=styles["label:train"])

        if history_std is not None:
            ax.fill_between(steps, history - history_std,
                            history + history_std,
                            alpha=0.3, color=styles["color:train"])

        if val_history is not None:
            ax.plot(steps, val_history, linestyle='solid', marker=marker,
                    color=styles["color:val"], label=styles["label:val"])

            if val_history_std is not None:
                ax.fill_between(steps, val_history - val_history_std,
                                val_history + val_history_std,
                                alpha=0.3, color=styles["color:val"])

        # TODO: plot position of best model if early stopping
        #       it is located at total_step - wait_step

        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        ax.legend()

        ax.set_xlabel('epochs')
        ax.set_ylabel(metric)

        if log is True:
            ax.set_yscale('log')

        if self.logger is not None:
            self.logger.save_fig(fig, filename=filename, logtime=logtime)

        return fig

    def summary_feature(self, feature, mode="", metrics=None, sigma=1,
                        signed_errors=True, density=True, bins=None,
                        log=False, filename="", logtime=True, **kwargs):

        return self.predictions[feature]\
                   .summary_feature(mode=mode, metrics=metrics, sigma=sigma,
                                    signed_errors=signed_errors,
                                    density=density, bins=bins, log=log,
                                    filename=filename, logtime=logtime,
                                    **kwargs)

    def summary(self, mode="", training_metrics=None, sigma=1,
                signed_errors=True, density=True, bins=None, log=False,
                include_io=False, filename="", logtime=True, show=False,
                extra_figs=None):

        # TODO: add information on inputs/outputs
        # TODO: add the number of models
        # TODO: add information on sampling

        # TODO: add computation of errors

        if filename == "":
            pdf_filename = ""
            csv_filename = ""
            metrics_filename = ""
            model_filename = ""
        else:
            pdf_filename = filename + "_summary.pdf"
            csv_filename = filename + "_predictions.csv"
            metrics_filename = filename + "_metrics.json"
            model_filename = filename + "_model"

        figs = []

        # first page of summary
        if self.logger is not None:
            intro_text = "# Summary -- {}\n\n"\
                            .format(self.logger.logtime_text())
        else:
            intro_text = "# Summary\n\n"
        intro_text += "Selected metrics:\n"
        intro_text += self.all_feature_metric(mode="text")

        figs.append(self.logger.text_to_fig(intro_text))

        if show is True:
            print(intro_text)

        # summary of all metrics
        metrics_text = "## Metrics\n\n"
        metrics_text += self.all_feature_all_metrics(mode="text",
                                                     filename=metrics_filename,
                                                     logtime=logtime)

        figs.append(self.logger.text_to_fig(metrics_text))

        if show is True:
            print()
            print(metrics_text)

        # table of errors
        error_text = "## Absolute errors -- table\n\n"
        error_text += str(self.describe_errors(relative=False))

        rel_error_text = "## Relative errors -- table\n\n"
        rel_error_text += str(self.describe_errors(relative=True))

        figs.append(self.logger.text_to_fig(error_text))
        figs.append(self.logger.text_to_fig(rel_error_text))

        if show is True:
            print()
            print(error_text)
            print()
            print(rel_error_text)

        # TODO: table of standard deviations

        # distributions and error plots
        figs += self.plot_all_features(sigma=sigma, plottype="step",
                                       density=density, bins=bins, log=log)
        # figs += self.plot_all_features(sigma=sigma, plottype="plain",
        #                                density=density, bins=bins, log=log)
        # figs += self.plot_all_features(sigma=sigma, plottype="seaborn",
        #                                density=density, bins=bins, log=log)
        figs += self.plot_all_errors(relative=False, signed=signed_errors,
                                     density=density, bins=bins,
                                     log=log)
        figs += self.plot_all_errors(relative=True, signed=signed_errors,
                                     density=density, bins=bins,
                                     log=log)

        # training curves
        # TODO: by default, plot curves for all model metrics
        if training_metrics is None:
            training_metrics = self.model.metrics
        else:
            if isinstance(training_metrics, str):
                training_metrics = [training_metrics]

        training_figs = [self.training_curve(log=True, metric=metric)
                         for metric in training_metrics]
        figs += [fig for fig in training_figs if fig is not None]

        # page on model information
        model_text = self.model.summary(filename=model_filename,
                                        logtime=logtime, logger=self.logger)

        figs.append(self.logger.text_to_fig(model_text))

        # page on input and output data structures
        # note: useful only if different from `model`
        if include_io is True:
            io_text = ""

            if self.inputs is not None:
                io_text += "\n"
                name = self.inputs.name or "Inputs"
                io_text += self.inputs.summary(name=name)

            if self.outputs is not None:
                io_text += "\n\n"
                name = self.outputs.name or "Outputs"
                io_text += self.outputs.summary(name=name)
                io_text += "\n"

            if io_text != "":
                io_text = "## Inputs and outputs\n\n" + io_text
                figs.append(self.logger.text_to_fig(io_text))

            if show is True:
                print()
                print(io_text)

        # additional figures created outside the function
        if extra_figs is not None:
            figs += extra_figs

        if self.logger is not None:
            self.logger.save_figs(figs, pdf_filename, logtime)

        # save results
        data = self.get_all_features(mode, csv_filename, logtime)

        for fig in figs:
            plt.close(fig)

        return data, figs


class TensorPredictions:
    """
    Evaluate predictions for a tensor.

    Tensor norms are computed  with the L2-norm, which cannot be changed.
    """

    # TODO: make scatter plot true/pred for 2d data
    #   (for d > 2, use reduction)

    method = TensorEval

    def __init__(self, feature, y_pred, y_true, all_y_pred=None, y_std=None,
                 idx=None, metric=None, logger=None):

        self.logger = logger
        self.idx = idx

        # default metric
        self.metric = metric

        # adding the feature name is necessary for plot labels
        self.feature = feature

        self.y_pred = y_pred
        self.y_true = y_true

        self.all_y_pred = all_y_pred
        self.y_std = y_std

        self.is_scalar = self.method.is_scalar(self.y_pred)

        self.errors = self.method.errors(y_true, y_pred)
        self.rel_errors = self.method.errors(y_true, y_pred, relative=True)

        if self.is_scalar is False:
            # TODO: add norm arguments?
            norm = 2

            self.norm_errors = self.method.norm_errors(y_true, y_pred, norm)
            self.rel_norm_errors = self.method.norm_errors(y_true, y_pred,
                                                           relative=True,
                                                           norm=norm)
        else:
            self.norm_errors = None
            self.rel_norm_errors = None

    def get_errors(self, mode="", relative=False, norm=False, **kwgargs):
        """
        Return errors.

        This method is just another way to retrieve the errors which are
        always save as attributes to avoid recomputing.

        `**kwargs` is used to catch parameters transmitted for other feature
        types.
        """

        if self.is_scalar is False and norm is True:
            if relative is True:
                return self.rel_norm_errors
            else:
                return self.norm_errors
        else:
            if relative is True:
                return self.rel_errors
            else:
                return self.errors

    def _pretty_metric_text(self, data, std=None, logger=None):

        dic = {}

        logger = logger or Logger
        float_fmt = logger.styles["print:float"]

        for k, v in data.items():
            # improve names
            new_k = self.method._metric_names.get(k, k)
            # format number according to float format
            dic[new_k] = float_fmt.format(v)

            if std is not None:
                dic[new_k] += " ± " + float_fmt.format(std[k])

        return datatools.equal_length_names(dic)

    def get_feature(self, mode="dict", filename="", logtime=True):
        """
        Summarize feature results in one dataframe.

        Return a dataframe with the following columns: real, pred, error, rel
        for a given feature.
        """

        if self.idx is not None:
            dic = {"id": self.idx}
        else:
            dic = {}

        dic["true"] = self.y_true
        dic["pred"] = self.y_pred

        if self.y_std is not None:
            dic["std"] = self.y_std

        dic["err"] = self.errors
        dic["rel_err"] = self.rel_errors

        if self.is_scalar is False:
            dic["norm_err"] = self.norm_errors

        # TODO: this wll not work with tensor, create appropriate function
        # to convert from dict

        df = pd.DataFrame(dic)

        if self.idx is not None:
            df = df.set_index("id").sort_values(by=['id'])

        if self.logger is not None:
            self.logger.save_csv(df, filename=filename, logtime=logtime)

        if mode == "dataframe":
            return df
        else:
            return dic

    def feature_metric(self, metric=None, mode=None):

        logger = self.logger or Logger
        float_fmt = logger.styles["print:float"]

        metric = metric or self.metric or self.method.default_metric

        if self.all_y_pred is not None:
            all_results = [self.method.evaluate(y, self.y_true, method=metric)
                           for y in self.all_y_pred]

            result = np.mean(all_results)
            std = np.std(all_results)
        else:
            result = self.method.evaluate(self.y_pred, self.y_true,
                                          method=metric)
            std = None

        if mode == "text":
            text = "{} = {}".format(self.method._metric_names
                                        .get(metric, metric),
                                    float_fmt.format(result))
            if std is not None:
                text += " ± {}".format(float_fmt.format(std))

            return text
        else:
            if std is None:
                return result
            else:
                return result, std

    def feature_all_metrics(self, metrics=None, mode=None, filename="",
                            logtime=True):

        logger = self.logger or Logger

        if self.all_y_pred is not None:
            all_results = [self.method.eval_metrics(y, self.y_true,
                                                    metrics=metrics)
                           for y in self.all_y_pred]

            results, std = datatools.average(all_results)
        else:
            results = self.method.eval_metrics(self.y_pred, self.y_true,
                                               metrics=metrics)
            std = None

        if self.logger is not None:
            if std is not None:
                json = {}
                for k in results:
                    json[k] = results[k]
                    json[k + "_std"] = std[k]
            else:
                json = results

            self.logger.save_json(json, filename=filename, logtime=logtime)

        if mode == "text" or mode == "dict_text":
            results = self._pretty_metric_text(results, std, logger)

            if mode == "dict_text":
                return results
            else:
                return logger.dict_to_text(results)
        elif mode == "series":
            return pd.Series(list(results.values()),
                             index=list(results.keys()))
        else:
            if std is None:
                return results
            else:
                return results, std

    def plot_feature(self, plottype="step", density=True, sigma=1, bins=None,
                     log=False, filename="", logtime=True, norm=True):
        """
        Plot the distribution of a feature.

        To get the best from this method, the class must have a `Logger`
        object.
        """

        # TODO: option to remove standard deviation

        logger = self.logger or Logger
        styles = logger.styles

        if norm is True and self.is_scalar is False:
            # TODO: change norm in argument?
            pred = self.method.norm(self.y_pred, norm=2)
            true = self.method.norm(self.y_true, norm=2)

            if self.y_std is not None:
                std = self.method.norm(self.y_std, norm=2)
            else:
                std = None
        else:
            pred = self.y_pred.reshape(-1)
            true = self.y_true.reshape(-1)

            if self.y_std is not None:
                std = self.y_std.reshape(-1)
            else:
                std = None

        xlabel = str(self.feature)

        if self.logger is not None:
            fig = describe.distribution(pred, true, std, sigma=sigma,
                                        plottype=plottype, density=density,
                                        bins=bins, log=log, xlabel=xlabel,
                                        logger=self.logger, filename=filename,
                                        logtime=logtime)
        else:
            fig, ax = plt.subplots()

            bins = bins or logger.find_bins(pred)
            ylabel = "PDF" if density is True else "Count"
            label = [styles["label:pred"], styles["label:true"]]
            color = [styles["color:pred"], styles["color:true"]]

            values, bins, _ = ax.hist([pred, true], linewidth=styles["linewidth:hist"],
                                      histtype='step', bins=bins,
                                      density=density, log=log,
                                      label=label, color=color)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()
            fig.tight_layout()

        plt.close(fig)

        return fig

    def plot_errors(self, relative=False, signed=True,
                    density=True, bins=None, log=False,
                    filename="", logtime=True,
                    norm=False):

        logger = self.logger or Logger
        styles = logger.styles

        # errors defined without sign in [Skiena, p. 222]

        # TODO: filter out infinite values for relative errors

        if norm is True and self.is_scalar is False:
            if relative is True:
                errors = self.rel_norm_errors
                xlabel = "%s (norm relative errors)" % self.feature
            else:
                errors = self.norm_errors
                xlabel = "%s (norm errors)" % self.feature
        else:
            if relative is True:
                if signed is True:
                    errors = self.rel_errors
                    xlabel = "%s (relative errors)" % self.feature
                else:
                    errors = np.abs(self.rel_errors)
                    xlabel = "%s (unsigned relative errors)" % self.feature
            else:
                if signed is True:
                    errors = self.errors
                    xlabel = "%s (absolute errors)" % self.feature
                else:
                    errors = np.abs(self.errors)
                    xlabel = "%s (unsigned absolute errors)" % self.feature

        if density is True:
            ylabel = "PDF"
            density = True
        else:
            ylabel = "Count"
            density = False

        bins = bins or logger.find_bins(errors)

        fig, ax = plt.subplots()

        # hack to remove infinite values
        # this happens when considering relative errors in the case where
        # the original value is zero
        # TODO: find better way
        err = np.array([e for e in errors if e != np.inf])

        ax.hist(err, linewidth=styles["linewidth:hist"], histtype='step', bins=bins,
                density=density, log=log,
                color=styles["color:errors"])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        fig.tight_layout()

        if self.logger is not None:
            self.logger.save_fig(fig, filename, logtime)

        plt.close(fig)

        return fig

    def plot_feature_errors(self, sigma=1, signed_errors=True,
                            density=True, bins=None, log=False,
                            filename="", logtime=True, norm=True):

        feat1 = self.plot_feature(plottype="step", sigma=sigma,
                                  density=density, bins=bins, log=log,
                                  norm=norm)

        feat2 = self.plot_feature(plottype="seaborn", sigma=sigma,
                                  density=density, bins=bins, log=log,
                                  norm=norm)

        feat3 = self.plot_feature(plottype="line", sigma=sigma,
                                  density=density, bins=bins, log=log,
                                  norm=norm)

        feat4 = self.plot_feature(plottype="plain", sigma=sigma,
                                  density=density, bins=bins, log=log,
                                  norm=norm)

        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
        # axes[0] = fig.axes[0]
        # axes[1] = fig.axes[0]

        err1 = self.plot_errors(relative=False, signed=signed_errors,
                                density=density, bins=bins, log=log,
                                norm=norm)

        err2 = self.plot_errors(relative=True, signed=signed_errors,
                                density=density, bins=bins, log=log,
                                norm=norm)

        figs = [feat1, feat2, feat3, feat4, err1, err2]

        if self.logger is not None:
            self.logger.save_figs(figs, filename, logtime)

        return figs

    def summary_feature(self, mode="", metrics=None, sigma=1,
                        signed_errors=True, density=True, bins=None, log=False,
                        filename="", logtime=True, norm=True):

        # TODO: table of errors (percentiles, max, min)?

        # TODO: add computation of errors

        if filename == "":
            fig_filename = ""
            csv_filename = ""
            json_filename = ""
        else:
            fig_filename = filename + "_plot.pdf"
            csv_filename = filename + "_predictions.csv"
            json_filename = filename + "_metrics.json"

        metric_results = self.feature_all_metrics(metrics,
                                                  filename=json_filename,
                                                  logtime=logtime)

        figs = self.plot_feature_errors(sigma, signed_errors, density,
                                        bins, log, filename=fig_filename,
                                        logtime=logtime, norm=norm)

        data = self.get_feature(mode, filename=csv_filename,
                                logtime=logtime)

        for fig in figs:
            plt.close(fig)

        return metric_results, data, figs


class BinaryPredictions:

    # TODO: prepare metrics dict using Logger.styles (convert to dict of str)
    #   indeed, it's not possible to write a generic dict_to_text method
    #   since it requires knowing about the custom styles and feature names

    # TODO: probbility distribution
    # probability mean value, variance

    pass
