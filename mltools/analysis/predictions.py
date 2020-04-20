# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from mltools.analysis.logger import Logger
from mltools.analysis.evaluation import TensorEval

from mltools.data import datatools
from mltools.data.features import CategoricalFeatures
from mltools.data.structure import DataStructure


# TODO: update class (or split) for unsupervised algorithms
#   visualizations of results is useful for both supervised/unsupervised
#   but discussion of errors is valid only for supervised
# TODO: make code compatible with X and y written as array
# TODO: write function to convert from dict to dataframe (taking into account
#   tensors, etc.)


class Predictions:

    def __init__(self, X, y_pred=None, y_true=None, y_std=None,
                 model=None, inputs=None, outputs=None,
                 categories_fn=None, integers_fn=None, postprocessing_fn=None,
                 mode='dict', logger=None):
        """
        Inits Predictions class.

        The goal of this class is to first convert the ML output to meaningful
        results. Then, to compare with the real data if they are known.

        The inputs, targets and predictions as given in outputs (or deduced
        frm the model) are stored in `X`, `y_true` and `y_pred`. They are
        represented as dict. Other formats can be accessed through the `get_*`
        methods. Available formats are `dict` and `dataframe`, and the default
        is defined in `mode`. The reason for not using arrays is that some
        features need more information (probability distribution...) which
        would be more difficult to represent as arrays.

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

        # if predictions are not given, compute them from the model
        # and input data
        if y_pred is None:
            self.y_pred = self.model.predict(X)
        else:
            self.y_pred = y_pred

        if self.outputs is not None:
            self.y_pred = self.outputs(self.y_pred, mode='col')

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

        # TODO: add this
        # self.y_std = y_std

        # get target feature list
        if self.outputs is not None:
            self.features = self.outputs.features
        else:
            self.features = list(self.y_pred.keys())

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

        # apply processing to predictions
        self._process_predictions()

        # evaluate predictions for each feature based on its type
        self.predictions = self._typed_predictions()

        # compute errors
        self.errors = {f: p.errors for f, p in self.predictions.items()}
        self.rel_errors = {f: p.errors for f, p in self.predictions.items()}

    def __getitem__(self, key):
        return self.get_feature(key)

    def _typed_predictions(self):
        """
        Built dict of predictions based on data type.
        """

        dic = {}

        for k in self.features:
            dic[k] = self._prediction_type(k)(k, self.y_pred[k], self.y_true[k],
                                              self.idx, self.logger)

        return dic

    def _prediction_type(self, feature):
        """
        Find the appropriate class to evaluate the feature.
        """

        # TODO: update to handle other types
        # check in CategoricalFeatures for classification

        return TensorPredictions

    def _process_predictions(self):

        for col, fn in self.categories_fn.items():
            self.y_pred[col] = fn(self.y_pred[col])

        for col, fn in self.integers_fn.items():
            self.y_pred[col] = fn(self.y_pred[col]).astype(int)

        if self.postprocessing_fn is not None:
            self.y_pred = self.postprocessing_fn(self.y_pred)

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

    def get_errors(self, mode=""):
        mode = mode or self.mode

        if mode == 'dataframe':
            return pd.DataFrame(self.errors)
        else:
            return self.errors

    def get_rel_errors(self, mode=""):
        mode = mode or self.mode

        if mode == 'dataframe':
            return pd.DataFrame(self.rel_errors)
        else:
            return self.rel_errors

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

    def plot_feature(self, feature, normalized=True, bins=None, log=False,
                     filename="", logtime=True, **kwargs):

        return self.predictions[feature]\
                   .plot_feature(normalized, bins, log,
                                 filename, logtime, **kwargs)

    def plot_all_features(self, normalized=True, bins=None, log=False,
                          filename="", logtime=True):

        figs = []

        for feature in self.features:
            figs.append(self.plot_feature(feature, normalized, bins, log))

        if self.logger is not None:
            self.logger.save_figs(figs, filename, logtime)

        return figs

    def plot_errors(self, feature, relative=False, signed=True,
                    normalized=True, bins=None, log=False,
                    filename="", logtime=True, **kwargs):
        """
        Plot feature errors.

        Non-universal arguments:
        - `norm` (valid for tensor).
        """

        return self.predictions[feature]\
                   .plot_errors(relative, signed, normalized, bins, log,
                                filename, logtime, **kwargs)

    def plot_all_errors(self, relative=False, signed=True, norm=True,
                        normalized=True, bins=None, log=False,
                        filename="", logtime=True):

        figs = []

        for feature in self.features:
            figs.append(self.plot_errors(feature, relative, signed,
                                         normalized, bins, log,
                                         norm=norm))

        if self.logger is not None:
            self.logger.save_figs(figs, filename, logtime)

        return figs

    def plot_feature_errors(self, feature, signed_errors=True,
                            normalized=True, bins=None, log=False,
                            filename="", logtime=True, **kwargs):

        return self.predictions[feature]\
                   .plot_feature_errors(signed_errors, normalized, bins, log,
                                        filename, logtime, **kwargs)

    def training_curve(self, history=None, log=True, filename="",
                       logtime=False):

        # TODO: improve and make more generic

        try:
            history = history or self.model.model.history
        except AttributeError:
            raise TypeError("Model `{}` is not trained by steps and has no "
                            "has no `history` attribute.".format(self.model))

        fig, ax = plt.subplots()

        styles = self.logger.styles

        ax.plot(history.history['loss'][:], 'o-',
                color=styles["color:train"], label=styles["label:train"])

        try:
            ax.plot(history.history['val_loss'][:], 'o--',
                    color=styles["color:val"], label=styles["label:val"])
        except (KeyError, IndexError):
            pass

        ax.legend()

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')

        if log is True:
            ax.set_yscale('log')

        self.logger.save_fig(fig, filename=filename, logtime=logtime)

        return fig

    def summary_feature(self, feature, mode="", signed_errors=True,
                        normalized=True, bins=None, log=False,
                        filename="", logtime=True, **kwargs):

        return self.predictions[feature]\
                   .summary_feature(mode, signed_errors, normalized, bins,
                                    log, filename, logtime, **kwargs)

    def summary(self, mode="", signed_errors=True,
                normalized=True, bins=None, log=False,
                filename="", logtime=True):

        # TODO: add computation of errors

        figs = []

        # distributions and error plots
        figs += self.plot_all_features(normalized=normalized, bins=bins,
                                       log=log)
        figs += self.plot_all_errors(relative=False, signed=signed_errors,
                                     normalized=normalized, bins=bins,
                                     log=log)
        figs += self.plot_all_errors(relative=True, signed=signed_errors,
                                     normalized=normalized, bins=bins,
                                     log=log)

        model_text = self.logger.dict_to_text(self.model.model_params)
        if model_text == "":
            model_text = "No parameters"
        else:
            model_text = "Parameters:\n" + model_text
        model_text = "Model - %s\n\n" % self.model + model_text

        if len(self.model.train_params_history) > 0:
            model_text += "\n\nTrain parameters:\n"
            model_text += self.logger.dict_to_text(self.model.get_train_params)

        try:
            figs.append(self.training_curve(log=True))
        except TypeError:
            pass

        figs.append(self.logger.text_to_fig(model_text))

        if self.logger is not None:
            self.logger.save_figs(figs, "%s_summary.pdf" % filename, logtime)

        # save results
        data = self.get_all_features(mode, "%s_predictions.csv" % filename,
                                     logtime)

        # save model parameters
        self.model.save_params(filename="%s_model_params.json" % filename,
                               logtime=logtime, logger=self.logger)

        return data, figs


class TensorPredictions:
    """
    Evaluate predictions for a tensor.

    Tensor norms are computed  with the L2-norm, which cannot be changed.
    """

    method = TensorEval

    def __init__(self, feature, y_pred, y_true, idx=None, logger=None):

        self.logger = logger
        self.idx = idx

        # adding the feature name is necessary for plot labels
        self.feature = feature

        self.y_pred = y_pred
        self.y_true = y_true

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

        dic.update({"true": self.y_true, "pred": self.y_pred,
                    # feature + "std": self.y_std[feature],
                    "err": self.errors,
                    "rel": self.rel_errors})

        if self.is_scalar is False:
            dic.update({"norm": self.norm_errors})

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

    def plot_feature(self, normalized=True, bins=None, log=False,
                     filename="", logtime=True, norm=True):

        # TODO: add ML standard deviation

        logger = self.logger or Logger
        styles = logger.styles

        if norm is True and self.is_scalar is False:
            # TODO: change norm in argument?
            pred = self.method.norm(self.y_pred, norm=2)
            true = self.method.norm(self.y_true, norm=2)
        else:
            pred = self.y_pred.reshape(-1)
            true = self.y_true.reshape(-1)

        xlabel = "{}".format(self.feature)

        if normalized is True:
            ylabel = "PDF"
            density = True
        else:
            ylabel = "Count"
            density = False

        if bins is None:
            # TODO: add option for this behaviour?
            bins = logger.find_bins(pred)

        fig, ax = plt.subplots()

        ax.hist([pred, true], linewidth=1., histtype='step',
                bins=bins, density=density, log=log,
                label=[styles["label:pred"], styles["label:true"]],
                color=[styles["color:pred"], styles["color:true"]])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.legend()

        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        fig.tight_layout()

        if self.logger is not None:
            self.logger.save_fig(fig, filename, logtime)

        return fig

    def plot_errors(self, relative=False, signed=True,
                    normalized=True, bins=None, log=False,
                    filename="", logtime=True,
                    norm=False):

        logger = self.logger or Logger

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

        if normalized is True:
            ylabel = "PDF"
            density = True
        else:
            ylabel = "Count"
            density = False

        if bins is None:
            bins = logger.find_bins(errors)

        fig, ax = plt.subplots()

        ax.hist(errors, linewidth=1., histtype='step', bins=bins,
                density=density, log=log,
                color=logger.styles["color:errors"])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        fig.tight_layout()

        if self.logger is not None:
            self.logger.save_fig(fig, filename, logtime)

        return fig

    def plot_feature_errors(self, signed_errors=True,
                            normalized=True, bins=None, log=False,
                            filename="", logtime=True, norm=True):

        fig_feature = self.plot_feature(normalized=normalized,
                                        bins=bins, log=log, norm=norm)

        fig_abs_error = self.plot_errors(relative=False,
                                         signed=signed_errors,
                                         normalized=normalized,
                                         bins=bins, log=log, norm=norm)

        fig_rel_error = self.plot_errors(relative=True,
                                         signed=signed_errors,
                                         normalized=normalized,
                                         bins=bins, log=log, norm=norm)

        figs = [fig_feature, fig_abs_error, fig_rel_error]

        if self.logger is not None:
            self.logger.save_figs(figs, filename, logtime)

        return figs

    def summary_feature(self, mode="", signed_errors=True,
                        normalized=True, bins=None, log=False,
                        filename="", logtime=True, norm=True):

        # TODO: table of errors (percentiles, max, min)?

        # TODO: add computation of errors

        figs = self.plot_feature_errors(signed_errors, normalized,
                                        bins, log, filename=filename + ".pdf",
                                        logtime=logtime, norm=norm)

        data = self.get_feature(mode, filename=filename + ".csv",
                                logtime=logtime)

        return data, figs


class BinaryPredictions:

    pass
