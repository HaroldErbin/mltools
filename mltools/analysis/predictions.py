# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from mltools.analysis.logger import Logger

# TODO: update class (or split) for unsupervised algorithms
#   visualizations of results is useful for both supervised/unsupervised
#   but discussion of errors is valid only for supervised


# TODO: make code compatible with X and y written as array


class Predictions:

    def __init__(self, X, y_pred=None, y_true=None, y_std=None,
                 model=None, inputs=None, outputs=None,
                 categories=None, integers=None, postprocessing_fn=None,
                 fmt='dict', logger=None):
        """
        Inits Predictions class.

        The goal of this class is to first convert the ML output to meaningful
        results. Then, to compare with the real data if they are known.

        `categories` and `integers` are dict mapping features to decision
        functions (to be applied to each sample). They can also be given as
        a list, in which case the default decision is applied:
        - integers: round
        - categories: 1 if p > 0.5 else 0 (or max p is multi-class)

        The `postprocessing_fn` is applied after everything else and can be
        used to perform additional processing of the predictions.

        Format indicates if data is stored as a dict or as a dataframe.
        """

        # TODO: update to take into account ensemble (keep all results
        #   in another variable)

        self.logger = logger
        self.fmt = fmt
        self.model = model

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

        # if predictions are not given, compute them from the model
        # and input data
        if y_pred is None:
            self.y_pred = self.model.predict(X)
        else:
            self.y_pred = y_pred

        if self.outputs is not None:
            self.y_pred = self.outputs(self.y_pred, mode='col')

        # get list of id
        if "id" in X:
            self.id = X["id"]
        elif isinstance(X, pd.DataFrame):
            self.id = list(X.index)
        else:
            # if no id is defined (in a column or as an index), then use
            # the list index?
            # list(range(len(X)))
            self.id = None

        # after having extracted targets and predictions, we can filter
        # the inputs
        if self.inputs is not None:
            self.X = self.inputs(X, mode='col')
        else:
            self.X = X

        # TODO: sort X and y by id if defined?

        # TODO: add this
        # self.y_std = y_std

        if categories is None:
            self.categories = {}
        elif isinstance(categories, list):
            # TODO: default decision function:
            #   1 if p > 0.5 else 0 (or max p is multi-class)
            raise NotImplementedError
        else:
            self.categories = categories

        if integers is None:
            self.integers = {}
        elif isinstance(integers, list):
            # default decision for integers: round
            self.integers = {col: np.round for col in integers}
        else:
            self.integers = integers

        self.postprocessing_fn = postprocessing_fn

        # apply processing to predictions
        self._process_predictions()

        # compute errors
        if self.y_true is not None:
            self.errors = self._compute_errors()
            self.rel_errors = self._compute_relative_errors()

        # TODO: how to handle tensors?
        #   first convert with datastructure to dict? (to work on array)
        #   errors for tensors: l-norm error for each sample,
        #                       signed error component by component

    def _compute_errors(self):
        if self.y_true is None:
            raise ValueError("Need real results to compute error.")

        errors = {}

        for k, t, p in zip(self.outputs.features,
                           self.outputs(self.y_true, mode='col').values(),
                           self.outputs(self.y_pred, mode='col').values()):
            errors[k] = np.subtract(t, p)

        return errors

    def _compute_relative_errors(self):
        if self.y_true is None:
            raise ValueError("Need real results to compute error.")

        errors = {}

        for k, t, p in zip(self.outputs.features,
                           self.outputs(self.y_true, mode='col').values(),
                           self.outputs(self.y_pred, mode='col').values()):
            errors[k] = np.subtract(t, p) / np.abs(t)

        return errors

    def _process_predictions(self):

        for col, fn in self.categories.items():
            self.y_pred[col] = fn(self.y_pred[col])

        for col, fn in self.integers.items():
            self.y_pred[col] = fn(self.y_pred[col])

        if self.postprocessing_fn is not None:
            self.y_pred = self.postprocessing_fn(self.y_pred)

    def __getitem__(self, key):
        return self.get_feature(key)

    def get_feature(self, feature, fmt=None, filename="", logtime=True):
        """
        Summarize feature results in one dataframe.

        Return a dataframe with the following columns: real, pred, error, rel
        for a given feature.
        """

        if self.y_true is None:
            return self.y_pred[feature]

        if self.id is not None:
            dic = {"id": self.id}
        else:
            dic = {}

        dic.update({feature + "_true": self.y_true[feature],
                    feature + "_pred": self.y_pred[feature],
                    # feature + "_std": self.y_std[feature],
                    feature + "_err": self.errors[feature],
                    feature + "_rel": self.rel_errors[feature]})

        fmt = fmt or self.fmt

        df = pd.DataFrame(dic)

        if self.id is not None:
            df = df.set_index("id").sort_values(by=['id'])

        if self.logger is not None:
            self.logger.save_csv(df, filename=filename, logtime=logtime)

        if fmt == "dataframe":
            return df
        else:
            return dic

    def get_all_features(self, fmt=None, filename="", logtime=True):

        if self.id is not None:
            dic = {"id": self.id}
        else:
            dic = {}

        for feature in self.y_pred:
            dic.update(self.get_feature(feature))

        fmt = fmt or self.fmt

        df = pd.DataFrame(dic)

        if self.id is not None:
            df = df.set_index("id").sort_values(by=['id'])

        if self.logger is not None:
            self.logger.save_csv(df, filename=filename, logtime=logtime)

        if fmt == "dataframe":
            return df
        else:
            return dic

    def plot_feature(self, feature, normalized=True, bins=None, log=False,
                     filename="", logtime=True):
        # errors defined without sign in [Skiena, p. 222]

        # TODO: plot_feature calls methods adapted to target type
        #  (real, binary, ...)
        #  if type has no function, do not plot

        # TODO: add standard deviation

        logger = self.logger or Logger
        styles = logger.styles

        pred = self.y_pred[feature]
        true = self.y_true[feature]

        xlabel = "{}".format(feature)

        if normalized is True:
            ylabel = "PDF"
            density = True
        else:
            ylabel = "Count"
            density = False

        if bins is None:
            bins = logger.find_bins(pred)

        fig, ax = plt.subplots()

        ax.hist([pred, true], linewidth=1., histtype='step', bins=bins,
                density=density, log=log,
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

    def plot_all_features(self, normalized=True, bins=None, log=False,
                          filename="", logtime=True):

        figs = []

        for feature in self.y_pred:
            figs.append(self.plot_feature(feature, normalized, bins, log))

        if self.logger is not None:
            self.logger.save_figs(figs, filename, logtime)

        return figs

    def plot_errors(self, feature, relative=False, signed=True,
                    normalized=True, bins=None, log=False,
                    filename="", logtime=True):

        logger = self.logger or Logger

        # errors defined without sign in [Skiena, p. 222]

        if relative is True:
            if signed is True:
                errors = self.rel_errors[feature]
                xlabel = "{} (relative errors)".format(feature)
            else:
                errors = self.rel_errors[feature]
                xlabel = "{} (unsigned relative errors)".format(feature)
        else:
            if signed is True:
                errors = self.errors[feature]
                xlabel = "{} (absolute errors)".format(feature)
            else:
                errors = self.errors[feature]
                xlabel = "{} (unsigned absolute errors)".format(feature)

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

    def plot_all_errors(self, relative=False, signed=True, normalized=True,
                        bins=None, log=False, filename="", logtime=True):

        figs = []

        for feature in self.y_pred:
            figs.append(self.plot_errors(feature, relative, signed, normalized,
                                         bins, log))

        if self.logger is not None:
            self.logger.save_figs(figs, filename, logtime)

        return figs

    def plot_feature_errors(self, feature, signed_errors=True,
                            normalized=True, bins=None, log=False,
                            filename="", logtime=True):

        fig_feature = self.plot_feature(feature, normalized, bins, log)

        fig_abs_error = self.plot_errors(feature, relative=False,
                                         signed=signed_errors,
                                         normalized=normalized,
                                         bins=bins, log=log)

        fig_rel_error = self.plot_errors(feature, relative=True,
                                         signed=signed_errors,
                                         normalized=normalized,
                                         bins=bins, log=log)

        figs = (fig_feature, fig_abs_error, fig_rel_error)

        if self.logger is not None:
            self.logger.save_figs(figs, filename, logtime)

        return figs

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

        self.logger(fig, filename=filename, logtime=logtime)

        return fig

    def summary_feature(self, feature, fmt=None, signed_errors=True,
                        normalized=True, bins=None, log=False,
                        filename="", logtime=True):

        # TODO: add computation of errors

        figs = self.plot_feature_errors(feature, signed_errors, normalized,
                                        bins, log, filename=filename + ".pdf",
                                        logtime=logtime)

        data = self.get_feature(feature, fmt, filename=filename + ".csv",
                                logtime=logtime)

        return data, figs

    def summary(self, fmt=None, signed_errors=True,
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
        data = self.get_all_features(fmt, "%s_predictions.csv" % filename,
                                     logtime)

        # save model parameters
        self.model.save_params(filename="%s_model_params.json" % filename,
                               logtime=logtime, logger=self.logger)

        return data, figs
