# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


class Predictions:

    def __init__(self, inputs, outputs, model, X, y_pred, y_true=None,
                 y_std=None, categories=None, integers=None, decision_fn=None,
                 fmt='dict'):
        """
        Inits Predictions class.

        The goal of this class is to first convert the ML output to meaningful
        results. Then, to compare with the real data if they are known.

        `categories` and `integers` are dict mapping features to decision
        functions (to be applied to each sample). They can also be given as
        a list, in which case the default decision is applied:
        - integers: round
        - categories: 1 if p > 0.5 else 0 (or max p is multi-class)

        The `decision_fn` is applied after everything else.

        Format indicates if data is stored as a dict or as a dataframe.
        """

        self.inputs = inputs
        self.outputs = outputs
        self.model = model

        self.X = X
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_std = y_std

        if categories is None:
            self.categories = {}
        else:
            self.categories = categories

        if integers is None:
            self.integers = {}
        else:
            self.integers = integers

        self.decision_fn = decision_fn

        self.fmt = fmt

        # get predictions

        if decision_fn is not None:
            decision_fn(self.y_pred)

        # compute errors

        # TODO: how to handle tensors?
        #   first convert with datastructure to dict? (to work on array)
        # add other measures of errors for vectord (l-norm)

        if self.y_true is not None:
            self.errors = self._compute_errors()
            self.rel_errors = self._compute_relative_errors()

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
            errors[k] = (t - p) / np.abs(p)

        return errors

    def see_feature(self, feature):
        """
        Summarize feature results in one dataframe.

        Return a dataframe with the following columns: real, pred, error, rel
        for a given feature.
        """

        if self.y_true is None:
            return self.y_pred[feature]

        dic = {"id": self.y_pred["id"],
               feature + "_true": self.y_true[feature],
               feature + "_pred": self.y_pred[feature],
               feature + "_err": self.errors[feature],
               feature + "_rel": self.rel_errors[feature]}

        if self.fmt == "dataframe":
            return pd.DataFrame(dic)
        else:
            return dic

    def __getitem__(self, key):
        return self.see_feature(key)

    def save_predictions(self, filename=""):

        # TODO: csv, json (+gz)
        # sort columns like in see_feature
        pass
