# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor

from .model import Model


class SVM(Model):

    @property
    def model_name(self):
        return f"SVM ({self.method})"

    def create_model(self):

        # default arguments
        if "kernel" not in self.model_params:
            self.model_params["kernel"] = "linear"
        if self.method is None:
            self.method = "regression"

        params = self.model_params.copy()

        kernel = params["kernel"]

        if kernel == "linear":
            del params["kernel"]

        if self.method == "classification":
            if kernel == "linear":
                return svm.LinearSVC(**params)
            else:
                return svm.SVC(**params)

            # TODO: check if MultiOutputClassification is needed

        elif self.method == "regression":
            if kernel == "linear":
                model = svm.LinearSVR(**params)
            else:
                model = svm.SVR(**params)

            if len(self.outputs) > 1:
                return MultiOutputRegressor(model)
            else:
                return model
        else:
            raise ValueError("`method` can only be regression or classification.")
