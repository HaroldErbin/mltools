# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor

from .model import Model


class SVM(Model):

    def __init__(self, inputs=None, outputs=None, model_params=None, n=1,
                 method="reg", name=""):

        Model.__init__(self, inputs, outputs, model_params, method, n, name)

        # default arguments
        if "kernel" not in self.model_params:
            self.model_params["kernel"] = "linear"

        if n > 1:
            self.model = [self.create_model() for n in range(self.n)]
        else:
            self.model = self.create_model()

        self.model_name = "SVM ({})".format(self.method)

    def create_model(self):
        params = self.model_params.copy()

        if self.model_params["kernel"] == "linear":
            del params["kernel"]

        if self.method == "classification":
            if self.model_params["kernel"] == "linear":
                return svm.LinearSVC(**params)
            else:
                return svm.SVC(**params)

            # TODO: check if MultiOutputClassification is needed

        elif self.method == "regression":
            if self.model_params["kernel"] == "linear":
                model = svm.LinearSVR(**params)
            else:
                model = svm.SVR(**params)

            if len(self.outputs) > 1:
                return MultiOutputRegressor(model)
            else:
                return model
