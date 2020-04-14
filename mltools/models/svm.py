# -*- coding: utf-8 -*-

from sklearn import svm

from .model import Model


class SVM(Model):

    def __init__(self, inputs=None, outputs=None, model_params=None, name="",
                 method="clf"):

        Model.__init__(self, inputs, outputs, model_params)

        # default arguments
        if "kernel" not in self.model_params:
            self.model_params["kernel"] = "linear"

        if method in ("clf", "classification"):
            self.method = "classification"
        elif method in ("reg", "regression"):
            self.method = "regression"
        else:
            raise ValueError("Method `%s` not permitted." % method)

        self.model = self.create_model()

        self.model_name = "SVM ({})".format(self.method)
        self.name = name

    def create_model(self):
        if self.method == "classification":
            if self.model_params["kernel"] == "linear":
                return svm.LinearSVC()
            else:
                return svm.SVC(**self.model_params)
        elif self.method == "regression":
            if self.model_params["kernel"] == "linear":
                return svm.LinearSVR()
            else:
                return svm.SVR(**self.model_params)

    def fit(self, X, y=None):

        # TODO: SVR expects 1d array
        # update general model and/or datastructure to allow for it

        if y is None:
            y = X

        if self.inputs is not None:
            X = self.inputs(X, mode='flat')
        if self.outputs is not None:
            y = self.outputs(y, mode='flat')

        return self.model.fit(X, y.reshape(-1))

    def predict(self, X):

        if self.inputs is not None:
            X = self.inputs(X, mode='flat')

        y = self.model.predict(X).reshape(-1, 1)

        if self.outputs is not None:
            y = self.outputs.inverse_transform(y)

        return y
