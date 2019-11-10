from sklearn import ensemble

from .model import Model


class RandomForest(Model):

    def __init__(self, inputs=None, outputs=None, model_params=None, name="",
                 method="clf"):

        Model.__init__(self, inputs, outputs, model_params)

        self.method = method

        if method in ("clf", "classif", "classification"):
            self.method = "classification"
            self.model = ensemble.RandomForestClassifier()
        elif method in ("reg", "regression"):
            self.method = "regression"
            self.model = ensemble.RandomForestRegressor()
        else:
            raise ValueError("Method `%s` not permitted." % method)

        default_name = "Random forest ({}) {}".format(self.method,
                                                      hex(id(self)))
        self.name = name or default_name

    def predict(self, X):

        # TODO: fix split_array to avoid this

        if self.inputs is not None:
            X = self.inputs(X, mode='flat')

        y = self.model.predict(X).reshape(-1, 1)

        if self.outputs is not None:
            y = self.outputs.inverse_transform(y)

        return y
