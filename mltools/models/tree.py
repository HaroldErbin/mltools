from sklearn import tree

from .model import Model


class DecistionTree(Model):

    def __init__(self, inputs=None, outputs=None, model_params=None, name="",
                 method="clf"):

        Model.__init__(self, inputs, outputs, model_params)

        self.method = method

        if method in ("clf", "classification"):
            self.method = "classification"
        elif method in ("reg", "regression"):
            self.method = "regression"
        else:
            raise ValueError("Method `%s` not permitted." % method)

        self.model = self.create_model()

        default_name = "Decision tree ({}) {}".format(self.method,
                                                      hex(id(self)))
        self.name = name or default_name

    def create_model(self):
        if self.method == "classification":
            return tree.DecisionTreeClassifier(**self.model_params)
        elif self.method == "regression":
            return tree.DecisionTreeRegressor(**self.model_params)

    def predict(self, X):

        # TODO: fix split_array to avoid this

        if self.inputs is not None:
            X = self.inputs(X, mode='flat')

        y = self.model.predict(X).reshape(-1, 1)

        if self.outputs is not None:
            y = self.outputs.inverse_transform(y)

        return y
