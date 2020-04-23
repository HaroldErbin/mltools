from sklearn import ensemble
from sklearn.multioutput import MultiOutputRegressor

from .model import Model


class RandomForest(Model):

    def __init__(self, inputs=None, outputs=None, model_params=None, n=1,
                 name="", method="clf"):

        Model.__init__(self, inputs, outputs, model_params, n, name)

        if method in ("clf", "classification"):
            self.method = "classification"
        elif method in ("reg", "regression"):
            self.method = "regression"
        else:
            raise ValueError("Method `%s` not permitted." % method)

        if n > 1:
            self.model = [self.create_model() for n in range(self.n)]
        else:
            self.model = self.create_model()

        self.model_name = "Random forest ({})".format(self.method)

    def create_model(self):
        if self.method == "classification":
            return ensemble.RandomForestClassifier(**self.model_params)
        elif self.method == "regression":
            return ensemble.RandomForestRegressor(**self.model_params)
