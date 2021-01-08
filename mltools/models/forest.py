from sklearn import ensemble
from sklearn.multioutput import MultiOutputRegressor

from .model import Model


# TODO: fix warning "DataConversionWarning: A column-vector y was passed when
#       a 1d array was expected. Please change the shape of y to (n_samples,),
#       for example using ravel()."


class RandomForest(Model):

    @property
    def model_name(self):
        return f"Random forest ({self.method})"

    def create_model(self):
        # default arguments
        if self.method is None:
            self.method = "regression"

        if self.method == "classification":
            return ensemble.RandomForestClassifier(**self.model_params)
        elif self.method == "regression":
            return ensemble.RandomForestRegressor(**self.model_params)
