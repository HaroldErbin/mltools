from sklearn import tree

from .model import Model


# TODO: implement export_graphviz, export_text


class DecistionTree(Model):

    @property
    def model_name(self):
        return f"Decision tree ({self.method})"

    def create_model(self):
        # default arguments
        if self.method is None:
            self.method = "regression"

        if self.method == "classification":
            return tree.DecisionTreeClassifier(**self.model_params)
        elif self.method == "regression":
            return tree.DecisionTreeRegressor(**self.model_params)
