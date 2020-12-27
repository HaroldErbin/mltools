from sklearn import tree

from .model import Model


# TODO: implement export_graphviz, export_text


class DecistionTree(Model):

    def __init__(self, inputs=None, outputs=None, model_params=None, model_fn=None, n=1,
                 method="reg", name=""):

        Model.__init__(self, inputs, outputs, model_params, model_fn, n, method, name)

        if n > 1:
            self.model = [self.create_model() for n in range(self.n)]
        else:
            self.model = self.create_model()

        self.model_name = "Decision tree ({})".format(self.method)

    def create_model(self):
        if self.method == "classification":
            return tree.DecisionTreeClassifier(**self.model_params)
        elif self.method == "regression":
            return tree.DecisionTreeRegressor(**self.model_params)
