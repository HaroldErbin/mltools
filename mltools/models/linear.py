from sklearn import linear_model

from .model import Model


# TODO: add L1 and L2 regularization
# TODO: add multi-task


class LinearRegression(Model):

    def __init__(self, inputs=None, outputs=None, model_params=None, n=1,
                 name=""):

        Model.__init__(self, inputs, outputs, model_params, n, name)

        if n > 1:
            self.model = [self.create_model() for n in range(self.n)]
        else:
            self.model = self.create_model()

        self.model_name = "LinearRegression"

    def create_model(self):

        return linear_model.LinearRegression()
