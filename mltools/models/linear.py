from sklearn import linear_model

from .model import Model


class LinearRegression(Model):

    def __init__(self, inputs=None, outputs=None, model_params=None, name=""):

        Model.__init__(self, inputs, outputs, model_params)

        self.name = name or 'LinearRegression {}'.format(hex(id(self)))

        self.model = self.create_model()

    def create_model(self):

        return linear_model.LinearRegression()
