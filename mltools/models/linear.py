from sklearn import linear_model

from .model import Model


class LinearRegression(Model):

    def __init__(self, inputs=None, outputs=None, name=""):

        Model.__init__(self, inputs, outputs)

        self.name = name or 'LinearRegression {}'.format(hex(id(self)))

        self.model = linear_model.LinearRegression()
