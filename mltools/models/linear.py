from sklearn import linear_model

from .model import Model


class LinearRegression(Model):

    def __init__(self, inputs=None, outputs=None, model_params=None, n=1,
                 method="reg", name=""):
        """
        Linear regression.

        Regularization can be given in `model_params` as `l1` and `l2` or
        as `alpha` and `rho`. The appropriate class is chosen depending
        on which regularization is used. The default choice for `alpha` only
        is Lasso regression.
        """

        # TODO: add classification task

        Model.__init__(self, inputs, outputs, model_params, n, method, name)

        if n > 1:
            self.model = [self.create_model() for n in range(self.n)]
        else:
            self.model = self.create_model()

        self.model_name = "LinearRegression"

    def create_model(self):

        l1 = self.model_params.get("l1", None)
        l2 = self.model_params.get("l2", None)

        alpha = self.model_params.get("alpha", None)
        rho = self.model_params.get("rho", None)

        if ((l1 is not None or l2 is not None)
                and (alpha is not None or rho is not None)):
            raise ValueError("Cannot set `l1` and/or `l2` together with "
                             "`alpha` and/or `rho`.")

        # depending on model, the sample loss may be divided by 2n

        if ((l1 is not None and l2 is not None)
                or (alpha is not None and rho is not None)):

            if alpha is None and rho is None:
                alpha = l1 + l2
                rho = 0 if alpha == 0 else l1 / alpha

            return linear_model.ElasticNet(alpha=alpha, l1_ratio=rho)
        elif l2 is not None and l1 is None:
            return linear_model.Ridge(alpha=l2)
        elif ((l1 is not None and l2 is None)
                or (alpha is not None and rho is None)):

            if alpha is None:
                alpha = l1

            return linear_model.Lasso(alpha=alpha)
        else:
            return linear_model.LinearRegression()

    def model_representation(self):
        """
        Return a representation of the model.

        This returns the formula for the linear regression.
        """

        slopes = self.inputs.inverse_transform(self.model.coef_)
        intercepts = self.inputs.inverse_transform(self.model.intercept_)

        # round numbers, write formula

        # when there are tensors, it is hard to write a nice text formula
        # limit to scalar cases and with categories

        text = "y = ..."

        raise NotImplementedError
