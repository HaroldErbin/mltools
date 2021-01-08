from sklearn import linear_model

from .model import Model


class LinearRegression(Model):
    """
    Linear regression.

    Regularization can be given in `model_params` as `l1` and `l2` or as `alpha` and `rho`. The
    appropriate class is chosen depending on which regularization is used. The default choice for
    `alpha` only is Lasso regression.
    """

    @property
    def model_name(self):
        return "Linear regression"

    def create_model(self):

        # TODO: add classification task

        params = self.model_params.copy()

        l1 = params.pop("l1", None)
        l2 = params.pop("l2", None)

        alpha = params.pop("alpha", None)
        rho = params.pop("rho", None)

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

            return linear_model.ElasticNet(alpha=alpha, l1_ratio=rho,
                                           **params)
        elif l2 is not None and l1 is None:
            return linear_model.Ridge(alpha=l2, **params)
        elif ((l1 is not None and l2 is None)
                or (alpha is not None and rho is None)):

            if alpha is None:
                alpha = l1

            return linear_model.Lasso(alpha=alpha, **params)
        else:
            params.pop("max_iter", None)
            return linear_model.LinearRegression(**params)

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
