"""
Define the metrics used for the different feature types.
"""

import numpy as np
import pandas as pd


# TODO: display metric names: upper case, not underscore, same length string


class TensorEval:

    metrics = ["mae", "rmse", "l1_mae", "l2_mae"]
    scalar_metrics = ["mae", "rmse"]

    @staticmethod
    def evaluate(y_pred, y_true, method="rmse", **kwargs):

        if method == "errors":
            return TensorEval.errors(y_true, y_pred)
        elif method == "norm_errors":
            return TensorEval.norm_errors(y_true, y_pred, **kwargs)
        elif method == "l1_mae":
            return TensorEval.norm_mae(y_true, y_pred, norm=1)
        elif method == "l2_mae":
            return TensorEval.norm_mae(y_true, y_pred, norm=2)
        elif method == "rmse":
            return TensorEval.rmse(y_true, y_pred)
        elif method == "mae":
            return TensorEval.mae(y_true, y_pred)
        elif callable(method):
            return method(y_pred, y_true)
        else:
            return None

    @staticmethod
    def get_metrics(y_pred, y_true, metrics=None):
        if metrics is None:
            if TensorEval.is_scalar(y_pred):
                metrics = TensorEval.scalar_metrics
            else:
                metrics = TensorEval.metrics

        return {m: TensorEval.evaluate(y_pred, y_true, method=m)
                for m in metrics}

    @staticmethod
    def is_scalar(tensor):
        return len(np.shape(tensor)) <= 1

    @staticmethod
    def norm(tensor, norm=1):
        """
        Compute the norm of a tensor instance-wise.

        This computes the norm for each instance using a given function.
        The result has the same shape as the input.

        The argument `norm` can be a number, in which case the L-norm of that
        order is computed for each tensor. Otherwise, it should be a function
        which computes the appropriate norm. Note that the L-norm takes the
        absolute value.
        """

        if isinstance(norm, (int, str)):
            return np.linalg.norm(tensor.reshape(len(tensor), -1),
                                  ord=norm, axis=1)
        elif callable(norm) is True:
            return norm(tensor)
        else:
            raise ValueError("Cannot compute a norm with object `{}`."
                             .format(norm))

    @staticmethod
    def norm_errors(y_true, y_pred, norm=1, relative=False):
        """
        Compute the errors between the tensor norms.

        For each instance, this computes the error in the tensor norm.
        The output is a vector of length `len(y_pred)`.
        """

        errors = TensorEval.norm(TensorEval.errors(y_true, y_pred),
                                 norm=norm)

        if relative is True:
            return errors / TensorEval.norm(y_true, norm)
        else:
            return errors

    @staticmethod
    def norm_mae(y_true, y_pred, norm=2):
        """
        Compute the mean absolute error of the tensor norms.
        """

        return np.mean(TensorEval.norm_errors(y_true, y_pred, norm=norm))

    @staticmethod
    def errors(y_true, y_pred, signed=True, relative=False):
        """
        Compute the errors between each component of the tensors.

        This computes the errors component-wise for each instance.
        The result has the same shape as the input.
        """

        errors = np.subtract(y_true, y_pred)

        if relative is True:
            errors = np.divide(errors, np.abs(y_true))

        if signed is False:
            errors = np.abs(errors)

        return errors

    @staticmethod
    def rmse(y_true, y_pred):
        n = np.size(y_pred)

        errors = TensorEval.norm_errors(y_true, y_pred, norm=2)

        return np.sqrt(np.sum(errors) / n)

    @staticmethod
    def mae(y_true, y_pred):
        n = np.size(y_pred)

        return np.sum(TensorEval.norm_errors(y_true, y_pred, norm=1)) / n

    # def max_error(y_true, y_pred, signed=False, relative=False):
    #     return np.max(TensorEval.errors(y_true, y_pred, signed, relative))

    # def min_error(y_true, y_pred, signed=False, relative=False):
    #     return np.min(TensorEval.errors(y_true, y_pred, signed, relative))

    # def median_error(y_true, y_pred, signed=False, relative=False):
    #     return np.median(TensorEval.errors(y_true, y_pred, signed, relative))

    # def percentile_error(y_true, y_pred, q, signed=False, relative=False):
    #     return np.percentile(TensorEval.errors(y_true, y_pred, signed,
    #                                            relative), q)


class BinaryEvaluation:

    pass
