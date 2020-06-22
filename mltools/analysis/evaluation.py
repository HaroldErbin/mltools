"""
Define the metrics used for the different feature types.
"""

import numpy as np
import pandas as pd


# TODO: define metric as class with: name, pretty_name, static method to
#       compute the metric
#       can define several names
#       format name thanks to function according to arguments if any
#       define as separate class or member of TensorEval?
#       first option may be better, but create one module per data type
#       class TensorEval allows to call any class easily (make list of
#       metrics, lookup by names)
#       can initialize class to set default arguments (but can be used
#       as static)

# TODO: allow passing argument to TensorEval.evaluate
#       to do this: `method` can be text, callable or pair
#       (text/callable, kwargs)


class TensorEval:

    tensor_metrics = ["rmse", "mae", "l1_mae", "l2_mae"]
    metrics = ["rmse", "mae"]

    default_metric = "rmse"

    _metric_names = {"mae": "MAE", "rmse": "RMSE", "l1_mae": "L1 MAE",
                     "l2_mae": "L2 MAE"}

    @staticmethod
    def evaluate(y_pred, y_true, method="rmse", **kwargs):

        if method == "errors":
            return TensorEval.errors(y_pred, y_true)
        elif method == "norm_errors":
            return TensorEval.norm_errors(y_pred, y_true, **kwargs)
        elif method == "l1_mae":
            return TensorEval.norm_mae(y_pred, y_true, norm=1)
        elif method == "l2_mae":
            return TensorEval.norm_mae(y_pred, y_true, norm=2)
        elif method == "rmse":
            return TensorEval.rmse(y_pred, y_true)
        elif method == "mae":
            return TensorEval.mae(y_pred, y_true)
        elif method == "accuracy":
            return TensorEval.accuracy(y_pred, y_true)
        elif callable(method):
            # callable must also be a class
            return method(y_pred, y_true)
        else:
            return TensorEval.evaluate(y_pred, y_true,
                                       method=TensorEval.default_metric,
                                       **kwargs)

    @staticmethod
    def eval_metrics(y_pred, y_true, metrics=None):
        if metrics is None:
            if TensorEval.is_scalar(y_pred):
                metrics = TensorEval.metrics
            else:
                metrics = TensorEval.tensor_metrics

        return {m: TensorEval.evaluate(y_pred, y_true, method=m)
                for m in metrics}

    @staticmethod
    def is_scalar(tensor):
        return len(np.shape(tensor)) <= 1

    @staticmethod
    def vector_norm(tensor, norm=1):
        """
        Compute the norm of a tensor instance-wise.

        This computes the norm for each instance using a given function.
        The result is a vector of the same length as the input first dimension.

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
    def norm(tensor, norm=1):
        """
        Compute the norm of a tensor.

        This computes the norm for each instance using a given function.
        The result is a scalar.

        The argument `norm` can be a number, in which case the L-norm of that
        order is computed for each tensor. Otherwise, it should be a function
        which computes the appropriate norm. Note that the L-norm takes the
        absolute value.
        """

        if isinstance(norm, (int, str)):
            return np.linalg.norm(tensor.reshape(-1), ord=norm)
        elif callable(norm) is True:
            return norm(tensor)
        else:
            raise ValueError("Cannot compute a norm with object `{}`."
                             .format(norm))

    @staticmethod
    def norm_errors(y_pred, y_true, norm=1, relative=False):
        """
        Compute the errors between the tensor norms.

        For each instance, this computes the error in the tensor norm.
        The output is a vector of length `len(y_pred)`.
        """

        errors = TensorEval.norm(TensorEval.errors(y_pred, y_true),
                                 norm=norm)

        if relative is True:
            return errors / TensorEval.vector_norm(y_true, norm)
        else:
            return errors

    @staticmethod
    def norm_mae(y_pred, y_true, norm=2):
        """
        Compute the mean absolute error of the tensor norms.
        """

        return np.mean(TensorEval.norm_errors(y_pred, y_true, norm=norm))

    @staticmethod
    def errors(y_pred, y_true, signed=True, relative=False):
        """
        Compute the errors between each component of the tensors.

        This computes the errors component-wise for each instance.
        The result has the same shape as the input.
        """

        errors = np.subtract(y_pred, y_true)

        if relative is True:
            errors = np.divide(errors, np.abs(y_true))

        if signed is False:
            errors = np.abs(errors)

        return errors

    @staticmethod
    def rmse(y_pred, y_true):
        n = np.sqrt(np.size(y_pred))

        return TensorEval.norm(TensorEval.errors(y_pred, y_true), norm=2) / n

    @staticmethod
    def mae(y_pred, y_true):
        n = np.size(y_pred)

        return TensorEval.norm(TensorEval.errors(y_pred, y_true), norm=1) / n

    @staticmethod
    def accuracy(y_pred, y_true, tol=0.):
        """
        Accuracy of the predictions

        The accuracy is defined by the number of predictions which exactly
        match the real values up to the precision `tol`. For tensors of
        integers, set `tol = 0`.

        Since this metric is very specific, it is not used by default.
        """

        # TODO: generalize to tensors

        n = np.size(y_pred)

        return np.sum(np.abs(TensorEval.errors(y_pred, y_true)) <= tol) / n


    # def max_error(y_pred, y_true, signed=False, relative=False):
    #     return np.max(TensorEval.errors(y_pred, y_true, signed, relative))

    # def min_error(y_pred, y_true, signed=False, relative=False):
    #     return np.min(TensorEval.errors(y_pred, y_true, signed, relative))

    # def median_error(y_pred, y_true, signed=False, relative=False):
    #     return np.median(TensorEval.errors(y_pred, y_true, signed, relative))

    # def percentile_error(y_pred, y_true, q, signed=False, relative=False):
    #     return np.percentile(TensorEval.errors(y_pred, y_true, signed,
    #                                            relative), q)


class BinaryEvaluation:

    pass
