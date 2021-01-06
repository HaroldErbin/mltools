"""
Define metrics used in evaluation.
"""

import numpy as np

import sklearn as sk

from mltools.analysis.logger import Logger
from mltools.data import datatools as dt


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


class MetricLookup(object):

    # singleton pattern
    _instance = None

    _metrics = []

    logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricLookup, cls).__new__(cls)

        return cls._instance

    @staticmethod
    def metrics():
        return {metric.key: metric for metric in MetricLookup._metrics}

    @staticmethod
    def all_names():
        return {name: metric for metric in MetricLookup._metrics for name in metric.names.values()}

    @staticmethod
    def get_metric(metric):
        return MetricLookup.all_names()[metric]

    @staticmethod
    def valid_metric_for(x):
        x_type = dt.infer_types(x)
        return {name: metric for name, metric in MetricLookup.metrics().items()
                if x_type in metric.types and x_type not in metric.excluded_types}

    @staticmethod
    def evaluate(x, y=None, metric=None, n=1):
        """
        Evaluate the metric(s) on the data provided.

        If no metric is given, return all the compatible metrics found by looking up.
        The data can be a table of named data, in which case the function is applied to each
        column. In this case, metric can be used to specify which metric to use for which feature.

        `n` indicates how many different `y` (if given, else `x`) are given for statistics. This
        implies that the first dimension must be preserved, so `keep = 1`.
        """

        keep = 1 if n > 1 else 0

        if isinstance(x, dict):
            features = set(x.keys())

            if y is None:
                pass
            elif isinstance(y, dict):
                features &= set(y.keys())
            else:
                raise TypeError("Both or none of `x` and `y` must be a dict. "
                                f"Found `{type(x)}` and `{type(y)}`.")

            if isinstance(metric, dict):
                features &= set(metric.keys())

            if y is None:
                y = {f: None for f in features}

            if metric is None:
                metric = {f: None for f in features}

            return {f: MetricLookup.evaluate(x[f], y[f], n=n, metric=metric.get(f))
                    for f in features}

        if n > 1 and y is not None:
            x = np.array(n * [x])

        if y is None:
            y = 0

        if isinstance(metric, str):
            return MetricLookup.all_names()[metric](x, y, keep=keep)
        elif isinstance(metric, (list, tuple)):
            return {m: MetricLookup.all_names()[m](x, y, keep=keep) for m in metric
                    if m in MetricLookup.valid_metric_for(x)}
        elif metric is None:
            metrics = MetricLookup.valid_metric_for(x)
            return {name: m(x, y, keep=keep) for name, m in metrics.items()}
        else:
            raise TypeError(f"Cannot evaluate metric of type `{type(metric)}`.")


class Metric:
    """
    Describe information on a metric.

    Instances define a specific metrics, including its different names, how to compute it, which
    types of data it can be used with.

    The attribute `name_priority` defines the priority for using the different names depending on
    the context:
    - plot: legend, axis...
    - text: writing the name in a report or on the screen
    - class_name: name appearing when using `str` or `repr`
    - key: default key used when computing a dict of metrics
    """

    # define priority used for names
    name_priority = {"plot": ["plot", "latex", "upper_abbrev", "lower_abbrev"],
                     "text": ["upper_abbrev", "lower_abbrev"],
                     "class_name": ["upper_abbrev", "lower_abbrev"],
                     "key": ["lower_abbrev", "keras"]}

    def __init__(self, metric_fn, names=None, types=None, is_ratio=False):
        """
        Initialization of the metric instance.

        Element of `types` starting with a minus sign are excluded.
        For a list of available types, see `structure.py`.

        Possible names are:
        - lower_abbrev
        - upper_abbrev
        - lower_name
        - upper_name
        - keras
        - latex
        - plot

        Args:
            metric_fn (TYPE): function to compute the metric.
            names (TYPE, optional): dict of names used for the metric, the key giving the context.
            types (TYPE, optional): types for which the metric can be used (this is used only
                for automatic discovery, direct call will never be prevented).
            is_ratio (TYPE, optional): indicates if the metric is ratio (percentage).
        """

        # register metric instance for automatic discovery of metrics
        MetricLookup._metrics.append(self)

        self.names = names
        self.is_ratio = is_ratio

        self.types = [t for t in types if not t.startswith("-")]
        self.excluded_types = [t[1:] for t in types if t.startswith("-")]

        self.metric_fn = metric_fn

    def __str__(self):
        return self.priority_name("class_name")

    def __repr__(self):
        return f"<Metric: {str(self)}>"

    def __call__(self, x, y=0, keep=0, params=None, mode=None):
        if params is None:
            params = {}

        val = self.metric_fn(x, y, keep=keep, **params)

        if mode == "text":
            return self.print_value(val)
        else:
            return val

    def priority_name(self, context):
        names = [self.names.get(name) for name in self.name_priority[context]]

        # TODO: default case if only None returned

        return next(name for name in names if name is not None)

    @property
    def key(self):
        return self.priority_name("key")

    def print_value(self, value):
        logger = MetricLookup.logger or Logger

        if self.is_ratio:
            return logger.styles["print:percent"].format(value)
        else:
            return logger.styles["print:float"].format(value)


def count(x, keep=0):
    """
    Count the number of instances summed over in the norm.

    This function is useful to normalize when computing MAE, MSE, etc.
    """

    return np.prod(np.shape(x)[keep:])


def distance(x, y=0, keep=0, norm=1):
    """
    Compute the distance between two tensors using the given norm.

    The first `keep` dimensions are kept as they are. By default, `keep = 0` such that the result
    is a scalar. This can be useful when computing the distance instance-wise and/or for a bag.

    If `y` is not given, compute the norm of `x`.

    The argument `norm` can be a number, in which case the L-norm of that order is computed.
    Otherwise, it should be a function which computes the appropriate norm. Note that the L-norm
    takes the absolute value.
    """

    x = np.subtract(x, y)
    x = np.reshape(x, (*np.shape(x)[:keep], -1))

    if isinstance(norm, int):
        return np.linalg.norm(x, ord=norm, axis=-1)
    elif callable(norm) is True:
        return norm(x)
    else:
        raise ValueError(f"Cannot compute a norm with object `{norm}`.")


def accuracy(x, y=0, keep=0, tol=0.):
    """
    Accuracy of y matching x.

    The accuracy is defined by the number of samples in x and y which match up to the precision
    `tol`. For tensors of integers, set `tol = 0`.
    """

    # TODO: generalize to tensors

    x = np.subtract(x, y)
    x = np.reshape(x, (*np.shape(x)[:keep], -1))

    n = np.size(x, axis=-1)

    return np.sum(np.abs(x) <= tol, axis=-1) / n


mae = Metric(lambda x, y, keep=0: distance(x, y, keep, norm=1) / count(x, keep),
             names={"lower_abbrev": "mae", "upper_abbrev": "MAE",
                    "lower_name": "mean absolute error", "keras": "mean_absolute_error"},
             types=["scalar", "tensor"])
rmse = Metric(lambda x, y, keep=0: distance(x, y, keep, norm=2) / np.sqrt(count(x, keep)),
              names={"lower_abbrev": "rmse", "upper_abbrev": "RMSE",
                     "lower_name": "root mean squared error"},
              types=["scalar", "tensor"])
# r2 = Metric(sk.metrics.r2_score,
#             names={"lower_abbrev": "r2", "upper_abbrev": "R2", "latex": "$R^2$",
#                    "lower_name": "coefficient of determination"},
#             types=[])

acc = Metric(lambda x, y: accuracy(x, y),
             names={"lower_abbrev": "acc",
                    "lower_name": "accuracy"},
             types=[], is_ratio=True)
