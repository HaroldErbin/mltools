"""
Define the structure of data

The data is organized in features. Each feature has a specific type (scalar,
tensor, label...).

A tensor type can be represented by its shape. For example, providing the
type `(4, 3, 2)` means that the data is a 3-tensor. The following alias
are used: scalar (0d tensor), vector (1d tensor), matrix (2d tensor). By
default, tensors are considered to have no channels and will be reshaped by
adding a `1` at the end.

The data structure can present the data according to 4 modes:
- merged: all objects are merged in the biggest possible tensor
- flat: all objects are merged in a single 1d array
- type: all objects of a type are merged in a single array
- col: by column name
"""


import numpy as np
import pandas as pd


# TODO: when transforming, if there is a single feature, can pass
#   single instance (int, etc.) or vec
# TODO: for col mode, can also output list(d.values())
# TODO: use datatypes to specify which labels to group when using
#   multi-label encoding, instead of all possibilities
# TODO: good format for tensorflow
# TODO: for muli-label, write as tuple of features

# transformation modes
_MODES = ['col', 'type', 'merged', 'flat']

# supported types
_TYPES = ['integer', 'scalar']
# future: bool, int, vector, matrix, 3-tensor, text, image, video
#         different types of classification

_TENSOR_ALIAS = {'tensor_0d': 'scalar', 'tensor_1d': 'vector',
                 'tensor_2d': 'matrix'}


# TODO: check how to ensure that all tensors have a shape


class DataStructure:
    """
    Represent data with conversion

    The goal of this class is to convert data from one format to another,
    performing some automatic conversion depending on the type.

    It is necessary to be explicit about all types (for example tensor shapes)
    because the structure also provides methods to transform back to the
    original format. This is useful for predictions.
    """

    def __init__(self, features=None, datatypes=None, shapes=None,
                 pipeline=None, scaling=None, infer=None,
                 mode='flat'):
        """
        List of features to be transformed. The types can be inferred from
        a dataframe or they can be forced.

        `features` can be a list of features or a dict mapping features
        to types. `datatypes` is a dictionary mapping types to features
        (hence the opposite order). `shapes` is a dict of features mapping
        to a shape used to enforce a specific shape.

        The class will determine the types according according to the
        following steps (by increasing priority):
        1. default to scalar
        2. inference from the argument `infer` (usually a dataframe)
        3. if `features` is a dict, then the values indicated
        4. the key in `datatypes` under which the feature appears

        Similarly, the form of the shapes will be determined according to
        the types, and the sizes of the different dimensions is chosen
        according to (note that there is no default choice):
        1. inference from the argument `infer`
        2. if `features` is a dict, then the values if they are tuples
        3. the values in `shapes`
        In case 1., the biggest shape appearing in a column will be used.

        :param features: sequence of features or dict of features with datatype
        :type features: any sequence, `dict`
        :param infer: data with named features to infer the structure
        :type infer: `pandas.DataFrame`, `dict`
        """

        if infer is not None:
            if isinstance(infer, dict):
                infer_cols = list(infer.keys())
            elif isinstance(infer, pd.DataFrame):
                infer_cols = list(infer.columns)
            else:
                raise TypeError("Inference works only from an object with "
                                "must have named features. "
                                "Supported types are: dict, dataframe.")
        else:
            infer_cols = None

        self.initial_features = []

        if features is None and datatypes is None:
            self.initial_features += list(infer_cols)
        else:
            if isinstance(features, dict):
                self.initial_features += list(features.keys())
            elif isinstance(features, (list, tuple)):
                self.initial_features += list(features)

            if isinstance(datatypes, dict):
                self.initial_features += list(features.keys())

        # TODO: update list of outputs
        # example: after label binazer, or computing moments for vector
        # (need to find how to get all names)
        # use list() to copy
        self.features = list(self.initial_features)

        # store all types, irrespective of shape
        self.feature_types = {}
        self.feature_shapes = {}

        if isinstance(features, dict):
            for f, v in features.items():
                # if the value is a shape, then the type must be a tensor
                if isinstance(v, (tuple, list, np.ndarray)):
                    self.feature_types[f] = 'tensor_{}d'.format(len(v))
                    self.feature_shapes[f] = v
                elif isinstance(v, str):
                    self.feature_types[f] = v
                else:
                    raise ValueError("`{}` type not usable.".format(type(v)))

        # get missing type by looking to the dataset `infer` if defined
        if infer_cols is not None:
            for f in self.features:
                # first element of the data for the feature
                first = infer[f][0]

                if isinstance(first, (np.ndarray, list, tuple)):
                    shape = np.shape(first)
                    # TODO: find maximal shape in series

                # for dataframe only: infer[f].dtype
                if f not in self.feature_types:
                    if np.issubdtype(type(first), np.integer):
                        self.feature_types[f] = 'integer'
                    elif np.issubdtype(type(first), np.floating):
                        self.feature_types[f] = 'scalar'
                    elif isinstance(first, (np.ndarray, list, tuple)):
                        self.feature_types[f] = 'tensor_{}d'.format(len(shape))
                        self.feature_shapes[f] = np.shape(first)
                    else:
                        raise TypeError("Type `{}` is not supported."
                                        .format(type(first)))

                # for tensor types defined by features but without the shape
                if (f not in self.feature_shapes
                        and isinstance(first, (np.ndarray, list, tuple))):
                    self.feature_shapes[f] = np.shape(first)

        # replace tensor types for low dimensions
        for f, t in self.feature_types.items():
            if t in _TENSOR_ALIAS:
                self.feature_types[f] = _TENSOR_ALIAS[t]

        # default type to scalar
        missing_types = {f: "scalar" for f in self.features
                         if f not in self.feature_types}
        self.feature_types.update(missing_types)

        for f in self.features:
            if f not in self.feature_types:
                raise ValueError("The feature `{}` has no type.".format(f))

        for f in self.feature_types:
            if (f in ('vector', 'matrix') or f.startswith('tensor')
                    and f not in self.feature_shapes):
                raise ValueError("The feature `{}` is a tensor without shape."
                                 .format(f))

        # default mode
        self.mode = mode

        # scaling: None, minmax, std
        if scaling is not None:
            raise NotImplementedError

        # pass data through scikit pipeline before fit or transformation
        # implemented by the class
        # TODO: consider more general method/function
        self.pipeline = pipeline

        if self.pipeline is not None:
            raise NotImplementedError

    def __repr__(self):
        return "<DataStructure: {}>".format(list(self.feature_types.keys()))

    def fit(self, X, y):

        pass

    def transform(self, X, mode=None):

        mode = None or self.mode

        pass

    def inverse_transform(self, y, mode=None):

        mode = None or self.mode

        pass
