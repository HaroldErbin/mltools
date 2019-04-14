"""
Define the structure of data

By default, the data is considered to have no channels and will be reshaped
by adding a `1` at the end, except for images.

The data structure can present the data according to 4 modes:
- merged: all objects are merged in the biggest possible tensor
- flat: all objects are merged in a single 1d array
- type: all objects of a type are merged in a single array
- col: by column name
"""


# TODO: when transforming, if there is a single input (or output), can pass
#   single instance (int, etc.) or vec
# TODO: for col mode, can also output list(d.values())
# TODO: use datatypes to specify which labels to group when using
#   multi-label encoding
# TODO: good format for tensorflow


# transformation modes
_MODES = ['col', 'type', 'merged', 'flat']

_TYPES = ['scalar']
# future: bool, int, vector, matrix, 3-tensor, text, image, video
#         different types of classification


class DataStructure:
    """
    Represent data

    The goal of this class is to convert data from one format to another,
    performing some automatic conversion depending on the type.

    It is necessary to be explicit about all shapes because the structure
    also provides methods to transform back to the original format. This is
    useful for predictions.
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
        3. if `features` is a dict, then the values
        4. the key in `datatypes` under which the feature appears

        Similarly, the form of the shapes will be determined according to
        the types, and the sizes of the different dimensions is chosen
        according to (note that there is no default choice):
        1. inference from the argument `infer`
        2. if `features` is a dict, then the values if they are tuples
        3. the values in `shapes`
        In case 1., the biggest shape appearing in a column will be used.
        """

        # pass data through scikit pipeline before fit or transformation
        # implemented by the class
        # TODO: consider more general method/function
        self.pipeline = pipeline
        if self.pipeline is not None:
            raise NotImplementedError

        # if None, take all columns from dataframe
        if features is None:
            self.initial_features = []
        elif isinstance(features, dict):
            self.initial_features = list(features.values())
        else:
            self.initial_features = list(features)

        # TODO: update list of outputs
        # example: after label binazer, or computing moments for vector
        # (need to find how to get all names)
        # use list() to copy
        self.features = list(self.initial_features)

        self.feature_types = {}
        self.feature_shapes = {}

        # datatypes: scalar, vector, matrix, label, etc.
        # if not given, infer (at the end, all inputs should be given a type)
        # infer shape for tensors, infer shape from data (use biggest shape)
        # (simpler to use than previous methods)
        # force_shape : force a specific shape for some object

        # scaling: None, minmax, std
        if scaling is not None:
            raise NotImplementedError

        # default mode
        self.mode = mode

    def fit(self, X, y):

        pass

    def transform(self, X, mode=None):

        mode = None or self.mode

        pass

    def inverse_transform(self, y, mode=None):

        mode = None or self.mode

        pass
