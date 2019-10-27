"""
Tools for manipulating data

Operations offered in this module allow to manipulate data with tensors
of different shapes. When such tensors must be merged (for example to an
array), the functions first pad each tensor such its shape is equal to the
embedding shape, which is the smallest shape in which all tensors can fit
*without flattening*.

Some functions flatten the output. If the input is made of tensors of
different shapes, the output is in general bigger than the one obtained
by first flattening, and then applying the function. The reason is that
flattening before embedding loses the information about the different tensor
directions.
"""

import numpy as np
import pandas as pd


_LOW_TENSORS = {'tensor_0d': 'scalar', 'tensor_1d': 'vector',
                'tensor_2d': 'matrix'}


def tensor_name(dim, channels=False):
    """
    Tensor name from its dimension.

    For `dim` = 0, 1, 2 return the alias (scalar, vector, matrix).

    :param dim: tensor dimension
    :type dim: int
    :param channels: true if channels must be excluded from counting
    :type channels: bool

    :return: tensor name
    :rtype: str
    """

    if channels is True:
        dim -= 1

    name = "tensor_{}d".format(dim)

    if name in _LOW_TENSORS:
        return _LOW_TENSORS[name]
    else:
        return name


def tensor_dim(shape, channels=False):
    """
    Return the dimension of an object from its shape.

    :param shape: tensor shape
    :type shape: tuple(int)
    :param channels: true if channels must be excluded from counting
    :type channels: bool

    :return: tensor dimension
    :rtype: int
    """

    if channels is True:
        return len(shape) - 1
    else:
        return len(shape)


def data_shapes(data):
    """
    Compute all shapes appearing in the data.

    If the data is a sequence (list, series, array), return the list of all
    shapes. If it is a table, return a dict of the shapes for each column.
    Note that sequences are treated as a set of samples, which means that the
    shape excludes the first dimension.

    :param data: data for which to compute the shapes
    :type data: list, series, array, dict, dataframe

    :return: shapes appearing in the sequence or in each column
    :rtype: list(tuple) or dict(str, list(tuple))
    """

    if isinstance(data, np.ndarray):
        return [np.shape(data)[1:]]
    elif isinstance(data, (list, tuple)):
        return sorted(list(set(map(lambda x: np.shape(x), data))))
    elif isinstance(data, pd.Series):
        return sorted(list(data.apply(lambda x: np.shape(x)).unique()))
    elif isinstance(data, dict):
        return {c: data_shapes(d) for c, d in data.items()}
    elif isinstance(data, pd.DataFrame):
        return {c: data_shapes(data[c]) for c in data.columns}
    else:
        raise TypeError("Data type `{}` is not supported.".format(type(data)))


def embedding_shape(data, shapelist=False):
    """
    Compute the shape in which all shapes in the data can be embedded.

    If the data is a table, compute the common shape for all columns.

    This function can be used to find the shape which allows to embed all the
    samples in a single array. It is given by the maximal value of
    `numpy.shape` along each dimension.

    If `shapelist` is true, then the function assumes that the list of shapes
    has already been computed by some other mean.

    :param data: data for which to compute the shapes or list of shapes
    :type data: list, series, array, dict, dataframe
    :param shapelist: true if `data` is a list of shapes
    :type shapelist: bool

    :return: shape in which each sample can be embedded
    :rtype: tuple or dict(str, tuple)
    """

    if shapelist is False:
        shapes = data_shapes(data)
    else:
        shapes = data

    if isinstance(shapes, dict):
        return {c: embedding_shape(l, True) for c, l in shapes.items()}

    max_size = max(len(s) for s in shapes)

    # special case for scalars
    if max_size == 0:
        return ()

    # fill shape with 1 to get all shapes of same length
    padded = [s + (1,) * (max_size - len(s)) for s in shapes]

    # take max for each dimension
    return tuple(np.max(padded, axis=0))


def is_homogeneous(data):
    """
    Give true if all elements of the data have the same shape.

    If the data is a table, return a dict where the function is applied to
    each column.

    :param data: The series from which to extract the shapes
    :type data: list, series, array, dict, dataframe

    :return: true if all shapes are identical
    :rtype: bool or dict(str, bool)
    """

    shapes = data_shapes(data)

    if isinstance(shapes, dict):
        return {c: is_homogeneous(l) for c, l in shapes.items()}

    return len(shapes) == 1


def pad_array(array, shape, value=0.):
    """
    Pad an array with a value to reach a given shape.

    If the target shape contains more dimensions than the original array,
    the latter is extended. One should be careful with this behaviour: for
    example, the components of a vector extended to a matrix will be in the
    first column.

    :param array: array to pad
    :type array: array
    :param shape: shape of the new array
    :type target_shape: tuple
    :param value: value to use for padding
    :type value: float

    :return: array made of the original array padded with the given value
    :rtype: array
    """

#    if (not isinstance(array, (list, tuple, np.ndarray))
#            or not np.issubdtype(type(array), np.floating)):
#        raise TypeError("`array` must be an array or an object which can be "
#                        "converted to an array, like a list.")

    array_shape = np.shape(array)

    # take into account the case where array is a scalar
    if array_shape == ():
        # if target is also scalar, then return the array
        if shape == ():
            return array

        array_shape = (1,)
        array = np.array((array,))

    length_diff = len(shape) - len(array_shape)

    if shape == (1,) and array_shape == (1,):
        return array

    if length_diff < 0:
        raise ValueError("Target shape must be bigger than the array shape.")
    elif length_diff > 0:
        array_shape += (1,) * length_diff
        array = np.reshape(array, array_shape)

    # compute the number of values that must be added in each dimension
    padding = np.subtract(shape, array_shape)
    # the argument is of the form (before, after): set before = 0
    padding = list(zip(np.zeros(padding.shape, dtype=int), padding))

    return np.pad(array, padding, mode="constant", constant_values=value)


def pad_data(data, shape=None, value=0.):
    """
    Pad all samples of the data to a common shape.

    The default shape is the biggest found in the column, but this can be
    changed with the argument `shape`.

    If the data is a dict or a dataframe and `shape` is a tuple or not given,
    pad all columns independently. Otherwise, `shape` can be a dict mapping
    columns to shapes: in this case, only the columns listed are updated with
    the given shape or with the embedding shape if the value is `None`.

    For a list or an array, the first dimension is ignored (since it lists the
    different samples.)

    A list or a tuple is converted to an array for simplicity.

    :param data: data to pad
    :type data: list, tuple, array, series, dataframe, dict
    :param shape: shape of the new array
    :type target_shape: tuple, dict(str, tuple|None)
    :param value: value to use for padding
    :type value: float

    :return: data padded with the given values
    :rtype: array | dict | dataframe
    """

    # if shape is not given, find the embedding shape
    if shape is None:
        shape = embedding_shape(data)

    if isinstance(data, (list, tuple, np.ndarray, pd.Series)):
        # if a dict is given for the shapes but the data is a sequence,
        # then raise error because it does not make sense
        if isinstance(shape, dict):
            raise ValueError("`shape` cannot be a dict if the data is a "
                             "sequence.")
    elif isinstance(data, (dict, pd.DataFrame)):
        # if the shape is a tuple and the data a table, then write a dict
        # where the same shape is used for all columns
        if isinstance(shape, (tuple, list)):
            shape = {c: shape for c in data}
    else:
        raise TypeError("Data type `{}` is not supported.".format(type(data)))

    size = len(data)

    if isinstance(data, np.ndarray):
        return pad_array(data, (size,) + shape, value)
    elif isinstance(data, (list, tuple)):
        return np.array([pad_array(e, shape, value) for e in data])
    elif isinstance(data, pd.Series):
        return data.apply(lambda x: pad_array(x, shape, value))
    elif isinstance(data, dict):
        # copy columns which are not updated
        dic = data.copy()
        for c, l in data.items():
            if c in shape:
                # compute embedding shape if value is None
                s = embedding_shape(l) if shape[c] is None else shape[c]
                dic[c] = pad_data(l, s, value)

        return dic
    elif isinstance(data, pd.DataFrame):
        # copy columns which are not updated
        df = data.copy()
        for c in data:
            if c in shape:
                # compute embedding shape if value is None
                s = embedding_shape(data[c]) if shape[c] is None else shape[c]
                df[c] = pad_data(data[c], s, value)

        return df


def seq_to_array(data, shape=None):
    """
    Transform a sequence to an array.

    This function is useful when the sequence contains tensors, especially
    of different shapes.

    If the values have different shapes, the data is padded to the embedding
    shape. The shape can also be enforced.

    :param data: data to convert to an array
    :type data: array
    :param shape: force embedding shape
    :type target_shape: tuple

    :return: original data written as an array
    :rtype: array
    """

    data = pad_data(data, shape=None)

    if isinstance(data, (tuple, list, np.ndarray)):
        # for a tuple, a list or an array, the pad_data already does everything
        return data
    elif isinstance(data, pd.Series):
        return np.stack(data.values)


def tab_to_array(data):
    """
    Convert a table to an array.

    All columns of the table are first converted to an array, which are then
    flatten and finally merged together.

    If a values in a column are tensors of different dimensions, then they
    are padded to reach the embedding shape before merging.
    """

    # works for both dict and dataframe
    size = len(data[list(data.keys())[0]])

    return np.hstack([seq_to_array(data[k]).reshape(size, -1) for k in data])
