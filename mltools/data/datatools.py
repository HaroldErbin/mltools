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


# TODO: merge arrays on channels (last dimension)

# TODO: functions to combine or update dataframes (argument may be dict)


def affix_keys(dic, prefix="", suffix=""):
    """
    Add prefix and suffix to all keys of a dict.
    """

    return {(prefix + k + suffix): v for k, v in dic.items()}


def equal_length_names(data, align="left"):
    """
    Set all a list of strings to the same length.

    If the argument is a dict, then apply the function to the keys.

    Alignment can be `left` or `right`.
    """

    if isinstance(data, dict):
        names = data.keys()
    else:
        names = data

    max_length = max(map(len, names))

    if align == "right":
        fmt = "{:>{}s}"
    else:
        fmt = "{:<{}s}"

    names = [fmt.format(name, max_length) for name in names]

    if isinstance(data, dict):
        return dict(zip(names, data.values()))
    elif isinstance(data, tuple):
        return tuple(names)
    else:
        return names


def exchange_list_dict(data):
    """
    Transform a list of dicts into a dict of lists.
    """

    dic = {}

    for d in data:
        for k, v in d.items():
            if k not in dic:
                dic[k] = []
            dic[k].append(v)

    return dic


def exchange_keyval(data):
    """
    Exchange keys and values of a dict.
    """

    dic = {}

    for k, v in data.items():
        if v not in dic:
            dic[v] = []
        dic[v].append(k)

    return dic


def infer_types(data, ncat=10, dict_type=False):
    """
    Infer data type.

    The parameter `ncat` indicates the threshold of different values below
    which the data is considered as categorical.

    If the data is a dict or a dataframe, then: if `dict_type` is `True, keys
    are data types and values are column names, otherwise it is the converse.
    """

    if isinstance(data, (tuple, list, np.ndarray)):
        shapes = data_shapes(data)

        # if len(shapes) > 1:
        #     return "object"
        if shapes != [()]:
            return "tensor"
        else:
            dtype = set(type(x) for x in data)
            values = np.unique([x for x in data])

            if len(dtype) > 1:
                raise ValueError("Can infer type only when data contains "
                                 "values of a single type.")
            dtype = dtype.pop()

            if np.issubdtype(dtype, np.str_):
                if len(values) <= ncat:
                    return "category"
                else:
                    return "string"
            elif np.issubdtype(dtype, np.integer):
                if np.all(values == [0, 1]):
                    return "binary"
                elif len(values) <= ncat:
                    return "category"
                else:
                    return "integer"
            elif np.issubdtype(dtype, np.floating):
                return "scalar"
            else:
                return str(dtype)
    elif isinstance(data, pd.Series):
        return infer_types(data.values, ncat=ncat)
    elif isinstance(data, (dict, pd.DataFrame)):
        dtypes = {col: infer_types(val, ncat=ncat)
                  for col, val in data.items()}

        if dict_type is True:
            return exchange_keyval(dtypes)
        else:
            return dtypes

    else:
        return str(type(data))


def filter_features(data, types, ncat=0):
    return [k for k, t in infer_types(data, ncat=ncat).items() if t in types]


def filter_on_types(data, types, ncat=0):
    cols = filter_features(data, types, ncat)

    if isinstance(data, pd.DataFrame):
        return data[cols]
    elif isinstance(data, dict):
        return {k: v for k, v in data.items() if k in cols}


def dict_to_dataframe(dic):
    """
    Convert a dict to a dataframe.

    Columns corresponding to types which cannot be converted are silently
    ignored.
    """

    # TODO: convert also tensors
    # TODO: convert other types

    features = [k for k, t in infer_types(dic, ncat=0).items()
                if t in ("scalar", "integer", "binary", "string")]

    df = pd.DataFrame({k: v for k, v in dic.items() if k in features})

    return df


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

    # this does not work for array obtained from Series.values
    # they contain themselves list
#    if isinstance(data, np.ndarray):
#        return [np.shape(data)[1:]]
    if isinstance(data, (list, tuple, np.ndarray, pd.Series)):
        return sorted(list(set(np.shape(x) for x in data)))
#    elif isinstance(data, pd.Series):
#        return sorted(list(data.apply(lambda x: np.shape(x)).unique()))
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

    If the array cannot be embedded in the target shape (for example if the
    target has less dimensions or if one dimension is smaller), raise an error.

    Contrary to most other functions of this module, this function treats all
    dimensions on an equal footing.

    This function always convert a list or tuple to an array even when
    reshapping or padding is not necessary.

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

    if isinstance(array, (list, tuple)):
        array = np.array(array)

    # if shapes are identical, do nothing
    if shape == array_shape:
        return array

    # convert scalar to array to use array methods
    # do this before all other steps
    # note: case array_shape == shape == () is taken care of above
    if array_shape == ():
        array_shape = (1,)
        array = np.array((array,))

        if shape == (1,):
            return array

    # differences between dimension of array and target
    length_diff = len(shape) - len(array_shape)
    dim_diff = [d1 - d2 for d1, d2 in zip(shape, array_shape)]

    # target has less dimensions or one dimension is smaller
    if length_diff < 0 or True in [d < 0 for d in dim_diff]:
        raise ValueError("Target shape must be bigger than the array shape. "
                         "Array has shape {}, target is {}."
                         .format(array_shape, shape))
#        return array

    # if difference is only 1 in last dimension, reshape
    if shape == array_shape + (1,):
        return array.reshape(*shape)

    # TODO: should be taken care of by previous cases, but it's not the case
#    if shape == (1,) and array_shape == (1,):
#        return array

    # match shape length by adding ones to the array shape and reshape
    if length_diff > 0:
        array_shape += (1,) * length_diff
        array = np.reshape(array, array_shape)

    # compute the number of values that must be added in each dimension
    padding = np.subtract(shape, array_shape)
    # the argument is of the form (before, after): set before = 0
    padding = list(zip(np.zeros(padding.shape, dtype=int), padding))

    return np.pad(array, padding, mode="constant", constant_values=value)


def pad_data(data, shape=None, value=0., toarray=False):
    """
    Pad all samples of the data to a common shape.

    The default shape is the biggest found in the column, but this can be
    changed with the argument `shape`.

    If the data is a dict or a dataframe and `shape` is a tuple or not given,
    pad all columns independently. Otherwise, `shape` can be a dict mapping
    columns to shapes: in this case, only the columns listed are updated with
    the given shape or with the embedding shape if the value is `None`.

    For list, tuple or array, the first dimension is ignored (since it lists
    the different samples). A list or a tuple is converted to an array for
    simplicity.

    Given a series, the result is the same series with individual element
    padded. To get an array, pass `series.values` to the function.

    One advantage of this function over `pad_array` is that it is not necessary
    to provide a target shape. This is useful when applying the function to
    many elements without having to decide which ones should be padded.

    :param data: data to pad
    :type data: list, tuple, array, series, dataframe, dict
    :param shape: shape of the new array
    :type target_shape: tuple, dict(str, tuple|None)
    :param value: value to use for padding
    :type value: float

    :return: data padded with the given values
    :rtype: array | dict | dataframe
    """
    # TODO: remove toarray ?

    # if shape is not given, find the embedding shape
    if shape is None:
        shape = embedding_shape(data)

    if isinstance(data, (list, tuple, np.ndarray, pd.Series)):
        # if a dict is given for the shapes but the data is a sequence,
        # then raise error because it does not make sense
        if isinstance(shape, dict):
            raise ValueError("`shape` cannot be a dict if the data is a "
                             "sequence.")

        if toarray is True:
            data = np.array(data)
    elif isinstance(data, (dict, pd.DataFrame)):
        # if the shape is a tuple and the data a table, then write a dict
        # where the same shape is used for all columns
        if isinstance(shape, (tuple, list)):
            shape = {c: shape for c in data}
    else:
        raise TypeError("Data type `{}` is not supported.".format(type(data)))

#    size = len(data)
#    shapes = data_shapes(data)

    if isinstance(data, (list, tuple, np.ndarray)):
#        if len(shapes) > 1 or isinstance(data, (list, tuple)):
        return np.array([pad_array(e, shape, value) for e in data])
#        else:
#            return pad_array(data, (size,) + shape, value)
    elif isinstance(data, pd.Series):
#        if is_homogeneous(data):
#            return pad_array(data.values, (len(data),) + shape, value)
#        else:
#        return data.apply(lambda x: pad_array(x, shape, value))

        # TODO: speed
        return pd.Series([pad_array(x, shape, value) for x in data],
                         index=data.index)
    elif isinstance(data, dict):
        # copy columns which are not updated
        dic = data.copy()
        for c, l in data.items():
            if c in shape:
                # compute embedding shape if value is None
                s = embedding_shape(l) if shape[c] is None else shape[c]
                dic[c] = pad_data(l, s, value, toarray=toarray)

        return dic
    elif isinstance(data, pd.DataFrame):
        # copy columns which are not updated
        df = data.copy()
        for c in data:
            if c in shape:
                # compute embedding shape if value is None
                s = embedding_shape(data[c]) if shape[c] is None else shape[c]
                df[c] = pad_data(data[c], s, value, toarray=toarray)

        return df


def seq_to_array(data):
    """
    Transform a sequence to an array.

    Note that no padding: entries with different shapes will yield an error.
    This is motivated by performance issues, especially with series.

    :param data: data to convert to an array
    :type data: array
    :param shape: shape of the final array
    :type target_shape: tuple
    :param value: value used for padding
    :type target_shape: float

    :return: original data written as an array
    :rtype: array
    """

    if isinstance(data, pd.Series):
        data = np.stack(data.values)
    elif isinstance(data, (tuple, list)):
        data = np.stack(data)

    return data


def tab_to_array(data, flatten=False):
    """
    Convert a table to a dict of arrays or to a flattened array.

    All columns of the table are first converted to an array.

    If flatten is `True`, then all arrays are flatten and merged together.
    The result is a matrix where first dimension is the number of samples
    and the second all other dimensions. The latter can be computed with the
    function `linear_shape`.

    Note that no padding: entries with different shapes will yield an error.
    """

    data = {k: seq_to_array(data[k]) for k in data}

    if flatten is True:
        # works for both dict and dataframe
        size = len(data[list(data.keys())[0]])

        return np.hstack([v.reshape(size, -1) for v in data.values()])
    else:
        return data


def linear_shape(shape, cum=False):
    """
    Compute the linear shape for a shape or a set of shapes.

    The linear shape is the product of all entries of the shape together.
    This corresponding to the size of the vector obtained by flattening the
    tensor of the corresponding shape.

    If `cum` is true, then `shape` must be a list of shapes. This describes
    the shape of the vector in which all tensors can be embedded after
    flattening.
    """

    if cum is True:
        return int(sum(np.multiply.reduce(s) for s in shape))
    else:
        return np.multiply.reduce(shape)


def linear_indices(shapes):
    """
    Give pair of indices to recover tensors embedded
    """

    # linear shape for each shape
    lin_shapes = [linear_shape(s) for s in shapes]

    # cumulative sum give end indices
    end = np.cumsum(lin_shapes, dtype=int).tolist()

    return list(zip([0] + end[:-1], end))


def split_array(array, shapes):
    """
    Split a matrix into arrays of lower dimensions.

    A vector is a valid input and will be seen as a matrix with one column.
    This is for compatibility with some algorithms returning a matrix or a
    vector depending if it is performing a single or several tasks.

    Note that the sum of linear dimensions of all shapes may be smaller than
    the linear dimension of the samples. This behaviour is not desirable but
    is left in case of need.
    On the other hand, there can be problem when the sum is bigger, such that
    an error is always raised.
    """

    if isinstance(array, (list, tuple)):
        array = np.array(array)

    if len(array.shape) > 2:
        raise ValueError("This function works only with matrix.")

    target_dim = linear_shape(shapes, True)

    # some algorithm returns a vector or a matrix, so consider a vector as
    # a valid matrix
    try:
        dim = array.shape[1]
    except IndexError:
        dim = 1
        return [array.reshape(-1, 1)]

    if target_dim > dim:
        raise ValueError("The shapes given do not fit in the vector.")

    indices = linear_indices(shapes)

    return [array[:, slice(*ind)].reshape(-1, *s)
            for s, ind in zip(shapes, indices)]


def array_to_dict(array, shapes):
    """
    Convert an array to a dict of arrays.

    The argument `shapes` is a dict associating keys to shapes.
    """

    array_list = split_array(array, list(shapes.values()))

    return {k: a for k, a in zip(shapes.keys(), array_list)}


def average(ensemble, axis=0):
    """
    Define basic average.
    """

    # if data is a dict of list, average
    if isinstance(ensemble, dict):
        mean = {f: np.mean(v, axis=axis) for f, v in ensemble.items()}
        std = {f: np.std(v, axis=axis) for f, v in ensemble.items()}

        return mean, std

    # if not, assume it is a list of dict or list
    types = set(map(type, ensemble))

    if len(types) > 1:
        raise TypeError(f"Can average only on an ensemble containing a "
                        "single type of predictions. FOund `{types}`.")

    datatype = types.pop()

    if datatype == dict:
        mean, std = average(exchange_list_dict(ensemble))
    elif datatype in (list, tuple, np.ndarray):
        mean = np.mean(ensemble, axis=axis)
        std = np.std(ensemble, axis=axis)
    else:
        raise TypeError(f"Cannot average {datatype}.")

    return mean, std
