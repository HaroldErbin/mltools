import pytest

import numpy as np
import pandas as pd

from mltools.data.datatools import (tensor_name, tensor_dim, data_shapes,
                                    embedding_shape, is_homogeneous, pad_array,
                                    pad_data, seq_to_array, tab_to_array,
                                    linear_shape, linear_indices, split_array,
                                    array_to_dict)


s = 8
v = np.arange(0, 4)
m = np.arange(0, 6).reshape(2, 3)
t = np.arange(0, 24).reshape(2, 3, 4)

col1 = np.arange(5).tolist()
col2 = np.arange(10).reshape(5, 2).tolist()
col3 = [1, (2, 3, 4), 5, (6, 7), (8, 9)]
col4 = [1, (8, 9, 14, 15), [(2, 3, 4), (9, 10, 11)], 5, (6, 7, 12, 13)]

dic = {"col1": col1, "col2": col2, "col3": col3, "col4": col4}
df = pd.DataFrame(dic)


# TODO: data_shapes(df['col3'].values) gives [()]


def test_tensor_name():
    assert tensor_name(0) == "scalar"
    assert tensor_name(1) == "vector"
    assert tensor_name(2) == "matrix"
    assert tensor_name(3) == "tensor_3d"


def test_tensor_name_channels():
    assert tensor_name(1, True) == "scalar"
    assert tensor_name(2, True) == "vector"
    assert tensor_name(3, True) == "matrix"
    assert tensor_name(4, True) == "tensor_3d"


def test_tensor_dim():
    assert tensor_dim((2,)) == 1
    assert tensor_dim((2, 3)) == 2
    assert tensor_dim((2, 1)) == 2


def test_tensor_dim_channels():
    assert tensor_dim((2,), True) == 0
    assert tensor_dim((2, 3), True) == 1
    assert tensor_dim((2, 1), True) == 1


def test_data_shapes_array():
    assert data_shapes(np.array(col1)) == [()]
    assert data_shapes(np.array(col2)) == [(2,)]


def test_data_shapes_list():
    assert data_shapes(col1) == [()]
    assert data_shapes(col2) == [(2,)]
    assert data_shapes(col3) == [(), (2,), (3,)]
    assert data_shapes(col4) == [(), (2, 3), (4,)]

    assert data_shapes(df['col3'].values) == [(), (2,), (3,)]


def test_data_shapes_series():
    assert data_shapes(pd.Series(col1)) == [()]
    assert data_shapes(pd.Series(col2)) == [(2,)]
    assert data_shapes(pd.Series(col3)) == [(), (2,), (3,)]
    assert data_shapes(pd.Series(col4)) == [(), (2, 3), (4,)]


def test_data_shapes_dict():
    assert data_shapes(dic) == {"col1": [()], "col2": [(2,)],
                                "col3": [(), (2,), (3,)],
                                "col4": [(), (2, 3), (4,)]}


def test_data_shapes_dataframe():
    assert data_shapes(df) == {"col1": [()], "col2": [(2,)],
                               "col3": [(), (2,), (3,)],
                               "col4": [(), (2, 3), (4,)]}


def test_embedding_shape_shapelist():
    assert embedding_shape([()], True) == ()
    assert embedding_shape([(), (2, 3), (4,)], True) == (4, 3)


def test_embedding_shape_list_seq():
    assert embedding_shape(col1) == ()
    assert embedding_shape(col2) == (2,)
    assert embedding_shape(col3) == (3,)
    assert embedding_shape(col4) == (4, 3)

    assert embedding_shape(np.array(col1)) == ()
    assert embedding_shape(np.array(col2)) == (2,)

    assert embedding_shape(pd.Series(col1)) == ()
    assert embedding_shape(pd.Series(col2)) == (2,)
    assert embedding_shape(pd.Series(col3)) == (3,)
    assert embedding_shape(pd.Series(col4)) == (4, 3)


def test_embedding_shape_cols():
    assert embedding_shape(dic) == {"col1": (), "col2": (2,), "col3":  (3,),
                                    "col4": (4, 3)}
    assert embedding_shape(df) == {"col1": (), "col2": (2,), "col3":  (3,),
                                   "col4": (4, 3)}


def test_is_homogeneous_seq():
    assert is_homogeneous(col1) is True
    assert is_homogeneous(col2) is True
    assert is_homogeneous(col3) is False
    assert is_homogeneous(col4) is False

    assert is_homogeneous(np.array(col1)) is True
    assert is_homogeneous(np.array(col2)) is True

    assert is_homogeneous(pd.Series(col1)) is True
    assert is_homogeneous(pd.Series(col2)) is True
    assert is_homogeneous(pd.Series(col3)) is False
    assert is_homogeneous(pd.Series(col4)) is False


def test_is_homogeneous_cols():
    assert is_homogeneous(dic) == {"col1": True, "col2": True, "col3":  False,
                                   "col4": False}
    assert is_homogeneous(df) == {"col1": True, "col2": True, "col3":  False,
                                  "col4": False}


def test_pad_array():
    """
    Test pad_array with default value (zero).
    """

    # scalar
    assert pad_array(s, ()) == s
    assert pad_array(s, (1,)) == np.array(s)

    # vector
    assert (pad_array(v, v.shape) == v).all()
    assert (pad_array(v, (6,)) == np.append(v, [0, 0])).all()
    assert (pad_array(v, (6, 2))
            == np.append(v, [0] * 8).reshape(2, 6).T).all()

    # matrix
    assert (pad_array(m, m.shape) == m).all()

    assert (pad_array(m, (4, 3)) == np.array([[0, 1, 2], [3, 4, 5],
                                              [0, 0, 0], [0, 0, 0]])).all()
    assert (pad_array(m, (2, 4)) == np.array([[0, 1, 2, 0],
                                              [3, 4, 5, 0]])).all()
    assert (pad_array(m, (4, 4)) == np.array([[0, 1, 2, 0],
                                              [3, 4, 5, 0],
                                              [0, 0, 0, 0],
                                              [0, 0, 0, 0]])).all()

    # tensors
    assert (pad_array(t, t.shape) == t).all()
    assert (pad_array(t, (3, 3, 4))
            == np.array([[[ 0,  1,  2,  3], [ 4,  5,  6,  7],
                          [ 8,  9, 10, 11]],
                         [[12, 13, 14, 15], [16, 17, 18, 19],
                          [20, 21, 22, 23]],
                         [[ 0,  0,  0,  0], [ 0,  0,  0,  0],
                          [ 0,  0,  0,  0]]])).all()

    # test with list
    assert (pad_array(list(v), v.shape) == v).all()
    assert (pad_array(tuple(v), v.shape) == v).all()
    assert (pad_array(list(v), (6,)) == np.append(v, [0, 0])).all()


def test_pad_array_value():
    """
    Test pad_array with given value (here 1).
    """

    assert pad_array(s, (1,), 1) == s

    assert (pad_array(s, (2,), 1) == np.array([s, 1])).all()

    assert (pad_array(v, v.shape, 1) == v).all()

    assert (pad_array(v, (6,), 1) == np.append(v, [1, 1])).all()

    assert (pad_array(m, m.shape, 1) == m).all()

    assert (pad_array(m, (4, 3), 1) == np.array([[0, 1, 2], [3, 4, 5],
                                                 [1, 1, 1], [1, 1, 1]])).all()


def test_pad_array_errors():
    with pytest.raises(ValueError):
        pad_array(v, ())

    with pytest.raises(ValueError):
        pad_array(v, (1,))


def test_pad_data_array():

    assert (pad_data(np.array(col1)) == np.array(col1)).all()
    assert (pad_data(np.array(col1), ()) == np.array(col1)).all()

    assert (pad_data(np.array(col1), (1,))
            == np.array(col1).reshape(5, 1)).all()

    assert (pad_data(np.array(col1), (2,))
            == np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])).all()

    assert (pad_data(np.array(col1), (2, 3))
            == np.array([[[0, 0, 0], [0, 0, 0]],
                         [[1, 0, 0], [0, 0, 0]],
                         [[2, 0, 0], [0, 0, 0]],
                         [[3, 0, 0], [0, 0, 0]],
                         [[4, 0, 0], [0, 0, 0]]])).all()


def test_pad_data_list():

    assert (pad_data(col1) == np.array(col1)).all()
    assert (pad_data(col1, ()) == np.array(col1)).all()

    assert (pad_data(col1, (2,))
            == np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])).all()

    assert (pad_data(col3)
            == np.array([[1, 0, 0], [2, 3, 4], [5, 0, 0],
                         [6, 7, 0], [8, 9, 0]])).all()


def test_pad_data_series():

    assert pad_data(pd.Series(col1)).equals(pd.Series(col1))
    assert pad_data(pd.Series(col1), ()).equals(pd.Series(col1))

    assert (pad_data(pd.Series(col1), (2,)).apply(lambda x: x.tolist())
            .equals(pd.Series([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])))

    assert (pad_data(pd.Series(col3)).apply(lambda x: x.tolist())
            .equals(pd.Series([[1, 0, 0], [2, 3, 4], [5, 0, 0],
                               [6, 7, 0], [8, 9, 0]])))

    # test output with attributes `values`
    assert (pad_data(pd.Series(col1).values)
            == np.array([0, 1, 2, 3, 4])).all()
    assert (pad_data(pd.Series(col3).values)
            == np.array([[1, 0, 0], [2, 3, 4], [5, 0, 0],
                         [6, 7, 0], [8, 9, 0]])).all()


def test_pad_data_dic():

    padded_dic = pad_data(dic)

    assert (padded_dic["col1"] == pad_data(col1)).all()
    assert (padded_dic["col2"] == pad_data(col2)).all()
    assert (padded_dic["col3"] == pad_data(col3)).all()
    assert (padded_dic["col4"] == pad_data(col4)).all()


def test_pad_data_dic_col():

    padded_dic = pad_data(dic, {"col1": (2,)})

    assert (padded_dic["col1"] == pad_data(col1, (2,))).all()
    assert padded_dic["col2"] == col2
    assert padded_dic["col3"] == col3
    assert padded_dic["col4"] == col4

    padded_dic = pad_data(dic, {"col3": None})

    assert padded_dic["col1"] == col1
    assert padded_dic["col2"] == col2
    assert (padded_dic["col3"] == pad_data(col3)).all()
    assert padded_dic["col4"] == col4

    padded_dic = pad_data(dic, {"col1": (2,), "col3": None})

    assert (padded_dic["col1"] == pad_data(col1, (2,))).all()
    assert padded_dic["col2"] == col2
    assert (padded_dic["col3"] == pad_data(col3)).all()
    assert padded_dic["col4"] == col4


def test_pad_data_dataframe():

    padded_df = pad_data(df)

    assert padded_df["col1"].equals(pad_data(pd.Series(col1)))
    assert (padded_df["col2"].apply(lambda x: x.tolist())
            .equals(pad_data(pd.Series(col2)).apply(lambda x: x.tolist())))
    assert (padded_df["col3"].apply(lambda x: x.tolist())
            .equals(pad_data(pd.Series(col3)).apply(lambda x: x.tolist())))
    assert (padded_df["col4"].apply(lambda x: x.tolist())
            .equals(pad_data(pd.Series(col4)).apply(lambda x: x.tolist())))


def test_pad_data_dataframe_col():

    padded_df = pad_data(df, {"col1": (2,)})
    s1 = pad_data(pd.Series(col1), (2,))

    assert (padded_df["col1"].apply(lambda x: x.tolist())
            .equals(s1.apply(lambda x: x.tolist())))
    assert padded_df["col2"].equals(pd.Series(col2))
    assert padded_df["col3"].equals(pd.Series(col3))
    assert padded_df["col4"].equals(pd.Series(col4))

    padded_df = pad_data(df, {"col3": None})

    assert padded_df["col1"].equals(pd.Series(col1))
    assert padded_df["col2"].equals(pd.Series(col2))
    assert (padded_df["col3"].apply(lambda x: x.tolist())
            .equals(pad_data(pd.Series(col3)).apply(lambda x: x.tolist())))
    assert padded_df["col4"].equals(pd.Series(col4))


def test_seq_to_array():

    assert (seq_to_array(v) == v).all()

    assert (seq_to_array(col1) == pad_data(col1)).all()
    assert (seq_to_array(col2) == pad_data(col2)).all()

    assert (seq_to_array(pd.Series(col1)) == pad_data(col1)).all()
    assert (seq_to_array(pd.Series(col2)) == pad_data(col2)).all()

    # use padding before
    assert (seq_to_array(pad_data(col3)) == pad_data(col3)).all()
    assert (seq_to_array(pad_data(col4)) == pad_data(col4)).all()

    assert (seq_to_array(pad_data(pd.Series(col3))) == pad_data(col3)).all()


def test_seq_to_array_errors():
    with pytest.raises(ValueError):
        seq_to_array(col3)

    with pytest.raises(ValueError):
        seq_to_array(pd.Series(col3))


def test_tab_to_array():

    res12 = {k: seq_to_array(v) for k, v
             in zip(["col1", "col2"], [col1, col2])}
    res1234 = {k: seq_to_array(pad_data(v)) for k, v
               in zip(["col1", "col2", "col3", "col4"],
                      [col1, col2, col3, col4])}

    for v1, v2 in zip(tab_to_array(df[["col1", "col2"]]), res12):
        assert v1 == v2

    for v1, v2 in zip(tab_to_array({k: v for k, v in dic.items()
                                    if k in ["col1", "col2"]}), res12):
        assert v1 == v2

    # use padding before
    for v1, v2 in zip(tab_to_array(pad_data(df)), res1234):
        assert v1 == v2

    for v1, v2 in zip(tab_to_array(pad_data(dic)), res1234):
        assert v1 == v2


def test_tab_to_array_flatten():

    res12 = np.hstack([seq_to_array(x).reshape(5, -1) for x in [col1, col2]])
    res1234 = np.hstack([seq_to_array(pad_data(x)).reshape(5, -1) for x
                         in [col1, col2, col3, col4]])

    assert (tab_to_array(df[["col1", "col2"]], flatten=True) == res12).all()

    assert (tab_to_array({k: v for k, v in dic.items()
                          if k in ["col1", "col2"]}, flatten=True)
            == res12).all()

    # use padding before
    assert (tab_to_array(pad_data(df), flatten=True) == res1234).all()
    assert (tab_to_array(pad_data(dic), flatten=True) == res1234).all()


def test_tab_to_array_errors():
    with pytest.raises(ValueError):
        tab_to_array(df)

    with pytest.raises(ValueError):
        tab_to_array(dic)


def test_linear_shape():

    assert linear_shape(()) == 1
    assert linear_shape((1,)) == 1
    assert linear_shape((2,)) == 2
    assert linear_shape((2, 3)) == 6


def test_linear_shape_cum():

    assert linear_shape([()], True) == 1
    assert linear_shape([(), ()], True) == 2
    assert linear_shape([(1,)], True) == 1

    assert linear_shape([(), (2,)], True) == 3
    assert linear_shape([(), (2, 3)], True) == 7
    assert linear_shape([(), (2,), (3, 4)], True) == 15


def test_linear_indices():

    assert linear_indices([()]) == [(0, 1)]
    assert linear_indices([(), ()]) == [(0, 1), (1, 2)]
    assert linear_indices([(1,)]) == [(0, 1)]

    assert linear_indices([(), (2,)]) == [(0, 1), (1, 3)]
    assert linear_indices([(), (2, 3)]) == [(0, 1), (1, 7)]
    assert linear_indices([(), (2,), (3, 4)]) == [(0, 1), (1, 3), (3, 15)]


def test_split_array():

    array = np.hstack([seq_to_array(pad_data(x)).reshape(5, -1) for x
                       in [col1, col2, col3, col4]])
    shapes = list(embedding_shape(df).values())

    true = list(pad_data(dic).values())

    for v1, v2 in zip(split_array(array, shapes), true):
        assert (v1 == v2).all()

    for v1, v2 in zip(split_array(tab_to_array(pad_data(dic), flatten=True),
                                  shapes), true):
        assert (v1 == v2).all()


def test_array_to_dict():

    array = np.hstack([seq_to_array(pad_data(x)).reshape(5, -1) for x
                       in [col1, col2, col3, col4]])
    shapes = embedding_shape(dic)

    result = array_to_dict(array, shapes)
    true = pad_data(dic)

    for k1, v1, k2, v2 in zip(result.keys(), result.values(),
                              true.keys(), true.values()):
        assert (v1 == v2).all()
        assert k1 == k2
