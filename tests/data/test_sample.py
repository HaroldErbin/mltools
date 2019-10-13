import pytest

import numpy as np
import pandas as pd

from mltools.data.sample import RatioSample


dic = {'id': np.arange(0, 10), 'letter': np.array(list('abcdefghij')),
       'number': np.arange(20, 30),
       'matrix': list(np. arange(0, 4, 0.1).reshape(-1, 2, 2))}
df = pd.DataFrame(dic)
array = df[['id', 'number']].values
dic['matrix'] = np.array(dic['matrix'])


def test_ratiosample_def_ratios_numbers():

    assert RatioSample(0.2).ratios == {"train": 0.2, "test": 0.8}
    assert RatioSample([0.2]).ratios == {"train": 0.2, "test": 0.8}

    assert RatioSample([0.2, 0.8]).ratios == {"train": 0.2, "test": 0.8}
    assert (RatioSample([0.2, 0.3]).ratios
            == {"train": 0.2, "val": 0.3, "test": 0.5})

    assert (RatioSample([0.2, 0.3, 0.4]).ratios
            == {"train": 0.2, "val": 0.3, "test": 0.4})

    assert (RatioSample([0.1, 0.2, 0.3, 0.3]).ratios
            == {"train": 0.1, "val": 0.2, "test": 0.3})
    assert (RatioSample([0.1, 0.2, 0.3, 0.4]).ratios
            == {"train": 0.1, "val": 0.2, "test": 0.3})


def test_ratiosample_def_ratios_dict():

    assert (RatioSample({"train": 0.2, "val": 0.3, "test": 0.4}).ratios
            == {"train": 0.2, "val": 0.3, "test": 0.4})
    assert (RatioSample({"train": 0.2, "val": 0.3, "test": 0.5}).ratios
            == {"train": 0.2, "val": 0.3, "test": 0.5})

    assert (RatioSample({"train-nn": 0.1, "train-reg": 0.2, "val": 0.3,
                         "test": 0.4}).ratios
            == {"train-nn": 0.1, "train-reg": 0.2, "val": 0.3, "test": 0.4})


def test_ratiosample_sum_ratios_error():

    with pytest.raises(ValueError):
        RatioSample([0.2, 0.9])


def test_ratiosample_make_samples():

    assert (RatioSample([0.2]).make_samples(10)
            == {'train': (0, 1), 'test': (2, 3, 4, 5, 6, 7, 8, 9)})

    assert (RatioSample([0.2, 0.3]).make_samples(10)
            == {'train': (0, 1), 'val': (2, 3, 4), 'test': (5, 6, 7, 8, 9)})


def test_ratiosample_make_samples_shuffle():

    np.random.seed(42)

    assert (RatioSample([0.2, 0.3]).make_samples(10, True)
            == {'train': (8, 1), 'val': (5, 0, 7), 'test': (2, 9, 4, 3, 6)})


def test_ratiosample_make_samples_rounding():
    r = RatioSample([0.25, 0.3])
    assert sum(map(len, r.make_samples(1500).values())) == 1500

    r = RatioSample([0.25, 0.23])
    assert sum(map(len, r.make_samples(1500).values())) == 1500

    r = RatioSample({"train": 0.19, "test": 0.56})
    assert sum(map(len, r.make_samples(2000).values())) == 1500


def test_ratiosample_call_array():

    sample = RatioSample([0.2, 0.3])

    array_true = {'train': np.array(((0, 20), (1, 21))),
                  'val': np.array(((2, 22), (3, 23), (4, 24))),
                  'test': np.array(((5, 25), (6, 26), (7, 27), (8, 28),
                                    (9, 29)))}
    array_res = sample(array)

    for k in array_true:
        np.testing.assert_array_equal(array_res[k], array_true[k])


def test_ratiosample_call_dataframe():

    sample = RatioSample([0.2, 0.3])

    df_true = {'train': df.loc[[0, 1]],
               'val': df.loc[[2, 3, 4]],
               'test': df.loc[[5, 6, 7, 8, 9]]}

    df_res = sample(df)

    for k in df_true:
        pd.testing.assert_frame_equal(df_res[k], df_true[k])


def test_ratiosample_call_dic():

    sample = RatioSample([0.2, 0.3])

    dic_true = {'train': {'id': np.array([0, 1]),
                          'letter': np.array(['a', 'b']),
                          'number': np.array([20, 21]),
                          'matrix': dic['matrix'][:2, :, :]},
                'val': {'id': np.array([2, 3, 4]),
                        'letter': np.array(['c', 'd', 'e']),
                        'number': np.array([22, 23, 24]),
                        'matrix': dic['matrix'][2:5, :, :]},
                'test': {'id': np.array([5, 6, 7, 8, 9]),
                         'letter': np.array(['f', 'g', 'h', 'i', 'j']),
                         'number': np.array([25, 26, 27, 28, 29]),
                         'matrix': dic['matrix'][5:, :, :]}}

    dic_res = sample(dic)

    for k1 in dic_true:
        for k2 in dic:
            np.testing.assert_array_equal(dic_true[k1][k2], dic_res[k1][k2])
