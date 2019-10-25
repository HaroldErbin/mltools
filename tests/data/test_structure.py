import pytest

import numpy as np
import pandas as pd

from mltools.data.structure import DataStructure


dic = {'id': np.arange(0, 10), 'letter': np.array(list('abcdefghij')),
       'number': np.arange(10.1, 20.),
       'square': list(np. arange(0, 4, 0.1).reshape(-1, 2, 2))}
df = pd.DataFrame(dic)
array = df[['id', 'number']].values
dic['square'] = np.array(dic['square'])


def test_structure_init_feature_dict():

    ds = DataStructure({'id': 'integer', 'number': 'scalar',
                        'square': (2, 2)})

    assert ds.features == ['id', 'number', 'square']
    assert ds.types == {'id': 'integer', 'number': 'scalar',
                        'square': 'matrix'}
    assert ds.shapes == {'square': (2, 2, 1)}


def test_structure_init_feature_list():

    ds = DataStructure(['id', 'number'])

    assert ds.features == ['id', 'number']
    assert ds.types == {'id': 'scalar', 'number': 'scalar'}
    assert ds.shapes == {}


def test_structure_init_feature_infer():

    ds = DataStructure(['id', 'number', 'square'], infer=df)

    assert ds.features == ['id', 'number', 'square']
    assert ds.types == {'id': 'integer', 'number': 'scalar',
                        'square': 'matrix'}
    assert ds.shapes == {'square': (2, 2, 1)}


def test_structure_init_feature_channels():

    ds = DataStructure(['id', 'number', 'square'], infer=df,
                       with_channels=['square'])

    assert ds.features == ['id', 'number', 'square']
    assert ds.types == {'id': 'integer', 'number': 'scalar',
                        'square': 'vector'}
    assert ds.shapes == {'square': (2, 2)}
