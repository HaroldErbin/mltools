import pytest

import numpy as np
import pandas as pd

from mltools.data.structure import DataStructure


dic = {'id': np.arange(0, 10), 'letter': np.array(list('abcdefghij')),
       'number': np.arange(10.1, 20.),
       'line': list(np. arange(0, 30).reshape(-1, 3)),
       'square': list(np. arange(0, 4, 0.1).reshape(-1, 2, 2))}
df = pd.DataFrame(dic)
array = df[['id', 'number']].values
dic['line'] = np.array(dic['line'])
dic['square'] = np.array(dic['square'])


def test_structure_init_feature_dict():

    ds = DataStructure({'id': (), 'number': (), 'line': (3,),
                        'square': (2, 2)})

    assert ds.features == ['id', 'number', 'line', 'square']
    assert ds.types == {'id': 'scalar', 'number': 'scalar', 'line': 'vector',
                        'square': 'matrix'}
    assert ds.shapes == {'id': (1,), 'number': (1,), 'line': (3, 1),
                         'square': (2, 2, 1)}

    ds = DataStructure({'id': 'scalar', 'number': 'scalar'})

    assert ds.features == ['id', 'number']
    assert ds.types == {'id': 'scalar', 'number': 'scalar'}
    assert ds.shapes == {'id': (1,), 'number': (1,)}


def test_structure_init_feature_list():

    ds = DataStructure(['id', 'number'])

    assert ds.features == ['id', 'number']
    assert ds.types == {'id': 'scalar', 'number': 'scalar'}
    assert ds.shapes == {'id': (1,), 'number': (1,)}


def test_structure_init_feature_infer():

    ds = DataStructure(['id', 'number', 'line', 'square'], infer=df)

    assert ds.features == ['id', 'number', 'line', 'square']
    assert ds.types == {'id': 'scalar', 'number': 'scalar', 'line': 'vector',
                        'square': 'matrix'}
    assert ds.shapes == {'id': (1,), 'number': (1,), 'line': (3, 1),
                         'square': (2, 2, 1)}

    ds = DataStructure({'id': None, 'number': None, 'line': None,
                        'square': None}, infer=df)

    assert ds.features == ['id', 'number', 'line', 'square']
    assert ds.types == {'id': 'scalar', 'number': 'scalar', 'line': 'vector',
                        'square': 'matrix'}
    assert ds.shapes == {'id': (1,), 'number': (1,), 'line': (3, 1),
                         'square': (2, 2, 1)}

    ds = DataStructure({'id': 'scalar', 'number': None, 'line': (3,),
                        'square': None}, infer=df)

    assert ds.features == ['id', 'number', 'line', 'square']
    assert ds.types == {'id': 'scalar', 'number': 'scalar', 'line': 'vector',
                        'square': 'matrix'}
    assert ds.shapes == {'id': (1,), 'number': (1,), 'line': (3, 1),
                         'square': (2, 2, 1)}


def test_structure_init_feature_channels():

    ds = DataStructure(['id', 'number', 'line', 'square'], infer=df,
                       with_channels=['square'])

    assert ds.features == ['id', 'number', 'line', 'square']
    assert ds.types == {'id': 'scalar', 'number': 'scalar', 'line': 'vector',
                        'square': 'vector'}
    assert ds.shapes == {'id': (1,), 'number': (1,), 'line': (3, 1),
                         'square': (2, 2)}

    ds = DataStructure(['id', 'number', 'line', 'square'], infer=df,
                       with_channels=['line', 'square'])

    assert ds.features == ['id', 'number', 'line', 'square']
    assert ds.types == {'id': 'scalar', 'number': 'scalar', 'line': 'scalar',
                        'square': 'vector'}
    assert ds.shapes == {'id': (1,), 'number': (1,), 'line': (3,),
                         'square': (2, 2)}
