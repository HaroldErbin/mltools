import pytest

import numpy as np
import pandas as pd

from mltools.data.sample import RatioSample


def test_ratiosample_definition_ratios():

    df = pd.DataFrame()

    assert RatioSample(df, 0.2).ratios == {"train": 0.2, "test": 0.8}
    assert RatioSample(df, [0.2]).ratios == {"train": 0.2, "test": 0.8}

    assert RatioSample(df, [0.2, 0.8]).ratios == {"train": 0.2, "test": 0.8}
    assert (RatioSample(df, [0.2, 0.3]).ratios
            == {"train": 0.2, "val": 0.3, "test": 0.5})

    assert (RatioSample(df, [0.2, 0.3, 0.4]).ratios
            == {"train": 0.2, "val": 0.3, "test": 0.4})

    assert (RatioSample(df, [0.1, 0.2, 0.3, 0.3]).ratios
            == {"train": 0.1, "val": 0.2, "test": 0.3})
    assert (RatioSample(df, [0.1, 0.2, 0.3, 0.4]).ratios
            == {"train": 0.1, "val": 0.2, "test": 0.3})


def test_ratiosample_sum_ratios_error():

    df = pd.DataFrame()

    with pytest.raises(ValueError):
        RatioSample(df, [0.2, 0.9])


def test_ratiosample_notimplemented_errors():
    with pytest.raises(NotImplementedError):
        RatioSample(np.array([(0, 10), (1, 20)]), 0.4)

    with pytest.raises(NotImplementedError):
        RatioSample({"id": [0, 1], "value": [10, 20]}, 0.4)
