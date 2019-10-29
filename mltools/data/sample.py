"""
Get data samples from a dataset

Note that these classes only provide method to split a dataset, it does not
store any data or references to the data itself.
"""

import numpy as np
import pandas as pd


# TODO: sample based on id
# TODO: sample with stratification
# TODO: sample built from several datasets


class RatioSample:
    """
    Select subsets from a dataset according to a list of ratios

    The ratios are defined by a list of numbers between 0 and 1, each
    designated by a label. The latter can be used in ML algorithms.

    The ratios don't have to add to 1 (in that case, some data are not used).

    A list can contain two (`train` and `test`) or three ratios (`train`,
    `val`, and `test`). All supplementary numbers are ignored. If the ratios
    of a list don't add to 1, then the different to 1 is added to the list.
    This helps to quickly change the number of training versus test data.
    """

    def __init__(self, ratios):
        """
        :param ratios: ratios used to divide the data in subsets
        :type ratios: `float`, `list`, `tuple` or `dict`
        """

        if not isinstance(ratios, (dict, tuple, list, float)):
            raise TypeError("The ratios must be given as a dict, a list or "
                            "be a single number.")

        if isinstance(ratios, float):
            # if the ratio is a number, put it inside a list
            ratios = [ratios, 1 - ratios]

        if isinstance(ratios, (tuple, list)):
            sum_ratios = sum(ratios)

            # add difference to 1 if the ratios don't add to 1
            if sum_ratios < 1:
                ratios.append(1 - sum_ratios)

            # set default labels
            if len(ratios) == 2:
                ratios = dict(zip(('train', 'test'), ratios))
            else:
                ratios = dict(zip(('train', 'val', 'test'), ratios[:3]))

        self.ratios = ratios

        sum_ratios = sum(self.ratios.values())

        if sum_ratios > 1 or sum_ratios <= 0:
            raise ValueError("The sum of ratios must be between 0 and 1, "
                             "found {}.".format(sum_ratios))

    def __repr__(self):
        return "<RatioSample: {}>".format(self.ratios)

    def __call__(self, dataset, shuffle=False):
        """
        Split a set of id into subsets according to the class ratios.

        The dataset is shuffled before extracting the subsets if `shuffle`
        is true.

        :param dataset: data to split in subsets
        :type dataset: `pandas.DataFrame`, `numpy.array`, `dict`
        :param shuffle: shuffle the dataset before extracting subsets
        :type shuffle: `bool`

        :return: subsets arranged in a `dict`
        :rtype: `dict`
        """

        if isinstance(dataset, dict):
            # for a dict, get number of samples by looking at the first key
            size = len(list(dataset.values())[0])
        elif isinstance(dataset, (pd.DataFrame, np.ndarray)):
            # works only for array, dataframe
            size = len(dataset)
        else:
            raise TypeError("Dataset type `{}` is not supported"
                            .format(type(dataset)))

        idx_splits = self.make_samples(size, shuffle)
        value_splits = {}

        for key, idx in idx_splits.items():
            if isinstance(dataset, dict):
                value_splits[key] = {}
                for col, val in dataset.items():
                        value_splits[key][col] = val[idx, ...]
            elif isinstance(dataset, np.ndarray):
                    value_splits[key] = dataset[idx, ...]
            elif isinstance(dataset, pd.DataFrame):
                    value_splits[key] = dataset.loc[list(idx)]

        return value_splits

    def make_samples(self, size, shuffle=False):
        """
        Split a set of id into subsets according to the class ratios.

        :param size: size of the dataset
        :type size: `int`
        :param shuffle: shuffle the indices before extracting subsets
        :type shuffle: `bool`

        :return: indices for each subsets arranged in a `dict`
        :rtype: `dict`
        """

        splits = {}
        indices = range(0, size)

        if shuffle is True:
            indices = np.random.permutation(indices)

        last = 0

        for key, ratio in self.ratios.items():
            delta = round(ratio * size)
            splits[key] = tuple(indices[last:last+delta])
            last += delta

        # TODO: this assumes that the samples are indexed with consecutive
        #   integers; this should work with more general id

        return splits
