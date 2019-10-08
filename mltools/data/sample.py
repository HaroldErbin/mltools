"""
Get data samples from a dataset
"""

import numpy as np
import pandas as pd


# TODO: sample with stratification
# TODO: sample built from several datasets

# Question: 1) store dataset in RatioSample, or 2) the opposite,
# or 3) make a mixed class?
# 1) Allow to quickly get data with sample['train'], but mix two different
#   ideas in a single class
# 2) more difficult to handle several datasets at once, except if there is
#   already a class to handle several datasets;
#   problem: to ensure that several calls always give the same outputs, this
#   requires to always use a Dataset class (for quick problem we may prefer
#   to use directly a dataframe)
#   (i.e. where to cache indices?)
# 3) maybe too heavy
# one possibily for the problem in 2) is to cache the indices in the instance
# itself using a dict id(obj) → indices (but is it reliable?)


class RatioSample:
    """
    Select subsets from a dataset according to a list of ratios

    The ratios are defined by a list of numbers between 0 and 1, each
    designated by a label. The latter is used by the ML algorithm to know
    when to use each subset.

    The ratios don't have to add to 1 (in that case, some data are not used).

    A list can contain two (`train` and `test`) or three ratios (`train`,
    `val`, and `test`). All supplementary numbers are ignored. If the ratios
    of a list don't add to 1, then the different to 1 is added to the list.
    This helps to quickly change the number of training versus test data.

    This class can extract samples from a dataset in two ways:
    1. store dataset in attribute, get samples from keys
    2. call the instance on
    A dataset can be stored in an attribute either at initialisation, or by
    calling the method `store_dataset`. In this case, the indices are cached
    for efficiency and to ensure that the same subsets are obtained (if the
    dataset does not change). The only consistency check which is performed
    is that the length of the dataset does not change.
    """

    # can decide to shuffle randomly between the different sets

    def __init__(self, ratios, shuffle=False, dataset=None):
        """
        :param ratios: ratios used to divide the data in subsets
        :type ratios: `float`, `list`, `tuple` or `dict`
        :param dataset: data to split in subsets
        :type dataset: `pandas.DataFrame`, `numpy.array`, `dict`
        :param shuffle: shuffle the dataset before extracting samples
        :type shuffle: `bool`
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

        self.shuffle = shuffle

#        if dataset is not None:
#            self.store_dataset(dataset, shuffle)

    def __repr__(self):
        return "<RatioSample: {}>".format(self.ratios)

    def __call__(self, dataset, shuffle=False):
        """
        Split a set of id into subsets according to the class ratios.

        :param dataset: data to split in subsets
        :type dataset: `pandas.DataFrame`, `numpy.array`, `dict`
        :return: subsets arranged in a `dict`
        :rtype: `dict`
        """

        if isinstance(dataset, dict):
            # for a dict, get number of samples by looking at the first
            # key
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

#    def store_dataset(self, dataset):
#
#        self.dataset = dataset
#
#        self._size = len(dataset)
#        self._indices = self.make_samples(self._size)
#
#        if not isinstance(dataset, (pd.DataFrame, np.ndarray, dict)):
#            raise TypeError("Dataset must be a dict, an array or dataframe.")
#
#        if isinstance(dataset, (np.ndarray, dict)):
#            raise NotImplementedError
#
#        self.dataset = dataset
#
#        # cache id for the different subsets
#        self._index_cache = []
