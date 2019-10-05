"""
Get data samples from a dataset
"""


# TODO: samples based for more complicated cases
# example: one dataset for both training and testing, another just for testing


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
    """

    # TODO: several datasets as arguments?
    # can decide to shuffle randomly between the different sets

    # TODO: accept other types of dataset as arguments (dict)

    def __init__(self, dataset, ratios):
        """
        :param dataset: data to split in subsets
        :type dataset: `pandas.DataFrame`
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

        self.dataset = dataset

        # cache id for the different subsets
        self._index_cache = []
