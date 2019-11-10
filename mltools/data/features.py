"""
Generate new features.
"""

import pandas as pd

from sklearn.preprocessing import LabelBinarizer, LabelEncoder


class CategoricalFeatures:
    """
    Provide maps for categorical features.
    """

    def __init__(self, categories=None):

        # TODO: add other types of categories

        if categories is None:
            self.categories = []
        else:
            self.categories = categories

        self.features = self.categories
        self.encoders = {}

    def fit(self, data):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError

    def fit_transform(self, data):
        raise NotImplementedError

    def fit_category(self, data, mode=None):

        # TODO: add possibility to restrict modes (to avoid creating too
        #       much data)

        if mode is not None and mode not in ["ord", "ordinal", "onehot"]:
            raise ValueError("Mode `{}` does not exist.".format(mode))

        for f in self.categories:
            self.encoders[f] = {}

            # TODO: map string to integers

            # ordinal
            enc = LabelEncoder()
            enc.fit(data[f])
            self.encoders[f]["ord"] = enc

            enc = LabelBinarizer()
            enc.fit(data[f])
            self.encoders[f]["onehot"] = enc

    def make_category(self, data, mode=None):

        # TODO: add possibility to restrict modes (to avoid creating too
        #       much data)

        if mode is not None and mode not in ["ord", "ordinal", "onehot"]:
            raise ValueError("Mode `{}` does not exist.".format(mode))

        new = {}

        for f in self.categories:
            new[f + "_ord"] = self.encoders[f]["ord"].transform(data[f])
            new[f + "_onehot"] = self.encoders[f]["onehot"].transform(data[f])

        return new

    def update_dataframe(self, data):
        new = self.make_category(data)

        return data.join(pd.DataFrame({k: list(v) for k, v in new.items()}))
