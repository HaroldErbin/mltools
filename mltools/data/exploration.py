"""
Explore data.
"""


class DataExploration:

    def __init__(self, inputs=None, outputs=None):
        # if no inputs, use all columns except the ones in outputs

        pass

    def describe(self, features=None):
        # set features restrict output

        # general statistics on dataset
        # add percentiles 10% and 90%
        pass

    def distribution(self, features=None):
        pass

    def correlations(self, features=None, outputs=None):
        # if outputs is None, display all graphs
        # if it is true, only graphs for the outputs with other variables
        # compute also coefficients

        pass

    def scatter_plots(self, features=None, outputs=None):
        pass

    def feature_importance(self, features=None, outputs=None):
        pass

    def baseline(self, features=None, outputs=None, models=None):
        # if model is None, run: linear regression, SVM, basic neural network
        # random forest

        pass
