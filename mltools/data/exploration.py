"""
Explore data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataExploration:

    def __init__(self, inputs=None, outputs=None):
        # if no inputs, use all columns except the ones in outputs

        if inputs is None:
            self.inputs = []
        else:
            self.inputs = inputs

        if outputs is None:
            self.outputs = []
        else:
            self.outputs = outputs

        self.features = self.inputs + self.outputs

    def _prepare(self, data, features):
        features = features or self.features

        if isinstance(features, str):
            features = [features]

        if features is None or len(features) == 0:
            if isinstance(data, dict):
                features = list(data.keys())
            elif isinstance(data, pd.DataFrame):
                features = list(data.columns)

        if isinstance(data, dict):
            data = pd.DataFrame({k: v for k, v in data.items()
                                 if k in features})
        elif isinstance(data, pd.DataFrame):
            data = data[features]
        else:
            raise TypeError("Data with type `{}` cannot be explored."
                            .format(type(data)))

        return data, features

    def summary(self, data, features=None, display_text=False,
                display_img=False, filename=None):

        print("# Dataset informations")
        self.info(data, features)
        print()

        text = self.describe(data, features)
        print("# Dataset statistics (numerical)")
        print(text)
        print()

        self.distribution(data, features, bins=50, figsize=(10, 10))
        print()

        self.correlations(data, features)
        print()

    def info(self, data, features=None):
        data, features = self._prepare(data, features)

        # use buf to get output

        data.info()

    def describe(self, data, features=None):
        # use include to also describe categories, string (separate display)

        data, features = self._prepare(data, features)

        text = data.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])

        return text

    def distribution(self, data, features=None, bins=50, figsize=(20, 15)):
        data, features = self._prepare(data, features)

        data.hist(bins=bins, figsize=figsize)

    def correlations(self, data, features=None):

        # TODO: move
        def display_corr(corr, labels):
            axes = plt.matshow(corr, vmin=-1, vmax=1)
            plt.colorbar()

            fig = axes.get_figure()

            # TODO: put ticks below?
            ax = fig.axes[0]
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticklabels(labels)

            return fig

        data, features = self._prepare(data, features)

        corr = data.corr()
        fig = display_corr(corr, features)

        return features

    def io_correlations(self, data, inputs=None, outputs=None):
        """
        Compute correlations between inputs and outputs.
        """

        # TODO: move
        def display_corr(corr, labels):
            axes = plt.matshow(corr, vmin=-1, vmax=1)
            plt.colorbar()

            fig = axes.get_figure()

            # TODO: put ticks below?
            ax = fig.axes[0]
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticklabels(labels)

            return fig

        features = []

        inputs = inputs or self.inputs
        outputs = outputs or self.outputs

        if len(inputs) == 0 or len(outputs) == 0:
            raise ValueError("Inputs and outputs must be non-empty.")

        data, features = self._prepare(data, inputs + outputs)

        corr = data.corr()

        corr_in = corr[inputs].loc[inputs]
        corr_out = corr[outputs].loc[outputs]

        if len(inputs) > 1:
            fig_in = display_corr(corr_in, inputs)
        if len(outputs) > 1:
            fig_out = display_corr(corr_out, outputs)

        for out in outputs:
            print("# Correlation with inputs: {}".format(out))
            print(corr[out][inputs].sort_values(ascending=False))
            print()

    def scatter_plots(self, data, inputs=None, outputs=None):

        pass

#        data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
#        pd.plotting.scatter_matrix(data)

    def feature_importance(self, data, inputs=None, outputs=None):
        pass

    def baseline(self, data, inputs=None, outputs=None, models=None):
        # if model is None, run: linear regression, SVM, basic neural network
        # random forest

        pass
