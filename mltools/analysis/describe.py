"""
Plotting methods.
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from mltools.data import datatools as dt

from mltools.analysis.logger import Logger
from mltools.data.structure import DataStructure
from mltools.models.forest import RandomForest


def distribution(x, x_true=None, x_err=None, sigma=2, plottype='step',
                 density=True, bins=None, range=None, log=False,
                 xlabel=None, label=None, filename="", logtime=False,
                 logger=None):
    """
    Plot the distribution of a variable.

    One can add standard deviation (or errors) and a comparative "true"
    distribution.

    The plot type are:
        - `seaborn`: use `seaborn.distplot`
        - `plain`: plot histogram with filled bars using `pyplot.hist`,
          put side by side the histograms if two are given
        - `step`: plot histogram with empty bars using `pyplot.hist`
        - `line`: plot only line

    Errors can be displayed only with `line` and `step` modes. Histograms
    with errors are computed by taking `x ± σ * x_err`. By default this
    shows the 2 sigma region. Note that σ must be an integer.
    In order to show the region associated to a deviation of σ, we compute
    the distributions for `y + 1 * std`, `y + 2 * std`, ..., `y + σ * std`.
    Then we keep the highest and lowest values for each bin. Considering
    only `y + σ * std` would not see all the intermediate variations.

    There is one important caveat: when plotting errors in the `line` mode,
    the upper and lower curves count the highest and lowest numbers of
    values per bin, by comparing the different values obtained for `x` and
     `x ± x_err`. Hence, the area under none of the curves is equal to 1
     when density is true.
     However, the solid curve displayed corresponds to the mean value
     and is correctly a density.

     A `step_legacy` method is available: it plots the histogram using the
     `plt.hist` method (and thus cannot handle errors). It can be used to
     check that the `step` method gives the correct result.
    """

    # TODO: PDF has a problem sometimes (see CICY)

    logger = logger or Logger

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()

    linewidth = 1.5

    alpha = logger.styles["alpha:hist"]
    alpha_err = logger.styles["alpha:err"]
    color_pred = logger.styles["color:pred"]
    color_true = logger.styles["color:true"]

    sigma = int(sigma)

    # compute range
    if range is None and x_true is not None:
        range = [np.min([x, x_true]), np.max([x, x_true])]

    if isinstance(label, list):
        label_pred, label_true = label
    elif isinstance(label, str):
        label_pred = label
        label_true = None
    else:
        label_pred = "pred"
        label_true = "true"
        # label_pred = self.styles["label:pred"]
        # label_true = self.styles["label:true"]

    # TODO: add option for this behaviour?
    bins = bins or logger.find_bins(x)

    # build histogram by hand
    x_hist, edges = np.histogram(x, bins=bins, range=range,
                                 density=density)

    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2

    if x_true is not None:
        x_true_hist, _ = np.histogram(x_true, bins=edges, density=density)

    if x_err is not None:
        x_var = np.array([x + s * x_err
                          for s in np.arange(- sigma, sigma + 1)])

        x_var_hist = np.array([np.histogram(xv, bins=edges,
                                            density=density)[0]
                               for xv in x_var])

        # for each bin, find the lowest and highest occupations
        # this will form the envelope of the graphs
        x_min_hist = np.min(x_var_hist, axis=0)
        x_max_hist = np.max(x_var_hist, axis=0)

    # extend values to left and right to close the graph
    widths = np.r_[widths[0], widths, widths[-1]]
    edges = np.r_[edges[0] - widths[0], edges, edges[-1] + widths[-1]]
    centers = np.r_[centers[0] - widths[0], centers,
                    centers[-1] + widths[-1]]
    x_hist = np.r_[0, x_hist, 0]

    if x_err is not None:
        x_min_hist = np.r_[0, x_min_hist, 0]
        x_max_hist = np.r_[0, x_max_hist, 0]

    if x_true is not None:
        x_true_hist = np.r_[0, x_true_hist, 0]

    if density is True:
        ylabel = "PDF"
    else:
        ylabel = "Count"

    if plottype == 'seaborn':
        # bins = np.min([bins, 50])
        sns.distplot(x, ax=ax, color=color_pred, label=label_pred,
                     bins=bins, norm_hist=density, kde=density,
                     hist_kws={"alpha": alpha})
        if x_true is not None:
            sns.distplot(x_true, ax=ax, color=color_true, label=label_true,
                         bins=bins, norm_hist=density, kde=density,
                         hist_kws={"alpha": alpha})

        if log is True:
            ax.set_yscale("symlog")

    elif plottype in ('plain', 'step_legacy'):
        if x_true is None:
            inputs = x
            color = color_pred
            label = label_pred
        else:
            inputs = [x, x_true]
            color = [color_pred, color_true]
            label = [label_pred, label_true]

        if plottype == 'plain':
            histtype = 'bar'
            lw = 0
            bins = np.min([bins, 30])
        else:
            histtype = 'step'
            lw = linewidth

        rwidth = 0.8

        ax.hist(inputs, color=color, label=label, bins=bins,
                density=density, log=log, histtype=histtype,
                linewidth=lw, rwidth=rwidth)

        # if x_err is not None:
        #     errors = np.c_[x_hist - x_min_hist, x_max_hist - x_hist].T
        #     if plottype == 'plain' and x_true is not None:
        #         # factor 4: two distributions, and center of one
        #         pos = centers - rwidth * widths / 4
        #         ax.errorbar(pos, x_hist, errors, ecolor="grey",
        #                     fmt='none')
        #     else:
        #         ax.errorbar(centers, x_hist, errors, ecolor="grey",
        #                     fmt='none')

    elif plottype == 'step':
        ax.step(centers, x_hist, color=color_pred, label=label_pred,
                where="mid")

        if x_true is not None:
            ax.step(centers, x_true_hist, color=color_true,
                    label=label_true, where="mid")

        if x_err is not None:
            ax.fill_between(edges[:-1], x_min_hist, x_max_hist,
                            alpha=alpha_err, step="post",
                            color=color_pred)

        ax.set_ylim(bottom=0)

        if log is True:
            ax.set_yscale("symlog")

    elif plottype == 'step':

        if x_err is None:
            ax.hist(x, color=color_pred, label=label_pred, bins=bins,
                    density=density, log=log, linewidth=linewidth,
                    histtype='step')
            if x_true is not None:
                ax.hist(x_true, color=color_true, label=label_true,
                        bins=bins, density=density, log=log,
                        linewidth=linewidth, histtype='step')
        else:
            ax.step(centers, x_hist, color=color_pred, where="mid")

            if x_true is not None:
                ax.step(centers, x_true_hist, color=color_true,
                        where="mid")

            ax.fill_between(edges[:-1], x_min_hist, x_max_hist,
                            alpha=alpha_err, step="post", color=color_pred)

            ax.set_ylim(bottom=0)

            if log is True:
                ax.set_yscale("symlog")

    elif plottype == 'line':
        # does this line have a sense when plottign the standard deviation?
        ax.plot(centers, x_hist, linestyle='-', color=color_pred,
                label=label_pred)

        if x_true is not None:
            ax.plot(centers, x_true_hist, linestyle='-', color=color_true,
                    label=label_true)

        if x_err is not None:
            ax.plot(centers, x_min_hist, linestyle='--',
                    alpha=alpha, color=color_pred)
            ax.plot(centers, x_max_hist, linestyle='--',
                    alpha=alpha, color=color_pred)
            ax.fill_between(centers, x_min_hist, x_max_hist,
                            alpha=alpha_err, color=color_pred)

        ax.set_ylim(bottom=0)

        if log is True:
            ax.set_yscale("symlog")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend()

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    fig.tight_layout()

    logger.save_fig(fig, filename, logtime)

    plt.close(fig)

    return fig


def correlations(data, features=None, targets=None, method="pearson",
                 cmap=None, y_rot=45, filename="", logtime=False,
                 logger=None):
    """
    Compute and plot correlations between variables.

    Compute correlations in a symmetric way if only `features` is given.
    If `targets` are given, compute only correlations between `targets` and
    `features`. If `features` is not given, it corresponds to all columns from
    the data.

    The method can be `pearson`, `spearman`, `kendall` or a callable.

    The function is using pandas `.corr()` method. If `data` is a dict, it is
    first converted to a dataframe. To prevent errors, only numerical types
    are conserved before conversion.

    Correlation are valid only for numerical features, thus, the results may
    miss some of the features and targets given as inputs if they are not
    numerical.
    """

    # TODO: check if pandas compute correlation for more general types like
    #       dates

    if method not in ("pearson", "spearman") and callable(method) is False:
        raise TypeError("Method must be `pearson`, `spearman` or callable. "
                        "Found object {}.".format(type(method)))

    all_features = []

    if features is None:

        if isinstance(data, pd.DataFrame):
            features = data.columns.to_list()
        elif isinstance(data, dict):
            # keep only numerical features from dict
            features = dt.filter_features(data,
                                          ("scalar", "integer", "binary"),
                                          ncat=0)
        else:
            raise TypeError("Cannot study correlation for object `{}`."
                            .format(type(data)))

    all_features += features

    if targets is not None and len(targets) > 0:
        all_features += targets
    else:
        targets = features

    if isinstance(data, dict):
        data = pd.DataFrame({k: v for k, v in data.items()
                             if k in all_features})

    corr = data[all_features].corr(method=method)

    # adapt features and targets if they contain columns for which
    # correlations could not be computed
    features = [c for c in corr.index.tolist() if c in features]
    targets = [c for c in corr.columns.tolist() if c in targets]

    if features != targets:
        corr = corr.loc[features, targets]

    fig, ax = plt.subplots()

    if cmap is None:
        cmap = "RdBu_r"

    pcm = ax.matshow(corr, vmin=-1, vmax=1, cmap=cmap)

    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features)

    if y_rot == 90 or y_rot == 0:
        ha = "center"
    else:
        ha = "left"

    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets, rotation=y_rot, ha=ha)

    ax.tick_params(bottom=False)

    # from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig.colorbar(pcm, ax=ax)

    sns.despine(fig=fig, top=True, bottom=True, left=True, right=True)

    fig.tight_layout()

    plt.close(fig)

    if logger is not None:
        logger.save_fig(fig, filename, logtime)

    return corr, fig


def correlation_text(corr):
    """
    Convert correlation dataframe to text.
    """

    corr_fmt = "{:+8.2f}"

    features = corr.index.tolist()
    targets = corr.columns.tolist()

    text = ""

    if features == targets:
        # write coefficients as table

        names = dt.equal_length_names(features, align="right")

        for i, (name, col) in enumerate(zip(names, features)):
            text += name
            text += "".join(corr_fmt.format(v) for j, v
                            in enumerate(corr.loc[col].values) if j <= i)
            text += "\n"
    else:
        # write coefficients as list

        names = dt.equal_length_names(dict(zip(features, features)),
                                             align="left")
        item_fmt = "  ∝ {}  " + corr_fmt

        for target in targets:
            text += "- {}\n".format(target)
            text += "\n".join(item_fmt.format(name, corr[target][col])
                              for name, col in names.items())

            text += "\n\n"

    text = text.strip("\n")

    return text


def importances(data, outputs, inputs=None, n_estimators=20, sum_tensor=False,
                label_rot=45, ymax=1, mode="dataframe", filename="",
                logtime=False, logger=None):
    """
    Compute input importances for outputs from random forest.
    """

    # TODO: adapt figure when there are a lot of features
    # - change size to prevent label overlap
    # - if no feature has high importance, zoom range (or give log scale)

    # TODO: when inputs are categories, insert all names using category struct
    # TODO: generalize when outputs is not a scalar

    if isinstance(outputs, str):
        outputs = [outputs]

    # keep only numerical features
    num_types = ("scalar", "integer", "binary", "tensor")
    num_inputs = dt.filter_features(data, num_types, ncat=0)

    if inputs is None:
        # if not inputs is given, take all columns from data except outputs
        inputs = [c for c in num_inputs if c not in outputs]
    else:
        inputs = [c for c in inputs if c in num_inputs]

    # TODO: if both onehot and ord are present, keep the first only

    # remove NaN values
    # TODO: extend to dict
    if isinstance(data, pd.DataFrame):
        data = data.dropna(how='any')

    # TODO: add random forest for classification

    model_params = {"n_estimators": n_estimators}

    struct = DataStructure(inputs, infer=data)
    importances = {}

    for o in outputs:
        out_struct = DataStructure([o], infer=data)

        model = RandomForest(inputs=struct, outputs=out_struct,
                             method="reg", model_params=model_params)
        model.fit(data)

        importances[o] = model.model.feature_importances_

        if sum_tensor is True:
            named = struct.inverse_transform(importances[o].reshape(1, -1))
            importances[o] = np.array([np.sum(v) for v in named.values()])

    # plot importance graph
    fig, ax = plt.subplots()

    for target in importances:
        ax.plot(importances[target], label=target, marker='.')

    # TODO: put ticks at all places
    if sum_tensor is True:
        ax.set_xticks(np.arange(len(inputs)))
    else:
        # for a tensor, put just one tick at the first component
        ticks = [x[0] for x in dt.linear_indices(struct.shapes.values())]
        ax.set_xticks(ticks)

        minor_ticks = [x for x in range(struct.linear_shape) if x not in ticks]
        ax.set_xticks(minor_ticks, minor=True)

        # TODO: put minor ticks for components

    if label_rot == 90:
        ax.set_xticklabels(inputs, rotation=label_rot, ha="center")
    else:
        ax.set_xticklabels(inputs, rotation=label_rot, ha="right")

    ax.set_ylabel("importance")
    ax.set_ylim(ymin=0, ymax=ymax)

    ax.legend()

    sns.despine(fig=fig)

    fig.tight_layout()

    plt.close(fig)

    if logger is not None:
        logger.save_fig(fig, filename, logtime)

    # convert data to dict with all inputs name or dataframe if requested
    if sum_tensor is True:
        # if components have been summed over, consider only dataframe
        if mode == "dataframe":
            importances = pd.DataFrame(importances, index=inputs)

    else:
        if mode in ("dict", "dataframe"):

            for o, imp in importances.items():
                val = struct.inverse_transform(imp.reshape(1, -1))
                for k, v in val.items():
                    s = np.shape(v)
                    if s == (1,):
                        val[k] = v.item()
                    elif len(s) > 1 and s[0] == 1:
                        val[k] = v.reshape(*s[1:])
                importances[o] = val

        if mode == "dataframe":
            importances = pd.DataFrame({o: pd.Series(v)
                                        for o, v in importances.items()})

    return importances, fig


def importance_text(importances):
    """
    Convert importances series to text.
    """

    float_fmt = "  {:.3f}"
    item_fmt = "- {}   {}\n"
    text = ""

    targets = importances.index.to_list()
    names = dt.equal_length_names(targets, align="left")

    array_padding = " " * (max(map(len, names)) + 2 + 3)

    for name, val in zip(names, importances):
        if isinstance(val, (tuple, list, np.ndarray)):
            val_str = np.array_str(val, precision=3, suppress_small=True)
            if len(np.shape(val)) > 1:
                val_str = val_str.replace("\n", "\n" + array_padding)
            else:
                val_str = " " + val_str
        else:
            val_str = float_fmt.format(val)

        text += item_fmt.format(name, val_str)

    text = text.strip("\n")

    return text
