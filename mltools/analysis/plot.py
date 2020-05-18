"""
Plotting methods.
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from mltools.analysis.logger import Logger


def distribution(x, x_true=None, x_err=None, sigma=2, plottype='step',
                 density=True, bins=None, range=None, log=False,
                 xlabel=None, label=None, logger=None, filename="",
                 logtime=False):
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
