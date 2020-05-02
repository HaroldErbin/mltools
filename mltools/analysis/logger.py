"""
Save and log results

The `Logger` class encapsulates all informations necessary to plot and save
the results and the properties of the ML models. It can be passed as an
argument to the functions displaying or saving results.
"""

import os
import time
import csv
import json

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


# TODO: prevent display of graphs

# TODO: custom plot method: take logger instance as argument


STYLES = {"color:true": "tab:blue",
          "color:pred": "tab:green",
          "color:train": "tab:blue",
          "color:val": "tab:red",
          "color:test": "tab:purple",
          "color:errors": "tab:blue",
          "label:true": "true",
          "label:pred": "pred",
          "label:train": "train",
          "label:val": "validation",
          "label:test": "test",
          "print:float": "{:.4f}",
          "print:percent": "{:.2%}",
          "print:datetime": "%Y-%m-%d %H:%M:%S",
          "save:float": "% .5g",
          # alpha parameter for displaying histograms
          "alpha:hist": 0.3,
          # alpha parameter for displaying errors
          "alpha:err": 0.2
          }


def inserttofilename(filename, text=""):
    """
    Insert some text in filename before the extension.

    Args:
        filename (str): filename, possibly including path
        text (str): text to insert in the filename before the extension

    Returns:
        str: original filename with `text` inserted before the extension.
    """

    name, ext = os.path.splitext(filename)

    return name + text + ext


class Logger:
    """
    Store informations to display and save results.

    The class contains a `styles` dictionary which describes the default value
    for various styling parameters. It can be updated per instance. Using it
    through the class `Logger` always gives the default values.

    Attributes:
        logtime (str): time at which the class
    """

    styles = STYLES

    def __init__(self, path="", logtime="folder",
                 logtime_fmt="%Y-%m-%d-%H%M%S"):
        """
        Inits Logger

        Args:
            path (str): base path where results are saved.
            logtime (str): indicate if time is inserted in the path. There
                are two (non-exclusive) possibilities:
                - if "filename" is in `logtime`: insert time before the
                  extension
                - if "folder" is in `logtime`: create folder named by time
                If none of these two cases is found, time is not logged.
                Default to "folder".
            logtime_fmt (str): time format.
        """

        self.styles = STYLES.copy()

        # set base path to use when only filename is given
        self.path = os.path.abspath(path)

        if "folder" in logtime:
            self._time_folder = True
        else:
            self._time_folder = False

        if "filename" in logtime:
            self._time_filename = True
        else:
            self._time_filename = False

        # set logtime to a fixed value, which is used for all files
        self.logtime = time.strftime(logtime_fmt)
        self.logtime_fmt = logtime_fmt

    def __repr__(self):
        return "<Logger, base = {}, logtime = {}>".format(self.path,
                                                          self.logtime)

    def logtime_text(self, fmt=None):
        fmt = fmt or self.styles["print:datetime"]

        return time.strftime(fmt,
                             time.strptime(self.logtime, self.logtime_fmt))

    def expandpath(self, filename="", logtime=True):
        """
        Find complete path.

        If `filename` is a relative path, prepend with the base path. Note
        that one can use a filename containing `..`.

        Time is added in filename and/or path as specified at initialisation.
        If `filename` is an absolute path, a folder is created directly on
        top of the file. On the other hand, time is added as a folder between
        the the base path and the relative path.
        Time logging can be disabled by setting the argument `logtime` to
        False. Note that setting it to True has no effect if time logging is
        disabled at initialisation.

        Folders are created recursively if they do not exist.

        Args:
            filename (str): filename with relative or absolute path.
            logtime (bool): if False disable time logging.

        Returns:
            Return filename with complete path. This includes the base path
            if the filename was relative and logtime if necessary.
        """

        # check if path is absolute
        if os.path.isabs(filename):
            # insert time folder if time logging is enabled
            if logtime is True and self._time_folder is True:
                head, tail = os.path.split(filename)
                filepath = os.path.join(head, self.logtime, tail)
            else:
                filepath = filename
        else:
            # insert time folder if time logging is enabled
            if logtime is True and self._time_folder is True:
                filepath = os.path.join(self.path, self.logtime, filename)
            else:
                filepath = os.path.join(self.path, filename)

        # insert time at end of filename, before extension
        if logtime is True and self._time_filename is True:
            filepath = inserttofilename(filepath, self.logtime)

        # check if folder exists, if not, create it
        folder = os.path.split(filepath)[0]
        if os.path.exists(folder) is False:
            os.makedirs(folder)

        return filepath

    def save_fig(self, fig=None, filename="", logtime=True, dpi=300):
        """
        Save figure

        Args:
            fig (figure): Figure to save in PDF. If None, grab the current
                figure from Matplotlib. Defaults to None.
            filename (str): Filename used to save the figure. If empty, the
                figure is not saved. Defaults to "".
            logtime (bool): Disable time logging if False. Defaults to True.
            dpi (int): Number of dpi to be used. Defaults to 300.
        """

        # TODO: special path for tmp figure (to generate PDF)

        # allow to always use the method even when one does not want to save
        if filename == "":
            return

        filename = self.expandpath(filename, logtime)

        if fig is None:
            fig = plt.getgcf()

        fig.savefig(filename, dpi=dpi, bbox_inches='tight')

    def save_figs(self, figs, filename="", logtime=True, dpi=300):
        """
        Save several figures

        Args:
            figs (list[figure]): Figures to save as a multipage PDF.
            filename (str): Filename used to save the PDF. If empty, the
                figure is not saved. Defaults to "".
            logtime (bool): Disable time logging if False. Defaults to True.
            dpi (int): Number of dpi to be used. Defaults to 300.
        """

        if filename == "":
            return

        filename = self.expandpath(filename, logtime)

        with PdfPages(filename) as pdf:
            for fig in figs:
                pdf.savefig(fig, dpi=dpi, bbox_inches='tight')

    def save_text(self, text, filename="", logtime=True):

        if filename == "":
            return

        filename = self.expandpath(filename, logtime)

        with open(filename, 'w') as f:
            f.write(text)

    def save_csv(self, data, sep='\t', float_fmt=None, filename="",
                 logtime=True):
        """
        Save data in CSV file.

        For dict and dataframe, use Pandas' method. Float are formatted
        according to the format given in `float_fmt` or in the styles dict.
        To keep all the digits, set the value in `styles` to None.

        If data is a list or a tuple, this calls the standard csv module
        without any formatting.
        """

        # TODO: add gzip compression

        if filename == "":
            return

        filename = self.expandpath(filename, logtime)

        float_fmt = float_fmt or self.styles["save:float"]

        if isinstance(data, dict):
            data = pd.DataFrame(data)

        if isinstance(data, (pd.DataFrame, pd.Series)):
            data.to_csv(filename, sep=sep, float_format=float_fmt)
        elif isinstance(data, (tuple, list)):
            with open(filename, 'w') as f:
                csv.writer(f).writerows(data)
        else:
            raise TypeError("Data with type `{}` cannot be saved to csv."
                            .format(type(data)))

    def save_json(self, data, filename="", logtime=True):

        if filename == "":
            return

        filename = self.expandpath(filename, logtime)

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def text_to_fig(self, text, filename="", logtime=True):
        """
        Convert text to figure.

        This is useful to make a single PDF summary with Matplotlib.

        Args:
            text (str): text to convert to string.
            filename (str): file to save the figure. Defaults to "".
            logtime (bool): If False, disable time logging. Defaults to True.

        Returns:
            fig (figure): figure containing the text.
        """

        # tabs cannot be read from matplotlib
        text = text.replace('\t', '  ')

        fig, ax = plt.subplots()

        # size=12
        ax.text(0, 1, text, fontfamily='monospace', verticalalignment='center')
        ax.set_axis_off()

        ax.margins(0, 0)
        fig.tight_layout(pad=10)

        self.save_fig(fig, filename=filename, logtime=logtime)

        return fig

    @staticmethod
    def dict_to_text(dic, text="", sep=" ="):
        """
        Convert dict to a text written as a list.

        This works recursively, indenting sublist.

        If `text` is not empty, then the list is added to the latter.
        """

        # TODO: write also function to convert to table

        if text != "":
            text += "\n"

        for k, v in dic.items():
            if isinstance(v, dict):
                text += "- %s\n\t" % k
                text += Logger.dict_to_text(v).replace('\n', '\n\t') + "\n"
            else:
                text += "- {}{} {}\n".format(k, sep, v)

        return text[:-1]

    @staticmethod
    def find_bins(data):
        """
        Find the number of bins appropriate for data.

        This uses the rule from [Skiena, 2017]: bins = max(floor(n/25), 100),
        where n is the number of samples.
        """

        return min(int(len(data) / 25), 100)

    def dist(self, x, x_true=None, x_err=None, sigma=2, plottype='step',
             density=True, bins=None, range=None, log=False,
             xlabel=None, label=None, filename="", logtime=False):
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
        shows the 2 sigma region.

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

        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots()

        linewidth = 1.5

        alpha = self.styles["alpha:hist"]
        alpha_err = self.styles["alpha:err"]
        color_pred = self.styles["color:pred"]
        color_true = self.styles["color:true"]

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
        bins = bins or self.find_bins(x)

        # build histogram by hand
        x_hist, edges = np.histogram(x, bins=bins, range=range,
                                     density=density)

        widths = np.diff(edges)
        centers = (edges[:-1] + edges[1:]) / 2

        if x_true is not None:
            x_true_hist, _ = np.histogram(x_true, bins=edges, density=density)

        if x_err is not None:
            x_low = x - sigma * x_err
            x_up = x + sigma * x_err

            x_low_hist, _ = np.histogram(x_low, bins=edges, density=density)
            x_up_hist, _ = np.histogram(x_up, bins=edges, density=density)

            x_values = np.c_[x_hist, x_low_hist, x_up_hist]
            n = len(x_hist)

            # for each bin, find the lowest and highest occupations
            # this will form the envelope of the graphs
            # the remaining values is used to plot the intermediate line
            # (or box)
            # NOTE: we have to do it by steps to avoid problems when all
            # entries are identical (if min = max, one recovers the same index)
            min_args = np.argmin(x_values, axis=1)
            x_min_hist = x_values[np.arange(n), min_args]

            mask = np.ones(np.shape(x_values), dtype=bool)
            mask[np.arange(n), min_args] = False
            x_values = x_values[mask].reshape(n, -1)

            max_args = np.argmax(x_values, axis=1)
            x_max_hist = x_values[np.arange(n), max_args]

            # NOTE: change this to plot the intermediate line in
            #   density histogram (but this is is NOT a density)
            #   if commented, plot the the real distribution
            # mask = np.ones(np.shape(x_values), dtype=bool)
            # mask[np.arange(n), max_args] = False
            # x_hist = x_values[mask]

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

        self.save_fig(fig, filename, logtime)

        return fig
