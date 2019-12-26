"""
Save and log results

The `Logger` class encapsulates all informations necessary to plot and save
the results and the properties of the ML models. It can be passed as an
argument to the functions displaying or saving results.
"""

import os
import time

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import seaborn as sns


# TODO: custom plot method: take logger instance as argument


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

    Attributes:
        logtime (str): time at which the class
    """

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

    def __repr__(self):
        return "<Logger, base = {}, logtime = {}>".format(self.path,
                                                          self.logtime)

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

        fig, ax = plt.subplots()

        # size=12
        ax.text(0, 1, text, fontfamily='monospace', verticalalignment='center')
        ax.set_axis_off()
        # fig.tight_layout(pad=0., w_pad=0., h_pad=0.)

        self.save_fig(fig, filename=filename, logtime=logtime)

        return fig

    @staticmethod
    def dict_to_text(dic, text=""):
        """
        Convert dict to text written as a list.

        If `text` is not empty, then the list is added to the latter.
        """

        # TODO: instead of list save as table (with tab between columns)

        if text != "":
            text += "\n"

        text += "\n".join('- %s = %s' % (k, v) for k, v in dic.items())

        return text
