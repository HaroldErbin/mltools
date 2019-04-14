# -*- coding: utf-8 -*-


class Logger:

    def __init__(self, logtime=False, base=None):
        # set logtime to a time value, to use the same for all files
        self.logtime = logtime
        # set default path to use when only filename is given
        self.base = base

    def expandpath(self, base=None):
        # different arguments: results, stats...
        if base is None:
            base = self.base

    def savefig(self, fig=None, filename=None, logtime=None):

        # allow to always use the method even when one does not want to save
        if filename is None:
            return

        # None: use default class option; otherwise, force behaviour
        if logtime is None:
            logtime = self.logtime

        # special path for tmp figure (to generate PDF); for example look if
        # filename is a path or just a filename


# custom plot method: take logger instance as argument

