# -*- coding: utf-8 -*-

from sklearn import svm

from .model import Model


class SVM(Model):

    def __init__(self, kernel='linear', method='clf'):
        self.kernel = kernel
        self.method = method

        if method in ('clf', 'classif', 'classification'):
            if kernel == 'linear':
                self.model = svm.LinearSVC
            else:
                self.model = svm.SVC
        elif method in ('reg', 'regression'):
            if kernel == 'linear':
                self.model = svm.LinearSVR
            else:
                self.model = svm.SVR
        else:
            raise ValueError("Method `%s` not permitted." % method)

    def create_model(self, **model_params):
        pass
