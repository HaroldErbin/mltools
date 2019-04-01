# -*- coding: utf-8 -*-

from sklearn import svm


class SVMModel:

    def __init__(self, kernel='linear', method='classification'):
        if method == 'classification':
            if kernel == 'linear':
                self.model = svm.LinearSVC
            else:
                self.model = svm.SVC
        elif method == 'regression':
            if kernel == 'linear':
                self.model = svm.LinearSVR
            else:
                self.model = svm.SVR
        else:
            raise ValueError("Method `%s` not permitted." % method)

    def create_model(self, **model_params):
        pass
