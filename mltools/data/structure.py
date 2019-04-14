# -*- coding: utf-8 -*-


# TODO:
# TODO: when transforming, if there is a single input (or output), can pass
# single instance (int, etc.) or vec

TYPES = ()
# future: bool, int, scalar, vector, matrix, 3-tensor, text, image, video
#         different types of classification


class DataStructure:

    def __init__(self, inputs=None, outputs=None, datatypes=None,
                 pipeline=None, force_shape=None, scaling=None):

        # pass data through scikit pipeline before fit or transformation
        # implemented by the class
        self.pipeline = pipeline

        # if None, take all columns from dataframe
        if inputs is None:
            self.inputs = []
        else:
            self.inputs = list(inputs)

        self.inputs = inputs

        # if None, take all columns from dataframe
        if outputs is None:
            self.initial_outputs = []
        else:
            self.initial_outputs = list(outputs)

        # TODO: update list of outputs
        # example: after label binazer, or computing moments for vector
        # (need to find how to get all names)
        # use list() to copy
        self.outputs = list(self.initial_outputs)

        # datatypes: scalar, vector, matrix, label, etc.
        # if not given, infer (at the end, all inputs should be given a type)
        # infer shape for tensors, infer shape from data (use biggest shape)
        # (simpler to use than previous methods)
        # force_shape : force a specific shape for some object

        # scaling: None, minmax, std
