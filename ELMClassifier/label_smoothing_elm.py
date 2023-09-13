#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/6 20:58
# @Author : LiFH
# @Site : 
# @File : label_smoothing_elm.py
# @Software: PyCharm


from ELMClassifier.random_hidden_layer import RBFRandomHiddenLayer
from ELMClassifier.elm import ELMClassifier
from sklearn.preprocessing import LabelBinarizer


class ELMClassifierLabelSmooth(ELMClassifier):
    def __init__(self,
                 hidden_layer=RBFRandomHiddenLayer(random_state=0),
                 regressor=None):
        super(ELMClassifierLabelSmooth, self).__init__(hidden_layer, regressor)
        self.binarizer_ = LabelSmoothBinarizer(-1, 1)
        print(self.binarizer_)


class LabelSmoothBinarizer(LabelBinarizer):
    def __init__(self, neg_label=0, pos_label=1, sparse_output=False):
        super(LabelSmoothBinarizer, self).__init__()

    def transform(self, y):
        return label_binarize(y, self.classes_,
                              pos_label=self.pos_label,
                              neg_label=self.neg_label,
                              sparse_output=self.sparse_output)



from collections import defaultdict
import itertools
import array
import warnings

import numpy as np
import scipy.sparse as sp



from sklearn.utils.sparsefuncs import min_max_axis
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _num_samples
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import type_of_target
def label_binarize(y, classes, neg_label=0, pos_label=1, sparse_output=False):

    if not isinstance(y, list):
        # XXX Workaround that will be removed when list of list format is
        # dropped
        y = check_array(y, accept_sparse='csr', ensure_2d=False, dtype=None)
    else:
        if _num_samples(y) == 0:
            raise ValueError('y has 0 samples: %r' % y)
    if neg_label >= pos_label:
        raise ValueError("neg_label={0} must be strictly less than "
                         "pos_label={1}.".format(neg_label, pos_label))

    if (sparse_output and (pos_label == 0 or neg_label != 0)):
        raise ValueError("Sparse binarization is only supported with non "
                         "zero pos_label and zero neg_label, got "
                         "pos_label={0} and neg_label={1}"
                         "".format(pos_label, neg_label))

    # To account for pos_label == 0 in the dense case
    pos_switch = pos_label == 0
    if pos_switch:
        pos_label = -neg_label

    y_type = type_of_target(y)
    if 'multioutput' in y_type:
        raise ValueError("Multioutput target data is not supported with label "
                         "binarization")
    if y_type == 'unknown':
        raise ValueError("The type of target data is not known")

    n_samples = y.shape[0] if sp.issparse(y) else len(y)
    n_classes = len(classes)
    classes = np.asarray(classes)

    if y_type == "binary":
        if n_classes == 1:
            if sparse_output:
                return sp.csr_matrix((n_samples, 1), dtype=int)
            else:
                Y = np.zeros((len(y), 1), dtype=np.int)
                Y += neg_label
                return Y
        elif len(classes) >= 3:
            y_type = "multiclass"

    sorted_class = np.sort(classes)
    if (y_type == "multilabel-indicator" and classes.size != y.shape[1]):
        raise ValueError("classes {0} missmatch with the labels {1}"
                         "found in the data".format(classes, unique_labels(y)))

    if y_type in ("binary", "multiclass"):
        y = column_or_1d(y)
        # pick out the known labels from y
        y_in_classes = np.in1d(y, classes)
        y_seen = y[y_in_classes]
        indices = np.searchsorted(sorted_class, y_seen)
        indptr = np.hstack((0, np.cumsum(y_in_classes)))

        data = np.empty_like(indices)
        data.fill(pos_label)
        Y = sp.csr_matrix((data, indices, indptr),
                          shape=(n_samples, n_classes))
    elif y_type == "multilabel-indicator":
        Y = sp.csr_matrix(y)
        if pos_label != 1:
            data = np.empty_like(Y.data)
            data.fill(pos_label)
            Y.data = data
    else:
        raise ValueError("%s target data is not supported with label "
                         "binarization" % y_type)

    if not sparse_output:
        Y = Y.toarray()
        Y = Y.astype(int, copy=False)
        if neg_label != 0:
            Y[Y == 0] = neg_label

        if pos_switch:
            Y[Y == pos_label] = 0
    else:
        Y.data = Y.data.astype(int, copy=False)

    # preserve label ordering
    if np.any(classes != sorted_class):
        indices = np.searchsorted(sorted_class, classes)
        Y = Y[:, indices]

    if y_type == "binary":
        if sparse_output:
            Y = Y.getcol(-1)
        else:
            Y = Y[:, -1].reshape((-1, 1))
    lb_smooth = 1
    num_classes = 101
    Y[Y == 0] = lb_smooth / num_classes
    Y[Y == 1] = 1 -lb_smooth

    return Y



