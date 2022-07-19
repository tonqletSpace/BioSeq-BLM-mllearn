from skmultilearn.base import MLClassifierBase
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset
import copy
from scipy.sparse import issparse, lil_matrix, hstack

from .utils_net import TrmDataset


class MllDeepNetSeq(object):
    def __init__(self, mll_type, require_dense=[True, True]):
        self.mll_type = mll_type
        self.require_dense = require_dense

    def mll_classifier(self, base_clf):
        if self.mll_type == 'LP':
            return BLMLabelPowerset(classifier=base_clf, require_dense=self.require_dense)
        if self.mll_type == 'BR':
            return BLMBinaryRelavence(classifier=base_clf, require_dense=self.require_dense)
        # if self.mll_type == 'CC':
        #     return BLMClassifierChain(classifier=base_clf, require_dense=self.require_dense)
        else:
            raise ValueError('err of mll')


class BLMLabelPowerset(LabelPowerset):
    def __init__(self, classifier=None, require_dense=None):
        super(BLMLabelPowerset, self).__init__(
            classifier=classifier, require_dense=require_dense)

    def fit(self, X, y=None):
        ds = None
        if isinstance(X, TrmDataset):
            ds = X
            X = ds.feature
            y = ds.target

        X = self._ensure_input_format(X, sparse_format='csr', enforce_sparse=True)
        X = self._ensure_input_format(X)
        y = self.transform(y)
        self.classifier.fit(get_ds_or_x(ds, X, y), None if ds else y)

        return self

    def predict_proba(self, X):
        ds = None
        if isinstance(X, TrmDataset):
            ds = X
            X = ds.feature

        X = self._ensure_input_format(X)
        lp_prediction = self.classifier.predict_proba(get_ds_or_x(ds, X))

        result = sparse.lil_matrix((X.shape[0], self._label_count), dtype='float')
        # comb_id_result = np.zeros(X.shape[0])
        for row in range(len(lp_prediction)):
            assignment = lp_prediction[row]  # 一个样本的预测概率
            # comb_id_result[row] = np.max(assignment)
            # row_result = np.zeros(self._label_count)
            for combination_id in range(len(assignment)):  # 多分类的每一类
                for label in self.reverse_combinations_[combination_id]:  # label是raw multi label，list of integers中的正项维
                    # row_result[label] += assignment[combination_id]
                    result[row, label] += assignment[combination_id]
            # row_result /= np.sum(row_result)  # norm
            # result[row, :] = row_result
        # return result, comb_id_result  # (N, q) normed probability，combination_id_probability
        return result

    def predict(self, X):
        ds = None
        if isinstance(X, TrmDataset):
            ds = X
            X = ds.feature

        X = self._ensure_input_format(X)
        lp_prediction = self.classifier.predict(get_ds_or_x(ds, X))

        return self.inverse_transform(lp_prediction)


class BLMBinaryRelavence(BinaryRelevance):
    def __init__(self, classifier=None, require_dense=None):
        super(BLMBinaryRelavence, self).__init__(
            classifier=classifier, require_dense=require_dense)

    def fit(self, X, y=None):
        ds = None
        if isinstance(X, TrmDataset):
            ds = X
            X = ds.feature
            y = ds.target

        X = self._ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)
        y = self._ensure_output_format(
            y, sparse_format='csc', enforce_sparse=True)

        self.classifiers_ = []
        self._generate_partition(X, y)
        self._label_count = y.shape[1]

        for i in range(self.model_count_):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self._generate_data_subset(y, self.partition_[i], axis=1)
            if issparse(y_subset) and y_subset.ndim > 1 and y_subset.shape[1] == 1:
                y_subset = np.ravel(y_subset.toarray())
            X = self._ensure_input_format(X)
            y_subset = self._ensure_output_format(y_subset)
            classifier.fit(get_ds_or_x(ds, X, y_subset), None if ds else y_subset)
            self.classifiers_.append(classifier)

        return self

    def predict_proba(self, X):
        ds = None
        if isinstance(X, TrmDataset):
            ds = X
            X = ds.feature

        result = lil_matrix((X.shape[0], self._label_count), dtype='float')
        for label_assignment, classifier in zip(self.partition_, self.classifiers_):
            if isinstance(self.classifier, MLClassifierBase):
                # the multilabel classifier should provide a (n_samples, n_labels) matrix
                # we just need to reorder it column wise
                result[:, label_assignment] = classifier.predict_proba(get_ds_or_x(ds, X))

            else:
                # a base classifier for binary relevance returns
                # n_samples x n_classes, where n_classes = [0, 1] - 1 is the probability of
                # the label being assigned
                result[:, label_assignment] = self._ensure_multi_label_from_single_class(
                    classifier.predict_proba(
                        self._ensure_input_format(get_ds_or_x(ds, X)))
                )[:, 1]  # probability that label is assigned

        return result

    def predict(self, X):
        ds = None
        if isinstance(X, TrmDataset):
            ds = X
            X = ds.feature

        X = self._ensure_input_format(X)
        predictions = [self._ensure_multi_label_from_single_class(
            self.classifiers_[label].predict(get_ds_or_x(ds, X)))
            for label in range(self.model_count_)]

        return hstack(predictions)


# class BLMClassifierChain(ClassifierChain):
#     def __init__(self, classifier=None, require_dense=None, order=None):
#         super(BLMClassifierChain, self).__init__(classifier, require_dense, order)
#
#     def fit(self, X, y=None, order=None):
#         ds = None
#         if isinstance(X, TrmDataset):
#             ds = X
#             X = ds.feature
#             y = ds.target
#
#         X_extended = self._ensure_input_format(X, sparse_format='csc', enforce_sparse=True)
#         y = self._ensure_output_format(y, sparse_format='csc', enforce_sparse=True)
#
#         self._label_count = y.shape[1]
#         self.classifiers_ = [None for x in range(self._label_count)]
#
#         for label in self._order():
#             self.classifier = copy.deepcopy(self.classifier)
#             y_subset = self._generate_data_subset(y, label, axis=1)
#
#             # X_extended =/
#             # y_subset =
#             self.classifier.n
#             self.classifiers_[label] = self.classifier.fit(
#                 get_ds_or_x(ds, self._ensure_input_format(X_extended), self._ensure_output_format(y_subset)),
#                 None if ds else self._ensure_output_format(y_subset))
#             # print(X_extended.shape)
#             # print(y_subset.shape)
#             # print(y_subset.toarray())
#             # exit()
#             X_extended = hstack([X_extended, y_subset])
#
#         return self
#
#     def predict_proba(self, X):  # 原本是
#         ds = None
#         if isinstance(X, TrmDataset):
#             ds = X
#             X = ds.feature
#
#         X_extended = self._ensure_input_format(
#             X, sparse_format='csc', enforce_sparse=True)
#
#         results = []
#         for label in self._order():
#             X_extended = self._ensure_input_format(X_extended)
#             prediction = self.classifiers_[label].predict(get_ds_or_x(ds, X_extended))
#
#             prediction = self._ensure_output_format(
#                 prediction, sparse_format='csc', enforce_sparse=True)
#
#             X_extended = self._ensure_input_format(X_extended)
#             prediction_proba = self.classifiers_[label].predict_proba(get_ds_or_x(ds, X_extended))
#
#             prediction_proba = self._ensure_output_format(
#                 prediction_proba, sparse_format='csc', enforce_sparse=True)[:, 1]
#
#             X_extended = hstack([X_extended, prediction]).tocsc()
#             results.append(prediction_proba)
#
#         return hstack(results)
#
#     def predict(self, X):
#         ds = None
#         if isinstance(X, TrmDataset):
#             ds = X
#             X = ds.feature
#
#         X_extended = self._ensure_input_format(
#             X, sparse_format='csc', enforce_sparse=True)
#
#         for label in self._order():
#             X_extended = self._ensure_input_format(X_extended)
#             prediction = self.classifiers_[label].predict(get_ds_or_x(ds, X_extended))
#             prediction = self._ensure_multi_label_from_single_class(prediction)
#             X_extended = hstack([X_extended, prediction])
#         return X_extended[:, -self._label_count:]


def get_ds_or_x(ds, x, y=None):
    if ds is None:
        return x
    else:
        ds.feature = x
        ds.target = y
        return ds