from skmultilearn.problem_transform import LabelPowerset
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset

from .utils_net import TrmDataset


class BLMLabelPowerset(LabelPowerset):
    def __init__(self, classifier=None, require_dense=None):
        super(BLMLabelPowerset, self).__init__(
            classifier=classifier, require_dense=require_dense)

    def fit(self, X, y=None):
        if isinstance(X, TrmDataset):
            ds = X
            inputs = self._ensure_input_format(
                ds.feature, sparse_format='csr', enforce_sparse=True)
            ds.feature = self._ensure_input_format(inputs)
            ds.target = self.transform(ds.target)

            # print("inputs.shape", inputs.shape)
            # print("y.shape", ds.target.shape)
            # print("ds.length.shape", ds.length.shape)
            # print("ds.length[:10]", ds.length[:10])
            # exit()

            self.classifier.fit(ds, None)
        else:
            inputs = self._ensure_input_format(
                X, sparse_format='csr', enforce_sparse=True)

            self.classifier.fit(self._ensure_input_format(inputs),
                                self.transform(y))

        return self

    def predict_proba(self, X):  # 原本是
        if isinstance(X, TrmDataset):
            ds=X
            ds.feature = self._ensure_input_format(ds.feature)
            lp_prediction = self.classifier.predict_proba(ds)
            X=ds.feature
        else:
            lp_prediction = self.classifier.predict_proba(self._ensure_input_format(X))

        result = sparse.lil_matrix((X.shape[0], self._label_count), dtype='float')
        comb_id_result = np.zeros(X.shape[0])
        for row in range(len(lp_prediction)):
            assignment = lp_prediction[row]  # 一个样本的预测概率
            comb_id_result[row] = np.max(assignment)
            row_result = np.zeros(self._label_count)
            for combination_id in range(len(assignment)):  # 多分类的每一类
                for label in self.reverse_combinations_[combination_id]:  # label是raw multi label，list of integers中的正项维
                    row_result[label] += assignment[combination_id]
            row_result /= np.sum(row_result)  # norm
            result[row, :] = row_result

        return result, comb_id_result  # (N, q) normed probability，combination_id_probability

    def predict(self, X):
        if isinstance(X, TrmDataset):
            ds = X
            ds.feature = self._ensure_input_format(ds.feature)
            lp_prediction = self.classifier.predict(ds)
        else:
            lp_prediction = self.classifier.predict(self._ensure_input_format(X))

        return self.inverse_transform(lp_prediction)