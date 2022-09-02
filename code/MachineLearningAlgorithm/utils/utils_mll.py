import shlex
import subprocess
import sys

import torch
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.adapt import MLkNN, BRkNNaClassifier, BRkNNbClassifier, MLARAM
from skmultilearn.base import MLClassifierBase
from skmultilearn.cluster import RandomLabelSpaceClusterer
from skmultilearn.cluster.base import LabelSpaceClustererBase
from skmultilearn.ensemble import RakelO, MajorityVotingClassifier, RakelD
from skmultilearn.ext import Meka
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
import numpy as np
from scipy import sparse
from skorch import NeuralNetClassifier
from torch import nn, optim
from torch.utils.data import Dataset
import copy
from scipy.sparse import issparse, lil_matrix, hstack

from .utils_net import TrmDataset, MllBaseTorchNetSeq

Mll_Instance_Based_Methods = ['MLkNN', 'BRkNNaClassifier', 'BRkNNbClassifier']
Mll_MEKA_Methods = ['CLR', 'FW', 'RT']
Mll_ENSEMBLE_Methods = ['RAkELo', 'RAkELd']


def get_lp_num_class(y):
    if issparse(y):
        if not isinstance(y, lil_matrix):
            y = lil_matrix(y)
    else:
        raise ValueError('y should be scipy sparse')
    unique_combinations_ = {}

    last_id = 0
    for labels_applied in y.rows:
        label_string = ",".join(map(str, labels_applied))

        if label_string not in unique_combinations_:
            unique_combinations_[label_string] = last_id
            last_id += 1

    return last_id  # output space {0,1,...,n_class-1}


def get_mll_deep_model(num_class, mll, ml, max_len, embed_size, params_dict):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

    if num_class is not None:
        initialized_torch_model = MllBaseTorchNetSeq(
            ml, max_len, None, params_dict).net_type(embed_size, num_class)
        base_clf = NeuralNetClassifier(
            initialized_torch_model,
            criterion=nn.CrossEntropyLoss(),
            batch_size=params_dict['batch_size'],
            optimizer=optim.Adam,
            optimizer__lr=params_dict['lr'],
            device=DEVICE,
            max_epochs=params_dict['epochs'],
            iterator_train__shuffle=True,
            train_split=False
        )
        require_dense = [True, True]
    else:
        base_clf = None
        require_dense = [False, False]

    mll_clf = MllDeepNetSeq(mll, params_dict, require_dense).mll_classifier(base_clf)

    return mll_clf


class MllDeepNetSeq(object):
    def __init__(self, mll_type, params=None, require_dense=[True, True]):
        self.mll_type = mll_type
        self.require_dense = require_dense
        self.params = params

    def mll_classifier(self, base_clf):
        if self.mll_type == 'LP':
            return BLMLabelPowerset(classifier=base_clf, require_dense=self.require_dense)
        elif self.mll_type == 'BR':
            return BLMBinaryRelevance(classifier=base_clf, require_dense=self.require_dense)
        # if self.mll_type == 'CC':
        #     return BLMClassifierChain(classifier=base_clf, require_dense=self.require_dense)
        elif self.mll_type == 'RAkELo':
            return BLMRAkELo(base_classifier=None,
                             base_classifier_require_dense=self.require_dense,
                             model_count=self.params['RAkELo_model_count'],
                             labelset_size=self.params['RAkEL_labelset_size'])
        elif self.mll_type == 'RAkELd':
            return BLMRAkELd(
                base_classifier=None,
                base_classifier_require_dense=self.require_dense,
                labelset_size=self.params['RAkEL_labelset_size']
            )
        else:
            raise ValueError('err of mll MllDeepNetSeq methods')


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


class BLMBinaryRelevance(BinaryRelevance):
    def __init__(self, classifier=None, require_dense=None):
        super(BLMBinaryRelevance, self).__init__(
            classifier=classifier, require_dense=require_dense)

    def fit(self, X, y=None, mll=None, ml=None, max_len=None, embed_size=None, params_dict=None):
        ds = None
        if isinstance(X, TrmDataset):
            ds = X
            X = ds.feature
            y = ds.target

        X = self._ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)
        y = self._ensure_output_format(
            y, sparse_format='csc', enforce_sparse=True)

        # exit()
        self.classifiers_ = []
        self._generate_partition(X, y)
        self._label_count = y.shape[1]

        for i in range(self.model_count_):
            y_subset = self._generate_data_subset(y, self.partition_[i], axis=1)
            if issparse(y_subset) and y_subset.ndim > 1 and y_subset.shape[1] == 1:
                y_subset = np.ravel(y_subset.toarray())

            X = self._ensure_input_format(X)
            y_subset = self._ensure_output_format(y_subset)

            if issparse(y_subset) and y_subset.ndim > 1 and y_subset.shape[1] == 1\
                    and params_dict is None:
                # BR clf
                classifier = copy.deepcopy(self.classifier)
            else:
                # ensemble clf
                if max_len is not None and embed_size is not None:
                    lp_args = mll, ml, max_len, embed_size, params_dict
                    classifier = get_mll_deep_model(get_lp_num_class(y_subset), *lp_args)  # dl
                else:
                    classifier = get_mll_ml_model(mll, ml, params_dict)  # ml

            # dl 的 model 要在fit前根据y来确定num_class
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


class BLMMeka(Meka):
    def __init__(self, meka_classifier=None, weka_classifier=None,
                 java_command=None, meka_classpath=None):
        super(BLMMeka, self).__init__(meka_classifier, weka_classifier,
                                   java_command, meka_classpath)

    def _run_meka_command(self, args):
        command_args = [
            self.java_command,
            '-cp', '"{}*"'.format(self.meka_classpath),
            self.meka_classifier,
        ]

        if self.weka_classifier is not None:
            # acadTags issue #198
            weka_classfier_name = self.weka_classifier.split(' ')[0]
            weka_classfier_param = ' '.join(self.weka_classifier.split(' ')[1:])
            command_args += ['-W', weka_classfier_name]
        else:
            weka_classfier_param = ''

        command_args += args

        # acadTags issue #198
        if weka_classfier_param != '':
            command_args += ['--', weka_classfier_param]

        meka_command = " ".join(command_args)
        # print(meka_command)

        if sys.platform != 'win32':
            meka_command = shlex.split(meka_command)

        pipes = subprocess.Popen(meka_command,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True)
        self.output_, self._error = pipes.communicate()
        if type(self.output_) == bytes:
            self.output_ = self.output_.decode(sys.stdout.encoding)
        if type(self._error) == bytes:
            self._error = self._error.decode(sys.stdout.encoding)

        if pipes.returncode != 0:
            raise Exception(self.output_ + self._error)


# RakelO start
class BLMRAkELo(RakelO):
    def __init__(self, base_classifier=None, model_count=None, labelset_size=3, base_classifier_require_dense=None):
        super(BLMRAkELo, self).__init__(
            base_classifier=base_classifier,
            model_count=model_count,
            labelset_size=labelset_size,
            base_classifier_require_dense=base_classifier_require_dense)

    def fit(self, X, y=None, mll=None, ml=None, max_len=None, embed_size=None, params_dict=None):
        self.classifier = BLMMajorityVotingClassifier(
            classifier=None,
            clusterer=RandomLabelSpaceClusterer(
                cluster_size=self.labelset_size,
                cluster_count=self.model_count,
                allow_overlap=True
            ),
            require_dense=[False, False]
        )

        return self.classifier.fit(
            X, y, mll, ml, max_len, embed_size, params_dict)  # BR fit

    def predict(self, X):
        return self.classifier.predict(X)


class BLMLabelSpacePartitioningClassifier(BLMBinaryRelevance):
    def __init__(self, classifier=None, clusterer=None, require_dense=None):
        super(BLMLabelSpacePartitioningClassifier, self).__init__(classifier, require_dense)
        self.clusterer = clusterer
        self.copyable_attrs = ['clusterer', 'classifier', 'require_dense']

    def predict(self, X):
        ds = None
        if isinstance(X, TrmDataset):
            ds = X
            X = ds.feature

        X = self._ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)
        result = sparse.lil_matrix((X.shape[0], self._label_count), dtype=int)

        for model in range(self.model_count_):
            predictions = self._ensure_output_format(self.classifiers_[model].predict(
                get_ds_or_x(ds, X)), sparse_format=None, enforce_sparse=True).nonzero()
            for row, column in zip(predictions[0], predictions[1]):
                result[row, self.partition_[model][column]] = 1

        return result

    def _generate_partition(self, X, y):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        self.partition_ = self.clusterer.fit_predict(X, y)
        self.model_count_ = len(self.partition_)
        self._label_count = y.shape[1]
        return self


class BLMMajorityVotingClassifier(BLMLabelSpacePartitioningClassifier):
    def __init__(self, classifier=None, clusterer=None, require_dense=None):
        super(BLMMajorityVotingClassifier, self).__init__(
            classifier=classifier, clusterer=clusterer, require_dense=require_dense
        )

    def predict(self, X):
        ds = None
        if isinstance(X, TrmDataset):
            ds = X
            X = ds.feature

        predictions = [
            self._ensure_input_format(self._ensure_input_format(
                c.predict(get_ds_or_x(ds, X))), sparse_format='csc', enforce_sparse=True)
            for c in self.classifiers_
        ]

        voters = np.zeros(self._label_count, dtype='int')
        votes = sparse.lil_matrix(
            (predictions[0].shape[0], self._label_count), dtype='int')
        for model in range(self.model_count_):
            for label in range(len(self.partition_[model])):
                votes[:, self.partition_[model][label]] = votes[
                                                         :, self.partition_[model][label]] + predictions[model][:, label]
                voters[self.partition_[model][label]] += 1

        nonzeros = votes.nonzero()
        for row, column in zip(nonzeros[0], nonzeros[1]):
            votes[row, column] = np.round(
                votes[row, column] / float(voters[column]))

        return self._ensure_output_format(votes, enforce_sparse=False)

    def predict_proba(self, X):
        raise NotImplemented("The voting scheme does not define a method for calculating probabilities")
# RakelO end


# RakelD
class BLMRAkELd(RakelD):
    def __init__(self, base_classifier=None, labelset_size=3, base_classifier_require_dense=None):
        super(BLMRAkELd, self).__init__(
            base_classifier=base_classifier,
            labelset_size=labelset_size,
            base_classifier_require_dense=base_classifier_require_dense
        )

    def fit(self, X, y, mll=None, ml=None, max_len=None, embed_size=None, params_dict=None):
        ds = None
        if isinstance(X, TrmDataset):
            ds = X
            X = ds.feature
            y = ds.target

        self._label_count = y.shape[1]
        self.model_count_ = int(np.ceil(self._label_count / self.labelset_size))
        self.classifier_ = BLMLabelSpacePartitioningClassifier(
            classifier=None,
            clusterer=RandomLabelSpaceClusterer(
                cluster_size=self.labelset_size,
                cluster_count=self.model_count_,
                allow_overlap=False
            ),
            require_dense=[False, False]
        )
        return self.classifier_.fit(
            get_ds_or_x(ds, X, y), None if ds else y,
            mll, ml, max_len, embed_size, params_dict)

    def predict(self, X):
        return self.classifier_.predict(X)


def get_ds_or_x(ds, x, y=None):
    """

    :param ds: torch.utils.data.Dataset or None
    :param x: sparse matrix, [n_samples, n_features]
    :param y: sparse matrix, [n_samples, n_features]
    :return: torch.utils.data.Dataset or x
    """
    if ds is None:
        return x
    else:
        # dl methods 的 feature 不能是 sparse matrix
        ds.feature = x.toarray() if issparse(x) else x
        ds.target = y
        return ds


def is_mll_instance_methods(mll):
    return mll in Mll_Instance_Based_Methods


def is_mll_meka_methods(mll):
    return mll in Mll_MEKA_Methods


def is_mll_ensemble_methods(mll):
    return mll in Mll_ENSEMBLE_Methods


def is_mll_proba_output_methods(mll):
    return not is_mll_instance_methods(mll) \
           and not is_mll_meka_methods(mll) \
           and not is_mll_ensemble_methods(mll)


def mll_sparse_check(mll, *data_list):
    if mll in ['MLARAM']:
        data_list = [e.tocsc() for e in data_list]

    return data_list


def mll_result_sparse_check(mll, res):
    if mll in ['MLARAM'] and not issparse(res):
        return lil_matrix(res)
    return res


def get_mll_ml_model(mll, ml, params_dict):
    if ml:
        return mll_ml_model_factory(mll, ml, params_dict)
    else:
        return mll_model_factory(mll, params_dict)


def mll_ml_model_factory(mll, ml, params_dict):
    if mll == 'BR':
        return BinaryRelevance(classifier=ml_model_factory(ml, params_dict), require_dense=[True, True])
    elif mll == 'CC':
        return ClassifierChain(classifier=ml_model_factory(ml, params_dict), require_dense=[True, True])
    elif mll == 'LP':
        return LabelPowerset(classifier=ml_model_factory(ml, params_dict), require_dense=[True, True])
    elif mll == 'RAkELo':
        return BLMRAkELo(
            base_classifier=None,
            base_classifier_require_dense=[True, True],
            labelset_size=params_dict['RAkEL_labelset_size'],
            model_count=params_dict['RAkELo_model_count']
        )
    elif mll == 'RAkELd':
        return BLMRAkELd(
            base_classifier=None,
            base_classifier_require_dense=[True, True],
            labelset_size=params_dict['RAkEL_labelset_size']
        )
    elif mll == 'FW':
        return BLMMeka(
            meka_classifier="meka.classifiers.multilabel.FW",  # Binary Relevance
            weka_classifier=ml_model_factory(ml, params_dict, True),  # with Naive Bayes single-label classifier
            meka_classpath=params_dict['meka_classpath'],  # obtained via download_meka
            java_command=params_dict['which_java']
        )
    elif mll == 'RT':
        return BLMMeka(
            meka_classifier="meka.classifiers.multilabel.RT",  # Binary Relevance
            weka_classifier=ml_model_factory(ml, params_dict, True),  # with Naive Bayes single-label classifier
            meka_classpath=params_dict['meka_classpath'],  # obtained via download_meka
            java_command=params_dict['which_java']
        )
    elif mll == 'CLR':
        return BLMMeka(
            meka_classifier="meka.classifiers.multilabel.MULAN -S CLR",  # Binary Relevance
            weka_classifier=ml_model_factory(ml, params_dict, True),  # with Naive Bayes single-label classifier
            meka_classpath=params_dict['meka_classpath'],  # obtained via download_meka
            java_command=params_dict['which_java']
        )
    else:
        raise ValueError('error of mll_ml_model_factory methods')


def ml_model_factory(ml, params_dict, is_weka_ml=False):
    if ml == 'SVM':
        if is_weka_ml:
            return "weka.classifiers.functions.SMO -C {}" \
                   " -K \"weka.classifiers.functions.supportVector.RBFKernel -G {}\"".\
                       format(2 ** params_dict['cost'], 2 ** params_dict['gamma'])

        return svm.SVC(C=2 ** params_dict['cost'], gamma=2 ** params_dict['gamma'], probability=True)
    elif ml == 'RF':
        if is_weka_ml:
            return "weka.classifiers.trees.RandomForest -S {} -I {}".format(42, params_dict['tree'])

        return RandomForestClassifier(random_state=42, n_estimators=params_dict['tree'])
    else:
        raise ValueError('ml method err')


def mll_model_factory(mll, params_dict):
    if mll == 'MLkNN':
        return MLkNN(k=params_dict['mll_kNN_k'],
                     s=params_dict['MLkNN_s'],
                     ignore_first_neighbours=params_dict['MLkNN_ignore_first_neighbours'])
    elif mll == 'BRkNNaClassifier':
        return BRkNNaClassifier(k=params_dict['mll_kNN_k'])
    elif mll == 'BRkNNbClassifier':
        return BRkNNbClassifier(k=params_dict['mll_kNN_k'])
    elif mll == 'MLARAM':
        return MLARAM(vigilance=params_dict['MLARAM_vigilance'],
                      threshold=params_dict['MLARAM_threshold'],
                      neurons=params_dict['MLARAM_neurons'] if 'MLARAM_neurons' in params_dict.keys()
                      else None)
    else:
        print(mll)
        raise ValueError('mll method err')