import shlex
import subprocess
import sys
import zipfile

from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
from skmultilearn.ext import Meka, download_meka
import os

from .utils.utils_mll import BLMMeka


def param_meka():
    meka_classpath = os.environ.get('MEKA_CLASSPATH')

    if meka_classpath is None:
        raise ValueError("No meka classpath defined")

def debug_setting_up_MEKA():
    data_home = os.environ.get('SCIKIT_ML_LEARN_DATA',
                               os.path.join('~', 'scikit_ml_learn_data', 'meka'))
    version = '1.9.2'
    meka_release_string = "meka-release-{}".format(version)
    file_name = meka_release_string + '-bin.zip'
    meka_path = get_data_home(subdirectory='meka')
    target_path = os.path.join(meka_path, file_name)
    path_to_lib = os.path.join(meka_path, meka_release_string, 'lib')
    print(target_path)
    print(path_to_lib)
    # 下载好了！
    # /Users/maiqi/scikit_ml_learn_data/meka/meka-release-1.9.2-bin.zip
    # /Users/maiqi/scikit_ml_learn_data/meka/meka-release-1.9.2/lib

    if not os.path.exists(path_to_lib):
        with zipfile.ZipFile(target_path, 'r') as meka_zip:
            print("Unzipping MEKA {} to {}".format(version, meka_path + os.path.sep))
            meka_zip.extractall(path=meka_path + os.path.sep)



    exit()
    meka_classpath = download_meka(version='1.9.2')
    print(meka_classpath)


def setting_up_MEKA():
    meka_classpath = download_meka()
    print(meka_classpath)
    return meka_classpath


def get_data_home(data_home=None, subdirectory=''):
    if data_home is None:
        if len(subdirectory) > 0:
            data_home = os.environ.get('SCIKIT_ML_LEARN_DATA', os.path.join('~', 'scikit_ml_learn_data', subdirectory))
        else:
            data_home = os.environ.get('SCIKIT_ML_LEARN_DATA', os.path.join('~', 'scikit_ml_learn_data'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def use_meka_via_sk_mllearn(meka_classpath):
    from skmultilearn.dataset import load_dataset
    X_train, y_train, _, _ = load_dataset('scene', 'train')
    X_test, y_test, _, _ = load_dataset('scene', 'test')

    # meka = BLMMeka(
    #     meka_classifier="meka.classifiers.multilabel.MULAN -S CLR",  # No!
    #     weka_classifier="weka.classifiers.trees.RandomForest -I 3",  # with Naive Bayes single-label classifier
    #     meka_classpath=meka_classpath,  # obtained via download_meka
    #     java_command='/usr/bin/java'  # path to java executable
    # )

    meka = BLMMeka(
        meka_classifier="meka.classifiers.multilabel.MULAN -S CLR",  # No!
        weka_classifier="weka.classifiers.functions.SMO -C 0.1"
                        " -K \"weka.classifiers.functions.supportVector.RBFKernel -G 0.1\"",
        meka_classpath=meka_classpath,  # obtained via download_meka
        java_command='/usr/bin/java'  # path to java executable
    )
    # " -- -G 0.1"
    # weka.classifiers.trees.RandomForest -I 3
    #

    print(meka)

    meka.fit(X_train, y_train)
    predictions = meka.predict(X_test)

    print(hamming_loss(y_test, predictions))



if __name__ == '__main__':
    # meka_classpath = '/Users/maiqi/scikit_ml_learn_data/meka/meka-release-1.9.2/lib/'
    meka_classpath = setting_up_MEKA()
    use_meka_via_sk_mllearn(meka_classpath)
