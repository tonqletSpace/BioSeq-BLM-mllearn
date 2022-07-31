import zipfile

from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
from skmultilearn.ext import Meka, download_meka
import os


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


def get_data_home(data_home=None, subdirectory=''):
    """Return the path of the scikit-multilearn data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the :code:`data_home` is set to a folder named
    :code:`'scikit_ml_learn_data'` in the user home folder.

    Alternatively, it can be set by the :code:`'SCIKIT_ML_LEARN_DATA'`
    environment variable or programmatically by giving an explicit
    folder path. The :code:`'~'` symbol is expanded to the user home
    folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str (default is None)
        the path to the directory in which scikit-multilearn data sets
        should be stored, if None the path is generated as stated above

    subdirectory : str, default ''
        return path subdirectory under data_home if data_home passed or under default if not passed

    Returns
    --------
    str
        the path to the data home
    """
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

    meka = Meka(
        meka_classifier="meka.classifiers.multilabel.BR",  # Binary Relevance
        weka_classifier="weka.classifiers.bayes.NaiveBayesMultinomial",  # with Naive Bayes single-label classifier
        meka_classpath=meka_classpath,  # obtained via download_meka
        java_command='/usr/bin/java'  # path to java executable
    )
    print(meka)

    meka.fit(X_train, y_train)
    predictions = meka.predict(X_test)

    print(hamming_loss(y_test, predictions))


if __name__ == '__main__':
    # setting_up_MEKA()
    meka_classpath = '/Users/maiqi/scikit_ml_learn_data/meka/meka-release-1.9.2/lib/'
    use_meka_via_sk_mllearn(meka_classpath)