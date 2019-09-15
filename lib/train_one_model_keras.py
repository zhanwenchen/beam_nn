'''
Method to train a single model, given the model dirname.
Save the model architecture, and the models (one model per frequency).
'''
# TODO: refactor data i/o acquisition from by frequency to all frequencies (vectorized)
# TODO: extract and save all the data to disk so we don't have to do this in code.
from itertools import product as itertools_product, repeat as itertools_repeat
from functools import partial as functools_partial
from os.path import join as os_path_join
from os import environ as os_environ, rename as os_rename, mkdir as os_mkdir
from glob import glob as glob_glob
from multiprocessing import Pool
# from multiprocessing.pool import ThreadPool
from random import seed as random_seed
from shutil import move as shutil_move

os_environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

# import numpy as np
from numpy import load as np_load
from numpy import arange as np_arange
from numpy.random import seed as np_random_seed, permutation as np_random_permutation
from numpy import hstack as np_hstack, vstack as np_vstack, split as np_split
# import keras
from keras.wrappers.scikit_learn import KerasRegressor
from h5py import File as h5py_File
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from joblib import dump as joblib_dump


# from lib.utils import save_model_params, ensure_dir, add_suffix_to_path, get_which_model_from_params_fname
from lib.get_model_keras_v1 import get_model_keras

DEFAULT_RANDOM_SEED = 1234
DATA_X_Y_PATH = 'data/BEAM_Reverb_20181004_L74_70mm/train_targets_X_y.npz'
MODELS_DIRNAME = 'DNNs'
MODEL_PARAMS_FNAME = 'model_params.json'
MODEL_DATA_SAVE_FNAME = 'model.joblib'

# Train/Valid/Test is 80%/10%/10%
PERCENT_TEST = 0.1
PERCENT_VALID = 0.1

DEFAULT_BATCH_SIZE = 32
NUM_CLASSES = 130
DEFAULT_EPOCHS = 10

FREQUENCIES_TO_TRAIN_SEPARATELY = [3, 4, 5]


def train_one_model_keras(model_dirpath):
    new_model_folder_name = model_dirpath.replace('_created', '_training')
    shutil_move(model_dirpath, new_model_folder_name)

    for frequency in FREQUENCIES_TO_TRAIN_SEPARATELY:
        train_one_model_per_frequency(model_dirpath, frequency)

    # with Pool() as pool:
    #     # Pool.map is ordered, which could be slower than imap_unordered
    #     print('train_one_model_keras: in pool')
    #     list(pool.imap_unordered(functools_partial(train_one_model_per_frequency, model_dirpath), FREQUENCIES_TO_TRAIN_SEPARATELY))


def train_one_model_per_frequency(model_dirname, frequency, num_epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE):
    model_frequency_dirname = os_path_join(model_dirname, 'k_'+str(frequency))
    os_mkdir(model_frequency_dirname)
    print('train_one_model_per_frequency. model_dirname={}, frequency={}'.format(model_dirname, frequency))

    X_all, y_all = get_training_data()
    X_all_frequency, y_all_frequency = X_all[:, frequency, :, :], y_all[:, frequency, :, :]
    print('train_one_model_per_frequency: got data')
    (X_frequency_train, X_frequency_valid, _), (y_frequency_train, y_frequency_valid, _) = split_examples(X_all_frequency, y_all_frequency)
    X_frequency_train = X_frequency_train.reshape(-1, X_frequency_train.shape[-2] * X_frequency_train.shape[-1])
    X_frequency_valid = X_frequency_valid.reshape(-1, X_frequency_valid.shape[-2] * X_frequency_valid.shape[-1])
    y_frequency_train = y_frequency_train.reshape(-1, y_frequency_train.shape[-2] * y_frequency_train.shape[-1])
    y_frequency_valid = y_frequency_valid.reshape(-1, y_frequency_valid.shape[-2] * y_frequency_valid.shape[-1])
    print('X_frequency_train.shape =', X_frequency_train.shape)
    print('y_frequency_train.shape =', y_frequency_train.shape)
    min_max_scaler = MinMaxScaler()
    model = KerasRegressor(build_fn=get_model_keras,
                           epochs=num_epochs,
                           batch_size=batch_size,
                           verbose=1)

    estimators = []
    # estimators.append(('standardize', StandardScaler()))
    estimators.append(('standardize', min_max_scaler))
    estimators.append(('mlp', model))
    pipeline = Pipeline(estimators)
    # kfold = KFold(n_splits=10, random_state=DEFAULT_RANDOM_SEED)

    print('train_one_model_per_frequency: begin training frequency={}'.format(frequency))
    # results = cross_val_score(pipeline, X_frequency_train, y_frequency_train, cv=kfold, verbose=1, error_score='raise')
    # print("Larger: %.4f (%.4f) MSE" % (results.mean(), results.std()))

    # Export the regressor to a file

    pipeline.fit(X_frequency_train, y_frequency_train)

    model_k_save_path = os_path_join(model_frequency_dirname, MODEL_DATA_SAVE_FNAME)
    joblib_dump(pipeline, model_k_save_path)

    prediction = pipeline.predict(X_frequency_valid)
    print('mean_squared_error(y_valid, prediction) =', mean_squared_error(y_frequency_valid, prediction))


    # param_grid = dict(epochs=[10,20,30])
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    # grid_result = grid.fit(X_train, y_train)

def get_training_data():
    X_y = np_load(DATA_X_Y_PATH)
    return X_y['x'], X_y['y']


def split_examples(X, y, percent_valid=PERCENT_VALID, percent_test=PERCENT_TEST):
    '''
    # Split by target after selecting a single frequency. So X.shape is probably
    [24, 2377, 65, 2]
    '''
    # TODO: prove that the split indices for X match those for y.
    # ## Train, Valid, Test split
    try:
        assert X.ndim == 4
    except:
        raise ValueError('X.ndim should be 4, but X.shape is {}'.format(X.shape))
    num_examples_all = len(X)
    assert len(y) == num_examples_all
    nums_splits = [int((1-percent_valid-percent_test)*num_examples_all), \
                    int((1-percent_test)*num_examples_all)]

    indices_original = np_arange(num_examples_all, dtype=int)
    indices_shuffled = np_random_permutation(indices_original)
    indices_train, indices_valid, indices_test = np_split(indices_shuffled, nums_splits) # pylint: disable=W0632
    X_train, X_valid, X_test = X[indices_train, :, :], X[indices_valid, :, :], X[indices_test, :, :]
    y_train, y_valid, y_test = y[indices_train, :, :], y[indices_valid, :, :], y[indices_test, :, :]
    return (X_train, X_valid, X_test), (y_train, y_valid, y_test)


# def see_x_y_y_hat(which_index):
#     plt.plot(X[which_index, :])
#     plt.plot(Y[which_index, :])
#
#     plt.show()

def seed_everything(seed=DEFAULT_RANDOM_SEED):
    random_seed(seed)
    np_random_seed(seed)
    os_environ['PYTHONHASHSEED'] = str(seed)
