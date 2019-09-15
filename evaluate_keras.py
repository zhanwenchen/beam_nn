# evaluate_models.py
# Description:
#    Runs all models prefixed by an identifier on simulated cyst, phantom cyst,
#    and in vivo data.
# Usage:
#    python evaluate_models.py 123984*
#    python evaluate_models.py
from glob import glob as glob_glob
from os.path import join as os_path_join, \
                    basename as os_path_basename, \
                    dirname as os_path_dirname
import argparse
from shutil import move as shutil_move
from time import time as time_time
from logging import basicConfig as logging_basicConfig, \
                    DEBUG as logging_DEBUG, \
                    INFO as logging_INFO

from lib.process_single_scan_battery_keras import process_single_scan_battery_keras
from lib.utils import copy_anything

SCAN_BATTERIES_TARGETS_GLOB_STRING = 'data/BEAM_Reverb_20181004_L74_70mm/target_*_SCR_*_0dB'
SCAN_BATTERIES_DIRNAME = 'scan_batteries'
MODEL_SAVE_FNAME = 'model.joblib'
MODELS_DIRNAME = 'DNNs'
SCRIPT_FNAME = os_path_basename(__file__)
PROJECT_DIRNAME = os_path_dirname(__file__)
LIB_DIRNAME = os_path_join(PROJECT_DIRNAME, 'lib')


def evaluate_one_model_keras(model_dirpath):
    # rename _trained as _evaluating
    new_folder_name = model_dirpath.replace('_trained', '_evaluating')
    shutil_move(model_dirpath, new_folder_name)
    model_name = os_path_basename(new_folder_name)
    copied_scan_battery_dirname = os_path_join(new_folder_name, os_path_basename(SCAN_BATTERIES_DIRNAME))
    copy_anything(SCAN_BATTERIES_DIRNAME, copied_scan_battery_dirname)

    time_start = time_time()

    for scan_battery_dirname in glob_glob(os_path_join(SCAN_BATTERIES_DIRNAME, '*')):
        process_single_scan_battery_keras(new_folder_name, scan_battery_dirname)
    print('{}: it took {:.2f} to evaluate model {} for all scan batteries'.format(SCRIPT_FNAME, time_time() - time_start, model_name))
    shutil_move(new_folder_name, new_folder_name.replace('_evaluating', '_evaluated'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('identifier', type=str, default="x", help='Option to load model params from a file. Values in this file take precedence.')
    parser.add_argument('max_to_evaluate', type=int, nargs='?', default=-1, help='The maximum number of models to evaluate, regardless of how many matched folders.')
    parser.add_argument('-v', '--verbose', help='incrase output verbosity', action='store_true')
    args = parser.parse_args()

    identifier = args.identifier
    max_to_evaluate = args.max_to_evaluate
    verbose = args.verbose

    if verbose:
        logging_basicConfig(level=logging_DEBUG)
    else:
        logging_basicConfig(level=logging_INFO)

    model_search_path = os_path_join(MODELS_DIRNAME, str(identifier) + '_trained')
    models = glob_glob(model_search_path)
    num_models = len(models)

    if num_models == 0:
        raise ValueError('evaluate_models: given identifier ' + str(identifier) + ' , expanded to ' + str(model_search_path) + ' matched no model.')

    if max_to_evaluate > -1:
        count = 0

    # Process each model
    for model_index, model_folder in enumerate(models):
        if max_to_evaluate > -1 and count >= max_to_evaluate:
            break

        evaluate_one_model_keras(model_folder)

        if max_to_evaluate > -1:
            count += 1
