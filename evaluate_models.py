# evaluate_models.py
# Description:
#    Runs all models prefixed by an identifier on simulated cyst, phantom cyst,
#    and in vivo data.
# Usage:
#    python evaluate_models.py 123984*
#    python evaluate_models.py "*"
import glob
import os
import argparse
import shutil
import time
import sys
import io


import matlab.engine

from lib.process_single_scan_battery import process_single_scan_battery
from lib.utils import clean_buffers


SCRIPT_FNAME = os.path.basename(__file__)
PROJECT_DIRNAME = os.path.dirname(__file__)
LIB_DIRNAME = os.path.join(PROJECT_DIRNAME, 'lib')
LIB_MATLAB_DIRNAME = os.path.abspath(os.path.join(LIB_DIRNAME, 'matlab'))
is_profiling_gpu = True

if is_profiling_gpu: from lib.gpu_profile import gpu_profile


if __name__ == '__main__':
    if is_profiling_gpu: sys.settrace(gpu_profile)

    parser = argparse.ArgumentParser()
    parser.add_argument('identifier', help='Option to load model params from a file. Values in this file take precedence.')
    parser.add_argument('max_to_evaluate', type=int, nargs='?', default=-1, help='The maximum number of models to evaluate, regardless of how many matched folders.')
    args = parser.parse_args()

    identifier = args.identifier
    max_to_evaluate = args.max_to_evaluate

    model_search_path = os.path.join('DNNs', str(identifier) + '_trained')
    models = glob.glob(model_search_path)
    num_models = len(models)

    if num_models == 0:
        raise ValueError('evaluate_models: given identifier ' + str(identifier) + ' , expanded to ' + str(model_search_path) + ' matched no model.')

    if max_to_evaluate > -1:
        count = 0

    # Initialize MATLAB things engine and buffers.
    eng = matlab.engine.start_matlab()
    out = io.StringIO()
    err = io.StringIO()

    # Add MATLAB helper scripts shared by all scan batteries.
    clean_buffers(out, err)
    try:
        eng.addpath(LIB_MATLAB_DIRNAME, nargout=0, stdout=out, stderr=err)
        sys.stdout.write(out.getvalue())
        sys.stderr.write(err.getvalue())
    except Exception as e:
        sys.stdout.write(out.getvalue())
        sys.stderr.write(err.getvalue())
        raise RuntimeError(str(e))


    # Process each model
    for model_index, model_folder in enumerate(models):
        if max_to_evaluate > -1 and count >= max_to_evaluate:
            break

        # rename _trained as _evaluating
        new_folder_name = model_folder.replace('_trained', '_evaluating')
        shutil.move(model_folder, new_folder_name)
        model_name = os.path.basename(new_folder_name)

        # Skip evaluation unless 3x model.dat are present.
        if not (os.path.isfile(os.path.join(new_folder_name, 'k_3', 'model.dat'))
            and os.path.isfile(os.path.join(new_folder_name, 'k_4', 'model.dat'))
            and os.path.isfile(os.path.join(new_folder_name, 'k_5', 'model.dat'))):

            print('evaluate_models: skipping untrained model', model_name)
            continue

        # Skip evaluation if already evaluated.
        if os.path.isfile(os.path.join(model_folder, 'scan_batteries', 'target_in_vivo', 'target_17', 'dnn.png')) \
            and os.path.isfile(os.path.join(model_folder, 'scan_batteries', 'target_in_vivo', 'target_19', 'dnn.png')):

            print('evaluate_models: skipping already proccessed model', model_name)
            continue

        time_start = time.time()

        if is_profiling_gpu: gpu_profile(frame=sys._getframe(), event='line', arg=None)

        print('{}: processing simulation for model {} of {}: {}'.format(SCRIPT_FNAME, model_index+1, num_models, model_name))
        process_single_scan_battery(new_folder_name, os.path.join('scan_batteries', 'target_anechoic_cyst_5mm'), matlab_session=eng)
        if not os.path.isfile(os.path.join(new_folder_name, 'scan_batteries', 'target_anechoic_cyst_5mm', 'target_5_SNR_10dB', 'dnn.png')):
            raise RuntimeError('evaluate_models.py: dnn.png check failed for simulation target_5.')
        if is_profiling_gpu: gpu_profile(frame=sys._getframe(), event='line', arg=None)

        print('{}: processing phantom for model {} of {}: {}'.format(SCRIPT_FNAME, model_index+1, num_models, model_name))
        process_single_scan_battery(new_folder_name, os.path.join('scan_batteries', 'target_phantom_anechoic_cyst_2p5mm'), matlab_session=eng)
        if not os.path.isfile(os.path.join(new_folder_name, 'scan_batteries', 'target_phantom_anechoic_cyst_2p5mm', 'target_5', 'dnn.png')):
            raise RuntimeError('evaluate_models.py: dnn.png check failed for phantom target_5.')
        if is_profiling_gpu: gpu_profile(frame=sys._getframe(), event='line', arg=None)

        print('{}: processing in vivo for model {} of {}: {}'.format(SCRIPT_FNAME, model_index+1, num_models, model_name))
        process_single_scan_battery(new_folder_name, os.path.join('scan_batteries', 'target_in_vivo'), matlab_session=eng)
        if not os.path.isfile(os.path.join(new_folder_name, 'scan_batteries', 'target_in_vivo', 'target_19', 'dnn.png')):
            raise RuntimeError('evaluate_models.py: dnn.png check failed for in_vivo target_19.')

        if is_profiling_gpu: gpu_profile(frame=sys._getframe(), event='line', arg=None)
        print('{}: it took {:.2f} to evaluate model {} for all scan batteries'.format(SCRIPT_FNAME, time.time() - time_start, model_name))
        shutil.move(new_folder_name, new_folder_name.replace('_evaluating', '_evaluated'))


        if max_to_evaluate > -1:
            count += 1
