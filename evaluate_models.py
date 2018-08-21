# evaluate_models.py
# Description:
#    Runs all models prefixed by an identifier on simulated cyst, phantom cyst,
#    and in vivo data.
# Usage:
#    python evaluate_models.py 123984*
#    python evaluate_models.py "*"
import glob
import os
from subprocess import Popen
import argparse
import shutil
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('identifier', help='Option to load model params from a file. Values in this file take precedence.')
    parser.add_argument('max_to_evaluate', type=int, nargs='?', default=-1, help='The maximum number of models to evaluate, regardless of how many matched folders.')
    parser.add_argument('disable_cudnn', action='store_true', help='To avoid mystery CuDNN error')
    args = parser.parse_args()

    identifier = args.identifier
    max_to_evaluate = args.max_to_evaluate
    disable_cudnn = args.disable_cudnn

    if disable_cudnn is True:
        torch.backends.cudnn.enabled = False

    model_search_path = os.path.join('DNNs', str(identifier) + '_trained')
    models = glob.glob(model_search_path)
    num_models = len(models)

    if num_models == 0:
        raise ValueError('evaluate_models: given identifier ' + str(identifier) + ' , expanded to ' + str(model_search_path) + ' matched no model.')

    if max_to_evaluate > -1:
        count = 0

    for model_index, model_folder in enumerate(models):
        if max_to_evaluate > -1 and count >= max_to_evaluate:
            break

        # rename _trained as _evaluating
        new_folder_name = model_folder.replace('_trained', '_evaluating')
        shutil.move(model_folder, new_folder_name)

        # Skip evaluation unless 3x model.dat are present.
        if not (os.path.isfile(os.path.join(new_folder_name, 'k_3', 'model.dat'))
            and os.path.isfile(os.path.join(new_folder_name, 'k_4', 'model.dat'))
            and os.path.isfile(os.path.join(new_folder_name, 'k_5', 'model.dat'))):

            print('evaluate_models: skipping untrained model', os.path.basename(new_folder_name))
            continue

        # Skip evaluation if already evaluated.
        if os.path.isfile(os.path.join(model_folder, 'scan_batteries', 'target_in_vivo', 'target_17', 'dnn.png')) \
            and os.path.isfile(os.path.join(model_folder, 'scan_batteries', 'target_in_vivo', 'target_19', 'dnn.png')):

            print('evaluate_models: skipping already proccessed model', os.path.basename(new_folder_name))
            continue

        commands = [
            './lib/process_single_scan_battery_anechoic_cyst.sh ' + new_folder_name,
            './lib/process_single_scan_battery_phantom_2p5mm.sh ' + new_folder_name,
            './lib/process_single_scan_battery_in_vivo.sh ' + new_folder_name,
        ]

        print('\n\nevaluate_models.py: processing simulation for model', model_index + 1, 'of', num_models, ':', os.path.basename(new_folder_name), '\n\n')
        Popen(commands[0], shell=True).wait()
        print('\n\nevaluate_models.py: processing phantom for model', model_index + 1, 'of', num_models, ':', os.path.basename(new_folder_name), '\n\n')
        Popen(commands[1], shell=True).wait()
        print('\n\nevaluate_models.py: processing in vivo for model', model_index + 1, 'of', num_models, ':', os.path.basename(new_folder_name), '\n\n')
        Popen(commands[2], shell=True).wait()

        # Check for dnn.png in all 3 experiments.
        if os.path.isfile(os.path.join(new_folder_name, 'scan_batteries', 'target_anechoic_cyst_5mm', 'target_5_SNR_10dB', 'dnn.png')) \
            and os.path.isfile(os.path.join(new_folder_name, 'scan_batteries', 'target_phantom_anechoic_cyst_2p5mm', 'target_5', 'dnn.png')) \
            and os.path.isfile(os.path.join(new_folder_name, 'scan_batteries', 'target_in_vivo', 'target_19', 'dnn.png')):

            shutil.move(new_folder_name, new_folder_name.replace('_evaluating', '_evaluated'))
            if max_to_evaluate > -1: count += 1
        else:
            raise "evaluate_models.py: dnn.png check failed."
