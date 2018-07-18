# evaluate_models.py
# Description:
#    Runs all models prefixed by an identifier on simulated cyst, phantom cyst,
#    and in vivo data.
# Usage:
#    python evaluate_models.py 123984
import glob
import os
from subprocess import Popen
import argparse

from utils import read_model_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('identifier', help='Option to load model params from a file. Values in this file take precedence.')
    args = parser.parse_args()

    identifier = args.identifier
    model_search_path = os.path.join('DNNs', str(identifier) + '_trained')
    models = glob.glob(model_search_path)
    num_models = len(models)

    if num_models == 0:
        raise ValueError('evaluate_models: given identifier ' + str(identifier) + ' , expanded to ' + str(model_search_path) + ' matched no model.')

    for model_index, model_folder in enumerate(models):
        # Skip evaluation unless 3x model.dat are present.
        if not (os.path.isfile(os.path.join(model_folder, 'k_3', 'model.dat'))
            and os.path.isfile(os.path.join(model_folder, 'k_4', 'model.dat'))
            and os.path.isfile(os.path.join(model_folder, 'k_5', 'model.dat'))):

            print('evaluate_models: skipping untrained model', os.path.basename(model_folder))
            continue

        # Skip evaluation if already evaluated.
        if os.path.isfile(os.path.join(model_folder, 'scan_batteries', 'target_in_vivo', 'target_17', 'dnn.png'))
            and os.path.isfile(os.path.join(model_folder, 'scan_batteries', 'target_in_vivo', 'target_19', 'dnn.png')):

            print('evaluate_models: skipping already proccessed model', os.path.basename(model_folder))
            continue

        commands = [
            './lib/process_single_scan_battery_anechoic_cyst.sh ' + model_folder,
            './lib/process_single_scan_battery_phantom_2p5mm.sh ' + model_folder,
            './lib/process_single_scan_battery_in_vivo.sh ' + model_folder,
        ]
        
        print('\n\nevaluate_models.py: processing simulation for model', model_index + 1, 'of', num_models, ':', os.path.basename(model_folder), '\n\n')
        Popen(commands[0], shell=True).wait()
        print('\n\nevaluate_models.py: processing phantom for model', model_index + 1, 'of', num_models, ':', os.path.basename(model_folder), '\n\n')
        Popen(commands[1], shell=True).wait()
        print('\n\nevaluate_models.py: processing in vivo for model', model_index + 1, 'of', num_models, ':', os.path.basename(model_folder), '\n\n')
        Popen(commands[2], shell=True).wait()
        os.rename(model_folder, model_folder.replace('_trained', '_evaluated'))
