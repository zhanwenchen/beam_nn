# evaluate_models.py
# Description:
#    Runs all models prefixed by an identifier on simulated cyst, phantom cyst,
#    and in vivo data.
# Usage:
#    python evaluate_models.py
import glob
import os
from subprocess import Popen
import argparse


from utils import read_model_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('model_params_path', help='Option to load model params from a file. Values in this file take precedence.')
    parser.add_argument('identifier', help='Option to load model params from a file. Values in this file take precedence.')
    args = parser.parse_args()

    identifier = args.identifier

    # last_identifier = read_model_params(os.path.join('DNNs', 'last_identifier.txt'))['last_identifier']
    models = glob.glob(os.path.join('DNNs', str(identifier) + '*'))

    for model_folder in models:
        commands = [
            './lib/process_single_scan_battery_anechoic_cyst.sh ' + model_folder,
            './lib/process_single_scan_battery_phantom_2p5mm.sh ' + model_folder,
            './lib/process_single_scan_battery_in_vivo.sh ' + model_folder,
        ]
        Popen(commands[0], shell=True).wait()
        Popen(commands[1], shell=True).wait()
        Popen(commands[2], shell=True).wait()

        # processes = [Popen(cmd, shell=True) for cmd in commands]
        # wait for completion
        # for p in processes: p.wait()
