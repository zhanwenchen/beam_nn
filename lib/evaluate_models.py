# evaluate_models.py
# Description:
#    Runs all models prefixed by an identifier on simulated cyst, phantom cyst,
#    and in vivo data.
# Usage:
#    python evaluate_models.py
import glob
import os
from subprocess import Popen


from utils import read_model_params


if __name__ == '__main__':
    last_identifier = read_model_params(os.path.join('DNNs', 'last_identifier.txt'))['last_identifier']
    models = glob.glob(os.path.join('DNNs', str(last_identifier) + '*'))

    for model_folder in models:
        commands = [
            './lib/process_single_scan_battery_anechoic_cyst.sh ' + model_folder,
            './lib/process_single_scan_battery_phantom_2p5mm.sh ' + model_folder,
            './lib/process_single_scan_battery_in_vivo.sh ' + model_folder,
        ]
        processes = [Popen(cmd, shell=True) for cmd in commands]
        # wait for completion
        for p in processes: p.wait()
