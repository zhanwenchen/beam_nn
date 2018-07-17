"""
@author: Adam Luiches
@author: Zhanwen Chen
"""

__all__ = ['get_speckle_stats_dnn_and_das', 'get_speckle_stats_dnn', 'get_speckle_stats_das']

import numpy as np
import pandas as pd
import scipy.stats
import os
import warnings
import glob
import sys

sys.path.insert(0, '../lib')
from utils import read_model_params


dirs_dnn_parent = '../DNNs/'


def get_df(identifier):
    # setup dnn directory list
    model_folders, num_models = get_models(identifier)

    # loop through dnns and store model params
    for i, dir_dnn in enumerate(model_folders):
        m_name = os.path.join(dir_dnn, 'k_4', 'model_params.txt')
        model_params = read_model_params(m_name)
        model_params['index'] = i
        model_params['name'] = os.path.basename(dir_dnn)

        df_single = pd.DataFrame([model_params])

        if i == 0:
            df = df_single
        else:
            df = pd.concat([df, df_single])

    df = df.set_index('index')


    # loop through dnns and load losses
    loss_val = np.zeros((num_models, 3))
    for i, dir_dnn in enumerate(model_folders):
        for kk, k in enumerate(range(3, 6)):
            dir = os.path.join(dir_dnn, 'k_' + str(k))
            dat = np.loadtxt(os.path.join(dir, 'log.txt'), delimiter=',')
            loss_val[i, kk] = dat[:, 3].min()


    df['loss_val_k_3'] = pd.Series(loss_val[:, 0], index=df.index)
    df['loss_val_k_4'] = pd.Series(loss_val[:, 1], index=df.index)
    df['loss_val_k_5'] = pd.Series(loss_val[:, 2], index=df.index)

    return df

## Load DNN results.
def get_speckle_stats_dnn(scan_battery_name, target_num_list, target_suffix, identifier):

    model_folders, num_models = get_models(identifier)

    # setup data storage
    speckle_stats_dnn = np.zeros((num_models, 7, len(target_num_list)))

    for i, dir_dnn in enumerate(model_folders):
        # select the scan battery
        scan_battery = os.path.join(dirs_dnn_parent, dir_dnn, 'scan_batteries', scan_battery_name)

        for m, target_num in enumerate(target_num_list):

            # data directory
            dir_scan = 'target_' + str(target_num) + target_suffix

            # filenames
            filename_dnn = os.path.join(scan_battery, dir_scan, 'speckle_stats_dnn.txt')

            # load DNN results
            try:
                speckle_stats_dnn[i, :, m] = np.loadtxt(filename_dnn, delimiter=',')
            except:
                warnings.warn('analysis_utils: Unable to find ' + filename_dnn + ' for ' + scan_battery_name)

    return speckle_stats_dnn


# Load DAS results.
def get_speckle_stats_das(scan_battery_name, target_num_list, target_suffix, identifier):
    """target_num_list can be a simulated cyst, a phantom cyst, or an in vivo target."""
    speckle_stats_das = np.zeros((7, len(target_num_list))) # TODO: remove magic number "7".

    model_folders, num_models = get_models(identifier)

    for m, (target_num, model_folder) in enumerate(zip(target_num_list, model_folders)):
        dir_scan = 'target_' + str(target_num) + target_suffix
        # filename_das = os.path.join('..', 'DNNs', identifier + '1', 'scan_batteries', scan_battery_name, dir_scan, 'speckle_stats_das.txt')
        filename_das = os.path.join(model_folder, 'scan_batteries', scan_battery_name, dir_scan, 'speckle_stats_das.txt')
        speckle_stats_das[:, m] = np.loadtxt(filename_das, delimiter=',')

    return speckle_stats_das


# Convenience function to load both DNN and DAS results.
def get_speckle_stats_dnn_and_das(scan_battery_name, target_num_list, target_suffix, identifier):
    speckle_stats_dnn = get_speckle_stats_dnn(scan_battery_name, target_num_list, target_suffix, identifier)
    speckle_stats_das = get_speckle_stats_das(scan_battery_name, target_num_list, target_suffix, identifier)

    return speckle_stats_dnn, speckle_stats_das

def get_models(identifier):
    model_search_path = os.path.join('..', 'DNNs', str(identifier) + '_evaluated')
    model_folders = glob.glob(model_search_path)
    num_models = len(model_folders)
    if num_models == 0:
        raise ValueError('analysis_utils: given identifier ' + str(identifier) + ' , expanded to ' + str(model_search_path) + ' matched no model.')

    return model_folders, num_models
