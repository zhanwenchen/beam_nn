__all__ = ['get_df']

import os
import warnings
import sys
import numpy as np
import pandas as pd
import scipy.stats
import glob
import random

sys.path.insert(0, '..')
from lib.utils import read_model_params


speckle_stats_cnr_idx = 1


def get_df(identifier):
    # TODO: 1. Use scan_batteries to load das stats instead of walk through models.
    # setup dnn directory list
    model_folders, num_models = get_models(identifier)

    random_model_folder = random.choice(model_folders)
    random_model_params_txt = os.path.join(random_model_folder, 'k_4', 'model_params.txt')
    random_model_params = read_model_params(random_model_params_txt)
    # random_model_params['index'] = -1
    random_model_params['name'] = 'lol'
    df = pd.DataFrame(columns=random_model_params.keys())

    # loop through dnns and store model params
    for i, model_folder in enumerate(model_folders):
        model_params_txt = os.path.join(model_folder, 'k_4', 'model_params.txt')
        model_params = read_model_params(model_params_txt)
        # model_params['index'] = i
        model_params['name'] = os.path.basename(model_folder)

        df.loc[len(df), :] = model_params

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


    for model_idx, model_folder in enumerate(model_folders):
        model_name = os.path.basename(model_folder)
        scan_batteries = glob.glob(os.path.join(model_folder, 'scan_batteries', '*'))

        # print('analysis_utils: processing model', model_idx, 'of', len(model_folders), model_name)

        for scan_battery_folder in scan_batteries:
            target_folders = glob.glob(os.path.join(scan_battery_folder, 'target_*'))

            scan_battery_name = os.path.basename(scan_battery_folder)

            for target_idx, target_folder in enumerate(target_folders):
                speckle_stats_das_fname = os.path.join(target_folder, 'speckle_stats_das.txt')
                speckle_stats_dnn_fname = os.path.join(target_folder, 'speckle_stats_dnn.txt')

                # cnr_das = np.loadtxt(speckle_stats_das_fname, delimiter=',')[1]
                cnr_das = pd.read_csv(speckle_stats_das_fname, delimiter=",").values[1]
                # cnr_dnn = np.loadtxt(speckle_stats_dnn_fname, delimiter=',')[1]
                cnr_dnn = pd.read_csv(speckle_stats_dnn_fname, delimiter=",").values[1]

                column_name = '_'.join([scan_battery_name, os.path.basename(target_folder)])

                df.loc[model_idx, column_name + '_das_cnr'] = cnr_das
                df.loc[model_idx, column_name + '_dnn_cnr'] = cnr_dnn


    return df


def get_models(identifier):
    model_search_path = os.path.join('..', 'DNNs', str(identifier) + '_evaluated')
    model_folders = glob.glob(model_search_path)
    num_models = len(model_folders)
    if num_models == 0:
        raise ValueError('analysis_utils: given identifier ' + str(identifier) + ' , expanded to ' + str(model_search_path) + ' matched no model.')

    return model_folders, num_models
