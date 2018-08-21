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
    # TODO: 2. Walk model_folders only once.
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

            columns_cnr_das = []
            columns_cnr_dnn = []

            columns_snr_das = []
            columns_snr_dnn = []


            for target_idx, target_folder in enumerate(target_folders):
                speckle_stats_das_fname = os.path.join(target_folder, 'speckle_stats_das.txt')
                speckle_stats_dnn_fname = os.path.join(target_folder, 'speckle_stats_dnn.txt')

                speckle_stats_das = pd.read_csv(speckle_stats_das_fname, delimiter=",", header=None).values
                speckle_stats_dnn = pd.read_csv(speckle_stats_dnn_fname, delimiter=",", header=None).values

                cnr_das = speckle_stats_das[1]
                cnr_dnn = speckle_stats_dnn[1]

                snr_das = speckle_stats_das[2]
                snr_dnn = speckle_stats_dnn[2]

                column_name = '_'.join([scan_battery_name, os.path.basename(target_folder)])

                # CNR
                column_name_cnr_das = column_name + '_cnr_das'
                column_name_cnr_dnn = column_name + '_cnr_dnn'

                columns_cnr_das.append(column_name_cnr_das)
                columns_cnr_dnn.append(column_name_cnr_dnn)

                df.loc[model_idx, column_name_cnr_das] = cnr_das
                df.loc[model_idx, column_name_cnr_dnn] = cnr_dnn

                # SNR
                column_name_snr_das = column_name + '_snr_das'
                column_name_snr_dnn = column_name + '_snr_dnn'

                columns_snr_das.append(column_name_snr_das)
                columns_snr_dnn.append(column_name_snr_dnn)

                df.loc[model_idx, column_name_snr_das] = snr_das
                df.loc[model_idx, column_name_snr_dnn] = snr_dnn

            df.loc[:, scan_battery_name + '_avg_cnr_das'] = df.loc[:, columns_cnr_das].mean(axis=1)
            df.loc[:, scan_battery_name + '_avg_cnr_dnn'] = df.loc[:, columns_cnr_dnn].mean(axis=1)

            df.loc[:, scan_battery_name + '_avg_snr_das'] = df.loc[:, columns_snr_das].mean(axis=1)
            df.loc[:, scan_battery_name + '_avg_snr_dnn'] = df.loc[:, columns_snr_dnn].mean(axis=1)

    # Convert model params columns to numeric
    numeric_columns = [ \
        'conv1_kernel_size',
        'conv1_num_kernels',
        'conv1_stride',
        'conv1_dropout',

        'pool1_kernel_size',
        'pool1_stride',

        'conv2_kernel_size',
        'conv2_num_kernels',
        'conv2_stride',
        'conv2_dropout',

        'pool2_kernel_size',
        'pool2_stride',

        'fcs_hidden_size',
        'fcs_num_hidden_layers',
        'fcs_dropout',

        'learning_rate',
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='raise')

    return df


def get_models(identifier):
    model_search_path = os.path.join('..', 'DNNs', str(identifier) + '_evaluated')
    model_folders = glob.glob(model_search_path)
    num_models = len(model_folders)
    if num_models == 0:
        raise ValueError('analysis_utils: given identifier ' + str(identifier) + ' , expanded to ' + str(model_search_path) + ' matched no model.')

    return model_folders, num_models
