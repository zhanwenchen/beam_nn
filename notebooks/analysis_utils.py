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


# CR
def append_speckle_stats(df, model_idx, column_name, stat_name, columns_stat_das, columns_stat_dnn, speckle_stats_das, speckle_stats_dnn, index):
    stat_das = speckle_stats_das[index]
    stat_dnn = speckle_stats_dnn[index]

    column_name_stat_das = column_name + '_' + stat_name + '_das'
    column_name_stat_dnn = column_name + '_' + stat_name + '_dnn'

    columns_stat_das.append(column_name_stat_das)
    columns_stat_dnn.append(column_name_stat_dnn)

    df.loc[model_idx, column_name_stat_das] = stat_das
    df.loc[model_idx, column_name_stat_dnn] = stat_dnn

    return columns_stat_das, columns_stat_dnn


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

        for scan_battery_folder in scan_batteries:
            target_folders = glob.glob(os.path.join(scan_battery_folder, 'target_*'))

            scan_battery_name = os.path.basename(scan_battery_folder)

            columns_cr_das = []
            columns_cr_dnn = []

            columns_cnr_das = []
            columns_cnr_dnn = []

            columns_snr_das = []
            columns_snr_dnn = []

            columns_mean_in_das = []
            columns_mean_in_dnn = []

            columns_mean_out_das = []
            columns_mean_out_dnn = []

            columns_var_in_das = []
            columns_var_in_dnn = []

            columns_var_out_das = []
            columns_var_out_dnn = []


            for target_idx, target_folder in enumerate(target_folders):
                speckle_stats_das_fname = os.path.join(target_folder, 'speckle_stats_das.txt')
                speckle_stats_dnn_fname = os.path.join(target_folder, 'speckle_stats_dnn.txt')

                speckle_stats_das = pd.read_csv(speckle_stats_das_fname, delimiter=",", header=None).values
                speckle_stats_dnn = pd.read_csv(speckle_stats_dnn_fname, delimiter=",", header=None).values

                column_name = '_'.join([scan_battery_name, os.path.basename(target_folder)])

                # CR
                columns_cr_das, columns_cr_dnn = append_speckle_stats(df, model_idx, column_name, 'cr', columns_cr_das, columns_cr_dnn, speckle_stats_das, speckle_stats_dnn, 0)
                # CNR
                columns_cnr_das, columns_cnr_dnn = append_speckle_stats(df, model_idx, column_name, 'cnr', columns_cnr_das, columns_cnr_dnn, speckle_stats_das, speckle_stats_dnn, 1)
                # SNR
                columns_snr_das, columns_snr_dnn = append_speckle_stats(df, model_idx, column_name, 'snr', columns_snr_das, columns_snr_dnn, speckle_stats_das, speckle_stats_dnn, 2)
                # mean_in
                columns_mean_in_das, columns_mean_in_dnn = append_speckle_stats(df, model_idx, column_name, 'mean_in', columns_mean_in_das, columns_mean_in_dnn, speckle_stats_das, speckle_stats_dnn, 3)
                # mean_out
                columns_mean_out_das, columns_mean_out_dnn = append_speckle_stats(df, model_idx, column_name, 'mean_out', columns_mean_out_das, columns_mean_out_dnn, speckle_stats_das, speckle_stats_dnn, 4)
                # var_in
                columns_var_in_das, columns_var_in_dnn = append_speckle_stats(df, model_idx, column_name, 'var_in', columns_var_in_das, columns_var_in_dnn, speckle_stats_das, speckle_stats_dnn, 5)
                # var_out
                columns_var_out_das, columns_var_out_dnn = append_speckle_stats(df, model_idx, column_name, 'var_out', columns_var_out_das, columns_var_out_dnn, speckle_stats_das, speckle_stats_dnn, 6)


            # Average speckle stats across targets
            df.loc[:, scan_battery_name + '_avg_cr_das'] = df.loc[:, columns_cr_das].mean(axis=1)
            df.loc[:, scan_battery_name + '_avg_cr_dnn'] = df.loc[:, columns_cr_dnn].mean(axis=1)

            df.loc[:, scan_battery_name + '_avg_cnr_das'] = df.loc[:, columns_cnr_das].mean(axis=1)
            df.loc[:, scan_battery_name + '_avg_cnr_dnn'] = df.loc[:, columns_cnr_dnn].mean(axis=1)

            df.loc[:, scan_battery_name + '_avg_snr_das'] = df.loc[:, columns_snr_das].mean(axis=1)
            df.loc[:, scan_battery_name + '_avg_snr_dnn'] = df.loc[:, columns_snr_dnn].mean(axis=1)

            df.loc[:, scan_battery_name + '_avg_mean_in_das'] = df.loc[:, columns_mean_in_das].mean(axis=1)
            df.loc[:, scan_battery_name + '_avg_mean_in_dnn'] = df.loc[:, columns_mean_in_dnn].mean(axis=1)

            df.loc[:, scan_battery_name + '_avg_mean_out_das'] = df.loc[:, columns_mean_out_das].mean(axis=1)
            df.loc[:, scan_battery_name + '_avg_mean_out_dnn'] = df.loc[:, columns_mean_out_dnn].mean(axis=1)

            df.loc[:, scan_battery_name + '_avg_var_in_das'] = df.loc[:, columns_var_in_das].mean(axis=1)
            df.loc[:, scan_battery_name + '_avg_var_in_dnn'] = df.loc[:, columns_var_in_dnn].mean(axis=1)

            df.loc[:, scan_battery_name + '_avg_var_out_das'] = df.loc[:, columns_var_out_das].mean(axis=1)
            df.loc[:, scan_battery_name + '_avg_var_out_dnn'] = df.loc[:, columns_var_out_dnn].mean(axis=1)

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
