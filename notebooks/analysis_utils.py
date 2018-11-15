import os
import warnings
import sys
import random
from glob import glob

import numpy as np
import pandas as pd
import scipy.stats
# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pprint import pprint

sys.path.insert(0, '..')
from lib.utils import read_model_params

SCRIPT_FNAME = os.path.basename(__file__)

MODEL_PARAMS_FNAME = 'model_params.json'
MODEL_PARAMS_FNAME_ALT = 'model_params.txt'
speckle_stats_cnr_idx = 1

MODELS_DIRNAME = 'DNNs'

USE_K = 'k_4'


plot_size_factor = 7

scan_batteries_names = ['target_anechoic_cyst_5mm', 'target_phantom_anechoic_cyst_2p5mm', 'target_in_vivo']

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

    columns = [
        'input_channel',
        'output_size',
        'batch_norm',
        'use_pooling',
        'pooling_method',
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
        'cost_function',
        'optimizer',
        'learning_rate',
        'momentum',
        'data_is_target',
        'data_train',
        'data_val',
        'batch_size',
        'data_noise_gaussian',
        'weight_decay',
        'patience',
        'cuda',
        'save_initial',
        'k',
        'save_dir',
        'name',
        'loss_val_k_3',
        'loss_val_k_4',
        'loss_val_k_5',
    ]

    df = pd.DataFrame(columns=columns)


    # loop through dnns and store model params
    for i, model_folder in enumerate(model_folders):
        model_params = os.path.join(model_folder, 'k_4', MODEL_PARAMS_FNAME)
        model_params_alt = os.path.join(model_folder, 'k_4', MODEL_PARAMS_FNAME_ALT)
        try:
            model_params = read_model_params(model_params)
        except:
            try:
                model_params = read_model_params(model_params_alt)
            except:
                raise OSError('{}'.format(SCRIPT_FNAME))
        # model_params['index'] = i
        model_params['name'] = os.path.basename(model_folder)

        df.loc[len(df), :] = model_params

    # loop through dnns and load losses
    loss_val = np.zeros((num_models, 3))
    for i, dir_dnn in enumerate(model_folders):
        for kk, k in enumerate(range(3, 6)):
            dir = os.path.join(dir_dnn, 'k_' + str(k))
            dat = np.loadtxt(os.path.join(dir, 'log.txt'), delimiter=',')

            try:
                loss_val[i, kk] = dat[:, 3].min()
            except:
                # Sometimes losses are NaN, where NumPy.min() raises.
                loss_val[i, kk] = np.nanmin(dat[:, 3])

    df['loss_val_k_3'] = pd.Series(loss_val[:, 0], index=df.index)
    df['loss_val_k_4'] = pd.Series(loss_val[:, 1], index=df.index)
    df['loss_val_k_5'] = pd.Series(loss_val[:, 2], index=df.index)


    for model_idx, model_folder in enumerate(model_folders):
        model_name = os.path.basename(model_folder)
        # scan_batteries_names = ['target_anechoic_cyst_5mm', 'target_in_vivo', 'target_phantom_anechoic_cyst_2p5mm']
        # scan_batteries = [os.path.join(model_folder, sb) for sb in scan_batteries_names]
        scan_batteries = glob(os.path.join(model_folder, 'scan_batteries', '*'))
        # scan_batteries = sorted(scan_batteries, key=os.path.getmtime)

        for scan_battery_folder in scan_batteries:
            target_folders = glob(os.path.join(scan_battery_folder, 'target_*'))

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
    # print(list(df))
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

    for numeric_column in numeric_columns:
        if not numeric_column in list(df):
            raise ValueError('{}: {} is not in the index'.format(SCRIPT_FNAME, numeric_column))


    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='raise')

    return df


def get_models(identifier):
    model_search_path = os.path.join('..', 'DNNs', str(identifier) + '_evaluated')
    model_folders = glob(model_search_path)
    num_models = len(model_folders)
    if num_models == 0:
        raise ValueError('analysis_utils: given identifier ' + str(identifier) + ' , expanded to ' + str(model_search_path) + ' matched no model.')

    return model_folders, num_models


def inspect_model_by_name(model_folder, df):
    if not MODELS_DIRNAME in model_folder:
        model_folder = os.path.join('..', MODELS_DIRNAME, model_folder)

    model_name = os.path.basename(model_folder)
    scan_batteries_dirname = os.path.join(model_folder, 'scan_batteries')
    # scan_batteries_folders = glob(os.path.join(scan_batteries_dirname, 'target_*'))
    # scan_batteries_folders = sorted(scan_batteries_folders, key=os.path.getmtime)
    scan_batteries_folders = [os.path.join(scan_batteries_dirname, sb) for sb in scan_batteries_names]
    num_scan_batteries = len(scan_batteries_folders)

    if num_scan_batteries == 0:
        raise ValueError('No scan batteries found in path {}'.format(os.path.join(scan_batteries_dirname, 'target_*')))

    df_model_row = df[df['name'] == model_name]
    for scan_batteries_folder in scan_batteries_folders:
        scan_battery_name = os.path.basename(scan_batteries_folder)
        targets = glob(os.path.join(scan_batteries_folder, 'target_*'))

        num_targets = len(targets)

        if num_targets == 0:
            raise ValueError('No targets batteries found in path {}'.format(os.path.join(scan_batteries_folder, 'target_*')))

        fig, axes = plt.subplots(num_targets, 2, figsize=(2 * plot_size_factor, num_targets * plot_size_factor), frameon=False)

        for j, target in enumerate(targets):
            target_name = os.path.basename(target)

            column_base_name = scan_battery_name + '_' + target_name

            target_cr_das = df_model_row.loc[:, column_base_name + '_cr_das'].values[0]
            target_cnr_das = df_model_row.loc[:, column_base_name + '_cnr_das'].values[0]
            target_snr_das = df_model_row.loc[:, column_base_name + '_snr_das'].values[0]

            target_cr_dnn = df_model_row.loc[:, column_base_name + '_cr_dnn'].values[0]
            target_cnr_dnn = df_model_row.loc[:, column_base_name + '_cnr_dnn'].values[0]
            target_snr_dnn = df_model_row.loc[:, column_base_name + '_snr_dnn'].values[0]

            axes[j, 0].imshow(plt.imread(os.path.join(target, 'das.png')))
            axes[j, 0].set_title('DAS\nmodel = {}\ntarget = {}\ncr = {}\ncnr = {}\nsnr = {}'.format(model_name, target_name, target_cr_das, target_cnr_das, target_snr_das))
            axes[j, 0].set_axis_off()

            axes[j, 1].imshow(plt.imread(os.path.join(target, 'dnn.png')))
            axes[j, 1].set_title('CNN\nmodel = {}\ntarget = {}\ncr = {}\ncnr = {}\nsnr = {}'.format(model_name, target_name, target_cr_dnn, target_cnr_dnn, target_snr_dnn))
            axes[j, 1].set_axis_off()

        fig.suptitle('model {}: {}'.format(model_name, scan_battery_name))

        plt.subplots_adjust(wspace=0, hspace=0)
        # plt.tight_layout()

    plt.show()


def compare_two_models(model_folder_1, model_folder_2, df):
    if not MODELS_DIRNAME in model_folder_1:
        model_folder_1 = os.path.join('..', MODELS_DIRNAME, model_folder_1)

    if not MODELS_DIRNAME in model_folder_2:
        model_folder_2 = os.path.join('..', MODELS_DIRNAME, model_folder_2)

    model_name_1 = os.path.basename(model_folder_1)
    model_name_2 = os.path.basename(model_folder_2)

    scan_batteries_dirname_1 = os.path.join(model_folder_1, 'scan_batteries')
    scan_batteries_dirname_2 = os.path.join(model_folder_2, 'scan_batteries')

    scan_batteries_folders_1 = glob(os.path.join(scan_batteries_dirname_1, 'target_*'))
    scan_batteries_folders_2 = glob(os.path.join(scan_batteries_dirname_2, 'target_*'))

    try:
        assert len(scan_batteries_folders_1) == len(scan_batteries_folders_2)
    except:
        raise ValueError('model {} and {} have different number of scan batteries'.format(targets_1, targets_2))

    num_scan_batteries = len(scan_batteries_folders_1)

    for i in range(num_scan_batteries):

        targets_1 = glob(os.path.join(scan_batteries_folders_1[i], 'target_*'))
        targets_2 = glob(os.path.join(scan_batteries_folders_2[i], 'target_*'))

        try:
            assert len(targets_1) == len(targets_2)
        except:
            raise ValueError('model {} and {} have different number of targets'.format(targets_1, targets_2))

        num_targets = len(targets_1)

        fig, axes = plt.subplots(num_targets, 2, figsize=(2 * plot_size_factor, num_targets * plot_size_factor), frameon=False)

        for j in range(num_targets):
            axes[j, 0].imshow(plt.imread(os.path.join(targets_1[j], 'dnn.png')))
            axes[j, 0].set_title('{} target 1 (DNN)'.format(model_name_1))
            axes[j, 0].set_axis_off()

            axes[j, 1].imshow(plt.imread(os.path.join(targets_2[j], 'dnn.png')))
            axes[j, 1].set_title('{} target 2 (DNN)'.format(model_name_2))
            axes[j, 1].set_axis_off()

        fig.suptitle('model {} vs {}: {}'.format(model_name_1, model_name_2, scan_batteries_folders_1[i]))

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()


    plt.show()
