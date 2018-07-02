# create_hyperparam_search.py
#

import os
import time
import random
import argparse
import json


from pprint import pprint # TODO: delete this. Debug only


from utils import save_model_params, ensure_dir


def choose_hyperparameters_from_file(file_name):

    loop_patience = 100

    with open(hyperparameter_ranges_file) as f:
        ranges = json.load(f)

    # Load constants.
    input_size = ranges['input_size']

    conv1_num_kernels = random.choice(list(range(*ranges['conv1_num_kernels'])))
    conv1_dropout = random.uniform(*ranges['conv1_dropout'])

    conv2_num_kernels = random.choice(list(range(*ranges['conv2_num_kernels'])))
    conv2_dropout = random.uniform(*ranges['conv2_dropout'])

    # Randomly choose model hyperparameters from ranges.
    conv1_kernel_size_range = list(range(*ranges['conv1_kernel_size']))
    conv1_stride_range = ranges['conv1_stride']

    pool1_kernel_size_range = list(range(*ranges['pool1_kernel_size']))
    pool1_stride_range = ranges['pool1_stride']

    conv2_kernel_size_range = list(range(*ranges['conv2_kernel_size']))
    conv2_stride_range = ranges['conv2_stride']

    pool2_kernel_size_range = list(range(*ranges['pool2_kernel_size']))
    pool2_stride_range = ranges['pool2_stride']

    # Size-constrained random hyperparameter search
    possible_size_combinations = []
    for conv1_kernel_size in conv1_kernel_size_range:
        for conv1_stride in conv1_stride_range:
            # Satisfy conv1 condition
            if (input_size - conv1_kernel_size) % conv1_stride != 0:
                continue
            conv1_output_size = (conv1_num_kernels, (input_size - conv1_kernel_size) / conv1_stride + 1)

            for pool1_kernel_size in pool1_kernel_size_range:
                for pool1_stride in pool1_stride_range:
                    if (conv1_output_size[1] - pool1_kernel_size) % pool1_stride != 0:
                        continue
                    pool1_output_size = (conv1_num_kernels, (conv1_output_size[1] - pool1_kernel_size) / pool1_stride + 1)

                    for conv2_kernel_size in conv2_kernel_size_range:
                        for conv2_stride in conv2_stride_range:
                            if (pool1_output_size[1] - conv2_kernel_size) % conv2_stride != 0:
                                continue
                            conv2_output_size = (conv2_num_kernels, (pool1_output_size[1] - conv2_kernel_size) / conv2_stride + 1)

                            for pool2_kernel_size in pool2_kernel_size_range:
                                for pool2_stride in pool2_stride_range:
                                    if (conv2_output_size[1] - pool2_kernel_size) % pool2_stride != 0:
                                        continue
                                    pool2_output_size = (conv2_num_kernels, (conv2_output_size[1] - pool2_kernel_size) / pool2_stride + 1)

                                    possible_size_combinations.append((conv1_kernel_size, conv1_stride, pool1_kernel_size, pool1_stride, conv2_kernel_size, conv2_stride, pool2_kernel_size, pool2_stride))


    if len(possible_size_combinations) == 0:
        raise ValueError('create_hyperparam_search: no possible combination for pool1 given conv1_output_size[1] = ' + str(conv1_output_size[1]) + '; pool1_kernel_size_ranges = ' + str(pool1_kernel_size_ranges) + '; pool1_stride_ranges = ' + str(pool1_stride_ranges))

    conv1_kernel_size, conv1_stride, pool1_kernel_size, pool1_stride, conv2_kernel_size, conv2_stride, pool2_kernel_size, pool2_stride = random.choice(possible_size_combinations)

    # print('create_hyperparam_search: ranges[\'fcs_hidden_size\'] =', ranges['fcs_hidden_size'])
    # print('create_hyperparam_search: list(range(*ranges[\'fcs_hidden_size\'])) =', list(range(*ranges['fcs_hidden_size'])))

    fcs_hidden_size = random.choice(list(range(*ranges['fcs_hidden_size'])))
    fcs_num_hidden_layers = random.choice(list(range(*ranges['fcs_num_hidden_layers'])))
    fcs_dropout = random.uniform(*ranges['fcs_dropout'])


    # Randomly choose training hyperparameters from ranges.
    cost_function = random.choice(ranges['cost_function'])
    momentum = random.uniform(*ranges['momentum'])
    optimizer = random.choice(ranges['optimizer'])
    if optimizer == 'Adam':
        learning_rate = random.uniform(*ranges['learning_rate_adam'])
    elif optimizer == 'SGD':
        learning_rate = random.uniform(*ranges['learning_rate_sgd'])


    hyperparameters = {
        'input_size': input_size,
        'output_size': ranges['output_size'],

        'use_pooling': ranges['use_pooling'],
        'pooling_method': ranges['pooling_method'],

        'conv1_kernel_size': conv1_kernel_size,
        'conv1_num_kernels': conv1_num_kernels,
        'conv1_stride': conv1_stride,
        'conv1_dropout': conv1_dropout,

        'pool1_kernel_size': pool1_kernel_size,
        'pool1_stride': pool1_stride,

        'conv2_kernel_size': conv2_kernel_size,
        'conv2_num_kernels': conv2_num_kernels,
        'conv2_stride': conv2_stride,
        'conv2_dropout': conv2_dropout,

        'pool2_kernel_size': pool2_kernel_size,
        'pool2_stride': pool2_stride,

        'fcs_hidden_size': fcs_hidden_size,
        'fcs_num_hidden_layers': fcs_num_hidden_layers,
        'fcs_dropout': fcs_dropout,

        'cost_function': cost_function,
        'optimizer': optimizer,
        'momentum': momentum,
        'learning_rate': learning_rate,
    }
    pprint(hyperparameters)

    return hyperparameters


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('num_networks', type=int, help='The number of networks to train.')
    parser.add_argument('hyperparameter_ranges_file', type=str, help='The number of networks to train.')
    args = parser.parse_args()

    num_networks = args.num_networks
    hyperparameter_ranges_file = args.hyperparameter_ranges_file


    identifier = str( round(time.time()) )
    k_list = [3, 4, 5]

    data_is_target_list = [0]
    num_scat_list = [1, 2, 3]
    batch_size_list = [32]
    data_noise_gaussian_list = [0, 1]
    #dropout_input_list = [0, 0.1, 0.2]
    #dropout_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    weight_decay_list = [0]

    input_dim = 65
    output_dim = 130

    count = 1
    while count <= num_networks:

        data_is_target = random.choice(data_is_target_list)
        n_scat = random.choice(num_scat_list)
        bs = random.choice(batch_size_list)
        data_noise_gaussian = random.choice(data_noise_gaussian_list)
        #dropout_input = random.choice(dropout_input_list)
        #dropout = random.choice(dropout_list)
        weight_decay = random.choice(weight_decay_list)

        # get params
        model_params = choose_hyperparameters_from_file(hyperparameter_ranges_file)

       # set other params
        model_params['data_is_target'] = data_is_target
        home = os.path.expanduser('~')
        model_params['data_train'] = os.path.join(home,'Downloads', '20180402_L74_70mm', 'train_' + str(n_scat) + '.h5')
        model_params['data_val'] = os.path.join(home, 'Downloads', '20180402_L74_70mm', 'val_' + str(n_scat) + '.h5')
        model_params['batch_size'] = bs
        model_params['data_noise_gaussian'] = data_noise_gaussian
        #model_params['dropout_input'] = dropout_input
        #model_params['dropout'] = dropout
        model_params['weight_decay'] = weight_decay
        model_params['patience'] = 20
        model_params['cuda'] = 1
        model_params['save_initial'] = 0
        model_params['input_dim'] = input_dim
        model_params['output_dim'] = output_dim


        for k in k_list:
            model_params['k'] = k
            model_params['save_dir'] = os.path.join('DNNs', identifier + '_' + str(count), 'k_' + str(k))

            print(model_params['save_dir'])
            ensure_dir(model_params['save_dir'])
            save_model_params(os.path.join(model_params['save_dir'], 'model_params.txt'), model_params)

        # Advance counter for everything except k
        count += 1
