# create_models.py
# Example: python lib/create_models.py 50 hyperparam_ranges.json
import os
import datetime
from random import choice, uniform
import argparse
import json

from lib.utils import save_model_params, ensure_dir, get_conv_output_dims, get_pool_output_dims


model_params_fname = 'model_params.json'


def choose_hyperparameters_from_file(hyperparameter_ranges_file):
    with open(hyperparameter_ranges_file) as f:
        ranges = json.load(f)

    # Load constants.
    input_channel = choice(ranges['input_channel'])
    output_size = ranges['output_size']
    input_size = output_size / input_channel
    if input_size.is_integer():
        input_size = int(input_size)
    else:
        raise ValueError('output_size / input_channel = {} / {} = {}'.format(output_size, input_channel, input_size))


    batch_norm = choice(ranges['batch_norm'])
    use_pooling = choice(ranges['use_pooling'])

    conv1_num_kernels = choice(list(range(*ranges['conv1_num_kernels'])))
    conv1_dropout = uniform(*ranges['conv1_dropout'])

    conv2_num_kernels = choice(list(range(*ranges['conv2_num_kernels'])))
    conv2_dropout = uniform(*ranges['conv2_dropout'])

    # Randomly choose model hyperparameters from ranges.
    conv1_kernel_width_range = list(range(*ranges['conv1_kernel_width']))
    conv1_stride_range = ranges['conv1_stride']

    conv2_kernel_size_range = list(range(*ranges['conv2_kernel_size']))
    conv2_stride_range = ranges['conv2_stride']

    if use_pooling is True:
        pool1_kernel_size_range = ranges['pool1_kernel_size']
        pool1_stride_range = ranges['pool1_stride']

        pool2_kernel_size_range = ranges['pool2_kernel_size']
        pool2_stride_range = ranges['pool2_stride']
    else:
        pool1_kernel_size_range = [1]
        pool1_stride_range = [1]

        pool2_kernel_size_range = [1]
        pool2_stride_range = [1]

    # Size-constrained random hyperparameter search
    possible_size_combinations = []
    for conv1_kernel_width in conv1_kernel_width_range:
        for conv1_stride in conv1_stride_range:
            conv1_pad_width = 0
            conv1_pad_height = 1

            conv1_input_width = 65
            conv1_input_height = 2
            conv1_input_depth = 1

            conv1_kernel_width = conv1_kernel_width
            conv1_kernel_height = 2

            conv1_stride_width = conv1_stride
            conv1_stride_height = 1
            # Satisfy conv1 condition
            conv1_output_width, conv1_output_height, conv1_output_depth = get_conv_output_dims(
                (conv1_input_width, conv1_input_height, conv1_input_depth),
                (conv1_pad_width, conv1_pad_height),
                (conv1_kernel_width, conv1_kernel_height),
                (conv1_stride_width, conv1_stride_height),
                conv1_num_kernels)

            if conv1_output_width <= 0 or \
                not conv1_output_width.is_integer() or \
                conv1_output_height <= 0 or \
                not conv1_output_height.is_integer():
                if conv1_output_width <= 0:
                    print('conv1_output_width = {} < 0'.format(conv1_output_width))
                if not conv1_output_width.is_integer():
                    print('conv1_output_width = {} is not an integer'.format(conv1_output_width))

                if conv1_output_height <= 0:
                    print('conv1_output_height = {} < 0'.format(conv1_output_height))
                if not conv1_output_height.is_integer():
                    print('conv1_output_height = {} is not an integer'.format(conv1_output_height))

                continue

            conv1_output_size = (conv1_output_width, conv1_output_height, conv1_output_depth)
            print('conv1_output_size =', conv1_output_size)
            for pool1_kernel_size in pool1_kernel_size_range:
                for pool1_stride in pool1_stride_range:
                    pool1_input_width = conv1_output_width
                    pool1_input_height = conv1_output_height
                    pool1_input_depth = conv1_output_depth

                    pool1_kernel_width = pool1_kernel_size
                    pool1_kernel_height = 1 # NOTE We only pool length-wise

                    pool1_stride_width = pool1_stride
                    pool1_stride_height = pool1_stride # NOTE Cannot be 2 when conv1_output_height = 3

                    # NOTE: In practice, conv1_output_height is 3, which can't
                    #       be coupled with a pool1 kernel of 1
                    pool1_output_width, pool1_output_height, pool1_output_depth = get_pool_output_dims(
                        (pool1_input_width, pool1_input_height, pool1_input_depth),
                        (pool1_kernel_width, pool1_kernel_height),
                        (pool1_stride_width, pool1_stride_height))
                    if pool1_output_width <= 0 or \
                        not pool1_output_width.is_integer() or \
                        pool1_output_height <= 0 or \
                        not pool1_output_height.is_integer():
                        # if pool1_output_width <= 0:
                            # print('pool1_output_width = {} < 0'.format(pool1_output_width))
                        # if not pool1_output_width.is_integer():
                            # print('type(pool1_output_width) =', type(pool1_output_width))
                            # print('pool1_output_width = {} is not an integer'.format(pool1_output_width))

                        # if pool1_output_height <= 0:
                            # print('pool1_output_height = {} < 0'.format(pool1_output_height))
                        # if not pool1_output_height.is_integer():
                            # print('type(pool1_output_height) =', type(pool1_output_height))
                            # print('pool1_output_height = {} is not an integer'.format(pool1_output_height))
                        continue
                    pool1_output_size = (pool1_output_width, pool1_output_height, pool1_output_depth)
                    # print('pool1_output_size =', pool1_output_size)


                    for conv2_kernel_size in conv2_kernel_size_range:
                        for conv2_stride in conv2_stride_range:
                            conv2_pad_width = 0
                            conv2_pad_height = 0

                            conv2_input_width = 65
                            conv2_input_height = 2
                            conv2_input_depth = pool1_output_depth

                            conv2_kernel_width = conv2_kernel_size
                            conv2_kernel_height = 2

                            conv2_stride_width = conv2_stride
                            conv2_stride_height = 1

                            conv2_output_width, conv2_output_height, conv2_output_depth = get_conv_output_dims(
                                (conv2_input_width, conv2_input_height, conv2_input_depth),
                                (conv2_pad_width, conv2_pad_height),
                                (conv2_kernel_width, conv2_kernel_height),
                                (conv2_stride_width, conv2_stride_height),
                                conv2_num_kernels)

                            if conv2_output_width <= 0 or \
                                not conv2_output_width.is_integer() or \
                                conv2_output_height <= 0 or \
                                not conv2_output_height.is_integer():
                                # if conv2_output_width <= 0:
                                    # print('conv2_output_width = {} < 0'.format(conv2_output_width))
                                # if not conv2_output_width.is_integer():
                                    # print('type(conv2_output_width) =', type(conv2_output_width))
                                    # print('conv2_output_width = {} is not an integer'.format(conv2_output_width))

                                # if conv2_output_height <= 0:
                                    # print('conv2_output_height = {} < 0'.format(conv2_output_height))
                                # if not conv2_output_height.is_integer():
                                    # print('type(conv2_output_height) =', type(conv2_output_height))
                                    # print('conv2_output_height = {} is not an integer'.format(conv2_output_height))

                                continue
                            conv2_output_size = (conv2_output_width, conv2_output_height, conv2_output_depth)
                            # print('conv2_output_size =', conv2_output_size)

                            for pool2_kernel_size in pool2_kernel_size_range:
                                for pool2_stride in pool2_stride_range:
                                    pool2_input_width = conv2_output_width
                                    pool2_input_height = conv2_output_height
                                    pool2_input_depth = conv2_output_depth

                                    pool2_kernel_width = pool2_kernel_size
                                    pool2_kernel_height = 1

                                    pool2_stride_width = pool2_stride
                                    pool2_stride_height = pool2_stride

                                    pool2_output_width, pool2_output_height, pool2_output_depth = get_pool_output_dims(
                                        (pool2_input_width, pool2_input_height, pool2_input_depth),
                                        (pool2_kernel_width, pool2_kernel_height),
                                        (pool2_stride_width, pool2_stride_height))

                                    if pool2_output_width <= 0 or \
                                        not pool2_output_width.is_integer() or \
                                        pool2_output_height <= 0 or \
                                        not pool2_output_height.is_integer():
                                        # if pool2_output_width <= 0:
                                            # print('pool2_output_width = {} <= 0'.format(pool2_output_width))
                                        # if not pool2_output_width.is_integer():
                                            # print('type(pool2_output_width) =', type(pool2_output_width))
                                            # print('pool2_output_width = {} is not an integer'.format(pool2_output_width))

                                        # if pool2_output_height <= 0:
                                            # print('pool2_output_height = {} <= 0'.format(pool2_output_height))
                                        # if not pool2_output_height.is_integer():
                                            # print('type(pool2_output_height) =', type(pool2_output_height))
                                            # print('pool2_output_height = {} is not an integer'.format(pool2_output_height))
                                        continue
                                    pool2_output_size = (pool2_output_width, pool2_output_height, pool2_output_depth)
                                    # print('pool2_output_size =', pool2_output_size)

                                    possible_size_combinations.append((conv1_kernel_width, conv1_stride, pool1_kernel_size, pool1_stride, conv2_kernel_size, conv2_stride, pool2_kernel_size, pool2_stride))


    if len(possible_size_combinations) == 0:
        raise ValueError('{fname}: no possible combination for pool1 given conv1_output_size = {conv1_output_size}; pool1_kernel_size_ranges = {pool1_kernel_size_range}; pool1_stride_ranges = {pool1_stride_range}'.format(fname=__name__, conv1_output_size=conv1_output_size, pool1_kernel_size_range=pool1_kernel_size_range, pool1_stride_range=pool1_stride_range))

    conv1_kernel_width, conv1_stride, pool1_kernel_size, pool1_stride, conv2_kernel_size, conv2_stride, pool2_kernel_size, pool2_stride = choice(possible_size_combinations)

    # print('create_models: ranges[\'fcs_hidden_size\'] =', ranges['fcs_hidden_size'])
    # print('create_models: list(range(*ranges[\'fcs_hidden_size\'])) =', list(range(*ranges['fcs_hidden_size'])))

    fcs_hidden_size = choice(list(range(*ranges['fcs_hidden_size'])))
    fcs_num_hidden_layers = choice(list(range(*ranges['fcs_num_hidden_layers'])))
    fcs_dropout = uniform(*ranges['fcs_dropout'])

    # Randomly choose training hyperparameters from ranges.
    cost_function = choice(ranges['cost_function'])
    optimizer = choice(ranges['optimizer'])

    # learning_rate = None

    if optimizer == 'SGD':
        momentum = uniform(*ranges['momentum'])
        learning_rate = uniform(*ranges['learning_rate_sgd'])
    elif optimizer == 'Adam':
        momentum = None
        learning_rate = uniform(*ranges['learning_rate_adam'])
    else:
        raise ValueError('{}.choose_hyperparameters_from_file: optimizer can only be \'SGD\' or \'Adam\'. Got {}'.format(__name__, optimizer))

    print('outside if: learning_rate =', learning_rate)
    hyperparameters = {
        'input_channel': input_channel,
        'output_size': output_size,

        'batch_norm': batch_norm,

        'use_pooling': use_pooling,
        'pooling_method': ranges['pooling_method'],

        'conv1_kernel_width': conv1_kernel_width,
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
        'learning_rate': learning_rate,
        'momentum': momentum,
    }

    return hyperparameters


def create_models(num_networks, hyperparameter_ranges_file):
    identifier = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    data_is_target_list = [0]
    num_scat_list = [1, 2, 3]
    batch_size_list = [32]
    # data_noise_gaussian_list = [0, 1]
    data_noise_gaussian_list = [1] # Decided on 11/22/2018 b/c better models
    #dropout_input_list = [0, 0.1, 0.2]
    #dropout_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    weight_decay_list = [0]

    for count in range(num_networks):
        data_is_target = choice(data_is_target_list)
        n_scat = choice(num_scat_list)
        bs = choice(batch_size_list)
        data_noise_gaussian = choice(data_noise_gaussian_list)
        #dropout_input = choice(dropout_input_list)
        weight_decay = choice(weight_decay_list)

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
        model_params['weight_decay'] = weight_decay
        model_params['patience'] = 20
        model_params['cuda'] = 1
        model_params['save_initial'] = 0

        k_list = [3, 4, 5]

        for k in k_list:
            model_params['k'] = k
            model_params['save_dir'] = os.path.join('DNNs', identifier + '_' + str(count+1) + '_created', 'k_' + str(k))

            # print(model_params['save_dir'])
            ensure_dir(model_params['save_dir'])
            save_model_params(os.path.join(model_params['save_dir'], model_params_fname), model_params)

        print('create_models: created model {}_{}'.format(identifier, count))

    return identifier


def main():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('num_networks', type=int, help='The number of networks to train.')
    parser.add_argument('hyperparameter_ranges_file', type=str, help='The number of networks to train.')
    args = parser.parse_args()

    num_networks = args.num_networks
    hyperparameter_ranges_file = args.hyperparameter_ranges_file
    return create_models(num_networks, hyperparameter_ranges_file)


if __name__ == '__main__':
    main()
