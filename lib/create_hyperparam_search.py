#!/usr/bin/python

from itertools import product
import os
import time
import random
import numpy as np


from utils import save_model_params, ensure_dir


def lenet_params(input_size):
    """
    """

    # conv1
    try_conv1_kernel_sizess = list(range(6, 10)) # CHANGED
    try_conv1_num_kernels = list(range(16, 41)) # CHANGED
    # try_conv1_strides = [1, 2] # CHANGED
    conv1_stride = 1

    # pool1
    try_pool1_sizes = [2, 3]
    # pool1_kernel_size = 3
    pool1_stride = 2

    # conv2
    # TODO in loop: conv2_kernel_size should <= conv1_kernel_size
    # TODO in loop: conv2_num_kernels should >= conv1_num_kernels
    try_conv2_kernel_sizess = list(range(2, 10))
    try_conv2_num_kernels = list(range(2, 33))
    # try_conv2_strides = [1, 2]
    conv2_stride = 1

    # pool2
    # try_pool2_sizes = [2, 3]
    pool2_kernel_size = 2
    pool2_stride = 2

    # fcs
    # try_fc_hidden_sizes = [16, 32, 64, 128, 256, 512] # CHANGED: eliminated 8, 1024
    try_fc_hidden_sizes = list(range(65, 521, 65))
    try_fc_num_hidden_layers = [1, 2, 3]


    # Random Hnv1_output_size[1], float) and not conv1_output_size[1].is_integer():
    #         # conv1_output_size must be an integer
    #         # random search is not the most efficient approach but I'm too lazy to filter right now.
    #         conv1_stride = choose_int(try_conv1_strides)
    #         conv1_output_size = yperparameter Search
    def choose_int(array): return int(np.random.choice(array)) # int() because PyTorch doesn't convert np.int64 to int.

    # choose random hyperparameters: optimization
#     batch_size = choose_int(try_batch_sizes)
#     learning_rate = np.random.choice(try_learning_rates)

    # choose random hyperparameters: model
    conv1_kernel_size = choose_int(try_conv1_kernel_sizess)
    conv1_num_kernels = choose_int(try_conv1_num_kernels)
#     conv1_stride = choose_int(try_conv1_strides)

    # enforce relative shape and divisibility
    conv1_output_size = (conv1_num_kernels, (input_size - conv1_kernel_size) / conv1_stride + 1)
#     while isinstance(conv1_output_size[1], float) and not conv1_output_size[1].is_integer():
#         # conv1_output_size must be an integer
#         # random search is not the most efficient approach but I'm too lazy to filter right now.
#         conv1_stride = choose_int(try_conv1_strides)
#         conv1_output_size = (conv1_num_kernels, (input_size - conv1_kernel_size) / conv1_stride + 1)

    # pool1_kernel_size = 3 # by default
    pool1_kernel_size = choose_int(try_pool1_sizes)
    if conv1_output_size[1] % 2 == 0: pool1_kernel_size = 2
    pool1_output_size = (conv1_num_kernels, (conv1_output_size[1] - pool1_kernel_size) / pool1_stride + 1)
    while isinstance(pool1_output_size[1], float) and not pool1_output_size[1].is_integer():
        # conv1_output_size must be an integer
        pool1_kernel_size = choose_int(try_pool1_sizes)
        pool1_output_size = (conv1_num_kernels, (conv1_output_size[1] - pool1_kernel_size) / pool1_stride + 1)

    conv2_kernel_size = choose_int(try_conv2_kernel_sizess)
    conv2_num_kernels = choose_int(try_conv2_num_kernels)
#     conv2_stride = choose_int(try_conv2_strides)

    conv2_output_size = (conv2_num_kernels, (pool1_output_size[1] - conv2_kernel_size) / conv2_stride + 1)
#     while isinstance(conv2_output_size[1], float) and not conv2_output_size[1].is_integer():
#         # conv2_output_size must be an integer
#         conv2_stride = choose_int(try_conv2_strides)
#         conv2_output_size = (conv2_num_kernels, (pool1_output_size[1] - conv2_kernel_size) / conv2_stride + 1)

    # by default pool2_kernel_size = 2
    if conv2_output_size[1] % 2 == 1: pool2_kernel_size = 3
    pool2_output_size = (conv2_num_kernels, (conv2_output_size[1] - pool2_kernel_size) / pool2_stride + 1)

#     while isinstance(pool2_output_size[1], float) and not pool2_output_size[1].is_integer():
#         # conv1_output_size must be an integer
#         pool2_kernel_size = choose_int(try_pool2_sizes)
#         pool2_output_size = (conv2_num_kernels, (conv2_output_size[1] - pool2_kernel_size) / pool2_stride + 1)

    fcs_hidden_size = choose_int(try_fc_hidden_sizes)
    fcs_num_hidden_layers = choose_int(try_fc_num_hidden_layers)



    model_params = {}
    model_params['conv1_kernel_size'] = conv1_kernel_size
    model_params['conv1_num_kernels'] = conv1_num_kernels
    model_params['conv1_stride'] = conv1_stride
    model_params['pool1_kernel_size'] = pool1_kernel_size
    model_params['pool1_stride'] = pool1_stride
    model_params['conv2_kernel_size'] = conv2_kernel_size
    model_params['conv2_num_kernels'] = conv2_num_kernels
    model_params['conv2_stride'] = conv2_stride
    model_params['pool2_kernel_size'] = pool2_kernel_size
    model_params['pool2_stride'] = pool2_stride
    model_params['fcs_hidden_size'] = fcs_hidden_size
    model_params['fcs_num_hidden_layers'] = fcs_num_hidden_layers

    return model_params





if __name__ == '__main__':

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
    total_networks = 1
    while count <= total_networks:

        data_is_target = random.choice(data_is_target_list)
        n_scat = random.choice(num_scat_list)
        bs = random.choice(batch_size_list)
        data_noise_gaussian = random.choice(data_noise_gaussian_list)
        #dropout_input = random.choice(dropout_input_list)
        #dropout = random.choice(dropout_list)
        weight_decay = random.choice(weight_decay_list)

        # get params
        model_params = lenet_params(input_dim)

       # set other params
        model_params['data_is_target'] = data_is_target
        model_params['data_train'] = os.path.join('/home', 'luchieac', 'train_datasets', '20180402_L74_70mm', 'train_' + str(n_scat) + '.h5')
        model_params['data_val'] = os.path.join('/home', 'luchieac', 'train_datasets', '20180402_L74_70mm', 'val_' + str(n_scat) + '.h5')
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
        count = count + 1
