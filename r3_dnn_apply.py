import numpy as np
import os
import h5py
import torch
from torch import nn
from torch.autograd import Variable
import argparse
from scipy.io import savemat
import time

# modify path
# import sys
# sys.path.insert(0, '../../../../../lib')
from lib.lenet import LeNet
from lib.utils import read_model_params


if __name__ == "__main__":

    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', help='Option to use GPU.', action="store_true")
    args = parser.parse_args()

    # cuda flag
    if args.cuda and torch.cuda.is_available():
        cuda = True
        print('r3_dnn_apply.py: Using ' + str(torch.cuda.get_device_name(0)))
    else:
        cuda = False
        print('r3_dnn_apply.py: Not using CUDA. CUDA is ' + str(torch.cuda.is_available()) + ' available')

    # load stft data
    f = h5py.File("old_stft.mat", "r")
    stft_real = np.asarray(f['old_stft_real'])
    stft_imag = np.asarray(f['old_stft_imag'])
    N_beams, N_elements, num_segments, N = stft_real.shape
    freqs = np.arange(N)/N

    # create copy of stft data
    stft_real_new = stft_real.copy()
    stft_imag_new = stft_imag.copy()

    # setup model dirs dictionary
    f = open('../model_dirs.txt', 'r')
    model_dirs = {}
    for line in f:
        [key, value] = line.split(',')
        value = value.rstrip()
        if key.isdigit():
            key = int(key)
        model_dirs[key] = value
    f.close()


    # process stft with networks
    k_mask = list(range(3, 6))
    for k in k_mask:
        print('r3_dnn_apply.py: processing k =', k)

        # start timer
        t0 = time.time()

        # load the model
        model_params = read_model_params(os.path.join(model_dirs[k], 'model_params.txt'))
        model = LeNet(model_params['input_size'],
                          model_params['output_size'],

                          model_params['batch_norm'],

                          model_params['use_pooling'],
                          model_params['pooling_method'],

                          model_params['conv1_kernel_size'],
                          model_params['conv1_num_kernels'],
                          model_params['conv1_stride'],
                          model_params['conv1_dropout'],

                          model_params['pool1_kernel_size'],
                          model_params['pool1_stride'],

                          model_params['conv2_kernel_size'],
                          model_params['conv2_num_kernels'],
                          model_params['conv2_stride'],
                          model_params['conv2_dropout'],

                          model_params['pool2_kernel_size'],
                          model_params['pool2_stride'],

                          model_params['fcs_hidden_size'],
                          model_params['fcs_num_hidden_layers'],
                          model_params['fcs_dropout'])
        # model.load_state_dict(torch.load(os.path.join(model_dirs[k], 'model.dat'), map_location='cpu'))
        if cuda == False:
            model.load_state_dict(torch.load(os.path.join(model_dirs[k], 'model.dat'), map_location='cpu'))
        else:
            model.load_state_dict(torch.load(os.path.join(model_dirs[k], 'model.dat')))
            model.cuda()

        model.eval()

        for i in range(num_segments):
            for m in range(N_beams):

                # make aperture data
                aperture_data = np.hstack([stft_real[m, :, i, k], stft_imag[m, :, i, k]])
                aperture_data = aperture_data[np.newaxis, :]

                # normalize by L0 norm
                aperture_data_norm = np.linalg.norm(aperture_data, ord=np.inf, axis=1)
                if aperture_data_norm == 0:
                    aperture_data_new = np.zeros(aperture_data.size)
                else:
                    # Get data for network to process
                    x = Variable(torch.from_numpy(aperture_data / aperture_data_norm).float())
                    if cuda == True:
                        x = x.cuda()

                    # predict new aperture data
                    aperture_data_new = model(x)
                    if cuda == False:
                        aperture_data_new = aperture_data_new.cpu().data.numpy() # TODO: This is by default cpu
                    else:
                        aperture_data_new = aperture_data_new.cuda().data.numpy() # TODO: This is by default cpu
                    # renormalize aperture data
                    aperture_data_new = aperture_data_new * aperture_data_norm

                # store results
                stft_real_new[m, :, i, k] = aperture_data_new[0, :N_elements]
                stft_imag_new[m, :, i, k] = aperture_data_new[0, N_elements:]

        # delete the model
        del model

        # stop timer
        print('r3_dnn_apply.py: it took: {:.2f} to process k ='.format(time.time() - t0), k)

    # set zero outside analysis frequency range
    mask = np.zeros(stft_real.shape)
    mask[:, :, :, k_mask] = 1
    stft_real_new = mask * stft_real_new
    stft_imag_new = mask * stft_imag_new

    # mirror data to negative frequencies
    stft_real_new[:, :, :, np.int32(N/2)+1:] = np.flip(stft_real_new[:, :, :, 1:np.int32(N/2)], axis=3)
    stft_imag_new[:, :, :, np.int32(N/2)+1:] = -np.flip(stft_imag_new[:, :, :, 1:np.int32(N/2)], axis=3)

    # change variable names
    new_stft_real = stft_real_new
    new_stft_imag = stft_imag_new

    # change dimensions
    new_stft_real = new_stft_real.transpose()
    new_stft_imag = new_stft_imag.transpose()

    # save new stft data
    savemat('new_stft.mat', {'new_stft_real': new_stft_real, 'new_stft_imag': new_stft_imag})
