# -*- coding: utf-8 -*-
"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import os
import argparse
import time
import h5py
import torch
from torch.cuda import is_available as get_cuda_available
from torch.autograd import Variable
import numpy as np
from scipy.io import savemat

from lib.utils import load_model


def process_with_model_by_beam_segment_and_k(model, beam, segment, k, stft_real, stft_imag, using_cuda=True):
    # make aperture data
    aperture_data = np.hstack([stft_real[beam, :, segment, k], stft_imag[beam, :, segment, k]])
    aperture_data = aperture_data[np.newaxis, :] # TODO: optimize this

    # normalize by L0 norm
    aperture_data_norm = np.linalg.norm(aperture_data, ord=np.inf, axis=1)

    if aperture_data_norm == 0:
        return np.zeros_like(aperture_data)
    else:
        # Get data for network to process
        x = Variable(torch.from_numpy(aperture_data / aperture_data_norm).float())

        del aperture_data

        if using_cuda == True:
            # x = x.cuda() # TODO: or just x.cuda()
            x.cuda()

        # predict new aperture data
        x = model(x)

        if using_cuda == False:
            x = x.cpu().data.numpy()
        else:
            x = x.data # TODO: Is this correct?

        # renormalize aperture data
        return x * aperture_data_norm


def process_all(using_cuda=True):
    # load stft data
    with h5py.File("old_stft.mat", "r") as old_stft:
        stft_real = np.asarray(old_stft['old_stft_real'])
        stft_imag = np.asarray(old_stft['old_stft_imag'])

    num_beams, num_elements, num_segments, num_frequencies = stft_real.shape

    # create copy of stft data
    stft = np.concatenate([stft_real, stft_imag], axis=1)
    del stft_real, stft_imag
    stft_start = stft.copy()

    # move element position axis
    stft = np.moveaxis(stft, 1, 2)

    # reshape the to flatten first two axes
    stft = np.reshape(stft, [num_beams * num_segments, 2 * num_elements, num_frequencies])

    # setup model dirs dictionary
    with open('../model_dirs.txt', 'r') as f:
        model_dirs = {}
        for line in f:
            [key, value] = line.split(',')
            value = value.rstrip()
            if key.isdigit():
                key = int(key)
            model_dirs[key] = value


    # process stft with networks
    ks = list(range(3, 6))
    for k in ks:
        print('r3_dnn_apply.py: processing k =', k)

        # start timer
        t0 = time.time()

        model = load_model(os.path.join(model_dirs[k], 'model_params.txt'))
        if using_cuda:
            model.cuda()
        # select data by frequency
        aperture_data = stft[:, :, k]

        # normalize by L1 norm
        aperture_data_norm = np.linalg.norm(aperture_data, ord=np.inf, axis=1)
        aperture_data = aperture_data / aperture_data_norm[:, np.newaxis]

        # load into torch and onto gpu
        aperture_data = torch.from_numpy(aperture_data).float()
        if using_cuda:
            aperture_data.cuda()

        # process with the network
        with torch.set_grad_enabled(False):
            aperture_data_new = model(aperture_data).to('cpu').data.numpy()

        # rescale the data
        aperture_data_new = aperture_data_new * aperture_data_norm[:, np.newaxis]

        # store new data in stft
        stft[:, :, k] = aperture_data_new

        # delete the model
        del model

        # stop timer
        print('r3_dnn_apply.py: it took: {:.2f} to process k ='.format(time.time() - t0), k)


    # reshape the stft data
    stft = np.reshape(stft, [num_beams, num_segments, 2 * num_elements, num_frequencies])

    # set zero outside analysis frequency range
    mask = np.zeros_like(stft, dtype=np.float32)
    mask[:, :, :, ks] = 1
    stft = mask * stft
    del mask

    # mirror data to negative frequencies using conjugate symmetry
    stft[:, :, :, np.int32(num_frequencies/2)+1:] = np.flip(stft[:, :, :, 1:np.int32(num_frequencies/2)], axis=3)
    stft[:, :, num_elements:2*num_elements, np.int32(num_frequencies/2)+1:] = -1 * stft[:, :, num_elements:2*num_elements, np.int32(num_frequencies/2)+1:]

    # move element position axis
    stft = np.moveaxis(stft, 1, 2)

    # change variable names
    new_stft_real = stft[:, :num_elements, :, :]
    new_stft_imag = stft[:, num_elements:, :, :]

    # change dimensions
    new_stft_real = new_stft_real.transpose()
    new_stft_imag = new_stft_imag.transpose()

    return new_stft_real, new_stft_imag


if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', help='Option to use GPU.', action="store_true")
    args = parser.parse_args()

    is_cuda_available = get_cuda_available()
    # cuda flag
    if args.cuda == True and is_cuda_available:
        using_cuda = True
        print('r3_dnn_apply.py: Using ' + str(torch.cuda.get_device_name(0)))
    else:
        using_cuda = False
        print('r3_dnn_apply.py: Not using CUDA. CUDA is ' +
              'not' if is_cuda_available else '' + ' available')


    new_stft_real, new_stft_imag = process_all(using_cuda=using_cuda)
    # save new stft data
    savemat('new_stft.mat', {'new_stft_real': new_stft_real, 'new_stft_imag': new_stft_imag})
