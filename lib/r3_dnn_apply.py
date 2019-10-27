from os.path import dirname as os_path_dirname, join as os_path_join, basename as os_path_basename
from h5py import File as h5py_File
from logging import getLogger as logging_getLogger, INFO as logging_INFO
# from joblib import load as joblib_load

from numpy import reshape as np_reshape, \
                  moveaxis as np_moveaxis, \
                  flip as np_flip, \
                  inf as np_inf, \
                  newaxis as np_newaxis, \
                  concatenate as np_concatenate, \
                  ones_like as np_ones_like
from numpy.linalg import norm as np_linalg_norm
from scipy.io import savemat
from torch import load as torch_load
from torch import device as torch_device # pylint: disable=E0611
from torch import from_numpy as torch_from_numpy # pylint: disable=E0611
from torch import no_grad as torch_no_grad
from torch.cuda import is_available as torch_cuda_is_cuda_available

from lib.utils import get_which_model_from_params_fname
# from lib.any_model import AnyModel

SCRIPT_FNAME = os_path_basename(__file__)
MODEL_SAVE_FNAME = 'model.dat'
MODEL_PARAMS_FNAME = 'model_params.json'
# MODEL_PARAMS_FNAME = 'model_params.json'

LOGGER = logging_getLogger()
LOGGER.setLevel(logging_INFO)

# TODO: we don't actually need to store/pass stft_real and stft

def r3_dnn_apply(target_dirname, old_stft_obj=None, using_cuda=True, saving_to_disk=True):
    LOGGER.info('{}: r3: Denoising original stft with neural network model...'.format(target_dirname))
    '''
    r3_dnn_apply takes an old_stft object (or side effect load from disk)
    and saves a new_stft object
    '''
    scan_battery_dirname = os_path_dirname(target_dirname)
    model_dirname = os_path_dirname(os_path_dirname(scan_battery_dirname))

    # load stft data
    if old_stft_obj is None:
        old_stft_fpath = os_path_join(target_dirname, 'old_stft.mat')
        with h5py_File(old_stft_fpath, 'r') as f:
            stft = np_concatenate([f['old_stft_real'][:], f['old_stft_imag'][:]], axis=1)
    else:
        stft = np_concatenate([old_stft_obj['old_stft_real'], old_stft_obj['old_stft_imag']], axis=1)

    N_beams, N_elements_2, N_segments, N_fft = stft.shape
    N_elements = N_elements_2 // 2

    # combine stft_real and stft_imag

    # move element position axis
    stft = np_moveaxis(stft, 1, 2) # TODO: Duplicate?

    # reshape the to flatten first two axes
    stft = np_reshape(stft, [N_beams*N_segments, N_elements_2, N_fft]) # TODO: Duplicate?

    # process stft with networks

    k_mask = list(range(3, 6))
    # breakpoint()
    for frequency in k_mask:
        process_each_frequency(model_dirname, stft, frequency, using_cuda=using_cuda)

    # reshape the stft data
    stft = np_reshape(stft, [N_beams, N_segments, N_elements_2, N_fft]) # TODO: Duplicate?

    # set zero outside analysis frequency range
    discard_mask = np_ones_like(stft, dtype=bool)
    discard_mask[:, :, :, k_mask] = False # pylint: disable=E1137
    stft[discard_mask] = 0
    del discard_mask

    # mirror data to negative frequencies using conjugate symmetry
    end_index = N_fft // 2
    stft[:, :, :, end_index+1:] = np_flip(stft[:, :, :, 1:end_index], axis=3)
    stft[:, :, N_elements:2*N_elements, end_index+1:] = -1 * stft[:, :, N_elements:2*N_elements, end_index+1:]

    # move element position axis
    stft = np_moveaxis(stft, 1, 2) # TODO: Duplicate?

    # change variable names
    # new_stft_real = stft[:, :N_elements, :, :]
    new_stft_real = stft[:, :N_elements, :, :].transpose()
    # new_stft_imag = stft[:, N_elements:, :, :]
    new_stft_imag = stft[:, N_elements:, :, :].transpose()

    del stft

    # change dimensions
    # new_stft_real = new_stft_real.transpose()
    # new_stft_imag = new_stft_imag.transpose()

    # save new stft data
    new_stft_obj = {'new_stft_real': new_stft_real, 'new_stft_imag': new_stft_imag}
    if saving_to_disk is True:
        new_stft_fname = os_path_join(target_dirname, 'new_stft.mat')
        savemat(new_stft_fname, new_stft_obj)
    LOGGER.info('{}: r3 Done.'.format(target_dirname))
    return new_stft_obj


def process_each_frequency(model_dirname, stft, frequency, using_cuda=True, scan_battery_is_reverb=False):
    '''
    Setter method on stft.
    '''
    is_using_cuda = using_cuda and torch_cuda_is_cuda_available()
    my_device = torch_device('cuda:0' if is_using_cuda else 'cpu')

    # 1. Instantiate Neural Network Model
    model_params_fname = os_path_join(os_path_join(model_dirname, 'k_' + str(frequency)), MODEL_PARAMS_FNAME)

    model_save_fpath = os_path_join(model_dirname, 'k_' + str(frequency), MODEL_SAVE_FNAME)
    model = get_which_model_from_params_fname(model_params_fname)
    model.load_state_dict(torch_load(os_path_join(os_path_dirname(model_save_fpath), 'model.dat'), map_location=my_device), strict=False)
    model.eval()
    model = model.to(my_device)

    if False:
        model.printing = True
        from lib.print_layer import PrintLayer
        new_model_net = []
        for layer in model.net:
            new_model_net.append(layer)
            new_model_net.append(PrintLayer(layer))

        from torch.nn import Sequential
        model.net = Sequential(*new_model_net)
    # 2. Get X_test
    LOGGER.debug('r3.process_each_frequency: stft.shape = {}'.format(stft.shape))
    # breakpoint()

    # Reverb data k = [0, 1, 2]. But model dirname use k = [3,4,5] regardless
    # if stft.shape[-1] == 2:
    #     frequency -= 3

    aperture_data = stft[:, :, frequency] # or stft_frequency

    # 2.1. normalize by L1 norm
    aperture_data_norm = np_linalg_norm(aperture_data, ord=np_inf, axis=1)
    aperture_data /= aperture_data_norm[:, np_newaxis]

    # load into torch and onto gpu
    # X_test = aperture_data
    aperture_data = torch_from_numpy(aperture_data).float().to(my_device)

    # 3. Predict
    # y_hat = loaded_model_pipeline.predict(X_test)
    # y_hat is aperture_data_new
    with torch_no_grad():
        aperture_data_new = model(aperture_data).cpu().data.numpy()

    # if stft.shape[-1] == 2:
    #     breakpoint()
    del aperture_data, model

    # 4. Postprocess on y_hat
    # rescale the data and store new data in stft
    stft[:, :, frequency] = aperture_data_new * aperture_data_norm[:, np_newaxis]
    del aperture_data_new, aperture_data_norm
