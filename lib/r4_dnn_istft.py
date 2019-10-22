from os.path import isdir as os_path_isdir, join as os_path_join
# from copy import copy
from logging import getLogger as logging_getLogger

from scipy.io import loadmat, savemat
# from numpy import zeros as np_zeros
from torch import zeros as torch_zeros # pylint: disable=E0611
from torch import float64 as torch_float64 # pylint: disable=E0611
from torch import stack as torch_stack # pylint: disable=E0611
from torch import from_numpy as torch_from_numpy # pylint: disable=E0611

# from lib.stft import stft
# from lib.non_iter_ls_inv_stft import non_iter_ls_inv_stft
from lib.stft_wrapper import stft
from lib.istft_wrapper import istft

CHANDAT_FNAME = 'chandat.mat'
NEW_STFT_FNAME = 'new_stft.mat'
LEN_EACH_SECTION = 16
FRAC_OVERLAP = 0.9
PADDING = 16
CHANDAT_DNN_SAVE_FNAME = 'chandat_dnn.mat'
LOGGER = logging_getLogger('evaluate_keras')


def r4_dnn_istft(target_dirname, chandat_obj=None, new_stft_object=None, is_saving_chandat_dnn=True):
    LOGGER.info('{}: r4: Doing istft on denoised stft...'.format(target_dirname))
    assert os_path_isdir(target_dirname)
    if chandat_obj is None:
        chandat_obj = loadmat(os_path_join(target_dirname, CHANDAT_FNAME))
    chandat_data = chandat_obj['chandat']
    num_rows, num_elements, num_beams = chandat_data.shape
    beam_position_x = chandat_obj['beam_position_x']
    depth = chandat_obj['depth']
    f0 = chandat_obj['f0']

    del chandat_obj

    if new_stft_object is None:
        new_stft_object = loadmat(os_path_join(target_dirname, NEW_STFT_FNAME))
    new_stft_real = torch_from_numpy(new_stft_object['new_stft_real']).double()
    new_stft_imag = torch_from_numpy(new_stft_object['new_stft_imag']).double()
    new_stft = torch_stack((new_stft_real, new_stft_imag), axis=-1)
    # new_stft = new_stft_real + 1j*new_stft_imag

    del new_stft_object

    chandat_stft = stft(torch_zeros(num_rows, num_elements, num_beams, dtype=torch_float64), LEN_EACH_SECTION, FRAC_OVERLAP, PADDING)
    chandat_stft['origSigSize'] = [num_rows, num_elements, num_beams]

    # create new and old stfts
    # chandat_stft_new = copy(chandat_stft)
    # chandat_stft_new['stft'] = new_stft
    chandat_stft['stft'] = new_stft

    chandat_new = istft(chandat_stft)
    # chandat_new = non_iter_ls_inv_stft(chandat_stft)
    # what = istft(chandat_stft, N, window=window, hop_length=2, center=False, onesided=False, normalized=False, pad_mode='constant', length=y)

    chandat_new[-3:-1, :, :] = 0

    chandat_dnn_object = {
        'chandat_dnn': chandat_new,
        'beam_position_x': beam_position_x,
        'depth': depth,
        'f0': f0,
    }

    if is_saving_chandat_dnn is True:
        savemat(os_path_join(target_dirname, CHANDAT_DNN_SAVE_FNAME), chandat_dnn_object)

    LOGGER.info('{}: r4: Done'.format(target_dirname))
    return chandat_dnn_object
