from os.path import isdir as os_path_isdir, join as os_path_join
# from copy import copy
from logging import info as logging_info, getLogger as logging_getLogger, INFO as logging_INFO

from scipy.io import loadmat, savemat
from numpy import zeros as np_zeros

from lib.stft import stft
from lib.non_iter_ls_inv_stft import non_iter_ls_inv_stft


CHANDAT_FNAME = 'chandat.mat'
NEW_STFT_FNAME = 'new_stft.mat'
LEN_EACH_SECTION = 16
FRAC_OVERLAP = 0.9
PADDING = 16
CHANDAT_DNN_SAVE_FNAME = 'chandat_dnn.mat'

# logging_getLogger().setLevel(logging_INFO)


def r4_dnn_istft(target_dirname, chandat_obj=None, new_stft_object=None, is_saving_chandat_dnn=True):
    logging_info('{}: r4: Doing istft on denoised stft...'.format(target_dirname))
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
    # import pdb; pdb.set_trace()
    new_stft = new_stft_object['new_stft_real'] + 1j*new_stft_object['new_stft_imag']

    del new_stft_object

    chandat_stft = stft(np_zeros((num_rows, num_elements, num_beams)), LEN_EACH_SECTION, FRAC_OVERLAP, PADDING)
    chandat_stft['origSigSize'] = [num_rows, num_elements, num_beams]

    # create new and old stfts
    # chandat_stft_new = copy(chandat_stft)
    # chandat_stft_new['stft'] = new_stft
    chandat_stft['stft'] = new_stft

    chandat_new = non_iter_ls_inv_stft(chandat_stft)

    chandat_new[-3:-1, :, :] = 0

    chandat_dnn_object = {
        'chandat_dnn': chandat_new,
        'beam_position_x': beam_position_x,
        'depth': depth,
        'f0': f0,
    }

    if is_saving_chandat_dnn is True:
        savemat(os_path_join(target_dirname, CHANDAT_DNN_SAVE_FNAME), chandat_dnn_object)

    logging_info('{}: r4: Done'.format(target_dirname))
    return chandat_dnn_object
