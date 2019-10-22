from os.path import isdir as os_path_isdir, join as os_path_join
from logging import getLogger as logging_getLogger

from scipy.io import loadmat, savemat

from lib.stft_wrapper import stft


CHANDAT_FNAME = 'chandat.mat'
NEW_STFT_FNAME = 'new_stft.mat'
LEN_EACH_SECTION = 16
FRAC_OVERLAP = 0.9
PADDING = 16
CHANDAT_DNN_SAVE_FNAME = 'chandat_dnn_test.mat'
OLD_STFT_SAVE_FNAME = 'old_stft.mat'

LOGGER = logging_getLogger('evaluate_keras')


def r2_dnn_stft(target_dirname, saving_to_disk=True):
    LOGGER.info('{}: r2: Generating old_stft from chandat'.format(target_dirname))
    assert os_path_isdir(target_dirname)
    chandat_obj = loadmat(os_path_join(target_dirname, CHANDAT_FNAME))
    chandat = chandat_obj['chandat']
    y, num_elements, num_beams = chandat.shape # y is depth

    # Take STFT of the channel data for specified A-line
    chandat_stft = stft(chandat, LEN_EACH_SECTION, FRAC_OVERLAP, PADDING)
    chandat_stft['origSigSize'] = [y, num_elements, num_beams]

    chandat_obj = {
        'old_stft_real': chandat_stft['stft'][:, :, :, 0],
        'old_stft_imag': chandat_stft['stft'][:, :, :, 1],
    }

    if saving_to_disk is True:
        savemat(os_path_join(target_dirname, OLD_STFT_SAVE_FNAME), chandat_obj)

    LOGGER.info('{}: r2: Done'.format(target_dirname))
    breakpoint()
    return chandat_obj
