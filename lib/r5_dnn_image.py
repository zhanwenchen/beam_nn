from os.path import join as os_path_join
from logging import info as logging_info, getLogger as logging_getLogger, INFO as logging_INFO

from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt
from scipy.interpolate import pchip
from numpy import zeros_like as np_zeros_like, \
                  log10 as np_log10, \
                  multiply as np_multiply, \
                  divide as np_divide, \
                  arange as np_arange, \
                  linspace as np_linspace, \
                  apply_along_axis as np_apply_along_axis

from lib.better_envelope import better_envelope
from lib.utils import get_dict_from_file_json, clip_to_eps

TARGET_PARAMETERS_FNAME = 'parameters.json'
TARGET_PARAMETERS_KEY_SCALE_UPSAMPLE = 'scale_upsample'
CHANDAT_FNAME = 'chandat.mat'
CHANDAT_DNN_FNAME = 'chandat_dnn.mat'
CHANDAT_IMAGE_SAVE_FNAME = 'dnn_image.mat'

# logging_getLogger().setLevel(logging_INFO)


def r5_dnn_image(target_dirname, chandat_obj=None, chandat_dnn_obj=None, is_saving_chandat_image=True):
    logging_info('{}: r5: Turning chandat into upsampled envelope...'.format(target_dirname))
    if chandat_obj is None:
        chandat_obj = loadmat(os_path_join(target_dirname, CHANDAT_FNAME))
    f0 = chandat_obj['f0']
    if chandat_dnn_obj is None:
        chandat_dnn_obj = loadmat(os_path_join(target_dirname, CHANDAT_DNN_FNAME))
    chandat_dnn = chandat_dnn_obj['chandat_dnn']
    beam_position_x = chandat_dnn_obj['beam_position_x']
    if f0.ndim and f0.ndim == 2:
        f0 = f0[0, 0]

    rf_data = chandat_dnn.sum(axis=1)
    # design a bandpass filter
    n = 4
    order = n / 2
    critical_frequencies = [1e6, 9e6]/(4*f0/2)
    b, a = butter(order, critical_frequencies, btype='bandpass') # Results are correct

    # chandat_dnn = chandat_dnn.astype(float, copy=False) # REVIEW: necessary?

    rf_data_filt = filtfilt(b, a, rf_data, axis=0, padtype='odd', padlen=3*(max(len(b),len(a))-1)) # Correct

    env = np_apply_along_axis(better_envelope, 0, rf_data_filt)
    # print('r5: env.shape =', env.shape)

    np_divide(env, env.max(), out=env)
    clip_to_eps(env)
    # np_clip(env, np_spacing(1), None, out=env)
    env_dB = np_zeros_like(env)
    np_log10(env, out=env_dB)
    np_multiply(env_dB, 20, out=env_dB)

    # Upscale lateral sampling
    up_scale = get_dict_from_file_json(os_path_join(target_dirname, TARGET_PARAMETERS_FNAME))[TARGET_PARAMETERS_KEY_SCALE_UPSAMPLE]
    up_scale_inverse = 1 / up_scale

    num_beams = env.shape[1]

    x = np_arange(1, num_beams+1)

    new_x = np_arange(1, num_beams+up_scale_inverse, up_scale_inverse)

    # TODO: optimization: instead of doing this apply thing, can we pass in the
    #       whole `env` and specify axis?
    def curried_pchip(y):
        return pchip(x, y)(new_x)

    env_up = np_apply_along_axis(curried_pchip, 1, env)
    # print('r5: env_up.shape =', env_up.shape)

    clip_to_eps(env_up)
    # np_clip(env_up, np_spacing(1), None, out=env_up)
    env_up_dB = np_zeros_like(env_up)
    np_log10(env_up, out=env_up_dB)
    np_multiply(env_up_dB, 20, out=env_up_dB)

    beam_position_x_up = np_linspace(beam_position_x.min(), beam_position_x.max(), env_up_dB.shape[1]) # pylint: disable=E1101

    chandat_image_path = os_path_join(target_dirname, CHANDAT_IMAGE_SAVE_FNAME)

    chandat_image_obj = {
        'rf_data': rf_data,
        'rf_data_filt': rf_data_filt,
        'env': env,
        'env_dB': env_dB,
        'envUp': env_up,
        'envUp_dB': env_up_dB,
        'beam_position_x_up': beam_position_x_up,
        'depth': chandat_dnn_obj['depth'],
    }

    if is_saving_chandat_image is True:
        savemat(chandat_image_path, chandat_image_obj)

    logging_info('{}: r5 Done'.format(target_dirname))
    return chandat_image_obj
