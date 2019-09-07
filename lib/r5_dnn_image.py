from os.path import join as os_path_join

from scipy.io import loadmat, savemat
from scipy.signal import butter, filtfilt
from scipy.interpolate import pchip
from numpy import zeros_like as np_zeros_like, \
                  spacing as np_spacing, \
                  log10 as np_log10, \
                  clip as np_clip, \
                  multiply as np_multiply, \
                  divide as np_divide, \
                  arange as np_arange, \
                  linspace as np_linspace, \
                  apply_along_axis as np_apply_along_axis

from lib.better_envelope import better_envelope


CHANDAT_FNAME = 'chandat.mat'
CHANDAT_DNN_FNAME = 'chandat_dnn.mat'
CHANDAT_IMAGE_SAVE_FNAME = 'chandat_image.mat'
EPS = np_spacing(1)

def r5_dnn_image(target_dirname):
    chandat_obj = loadmat(os_path_join(target_dirname, CHANDAT_FNAME))
    f0 = chandat_obj['f0']
    chandat_dnn_obj = loadmat(os_path_join(target_dirname, CHANDAT_DNN_FNAME))
    chandat_dnn = chandat_dnn_obj['chandat_dnn']
    beam_position_x = chandat_dnn_obj['beam_position_x']
    if f0.ndim and f0.ndim == 2:
        f0 = f0[0, 0]

    # design a bandpass filter
    n = 4
    order = n / 2
    critical_frequencies = [1e6, 9e6]/(4*f0/2)
    b, a = butter(order, critical_frequencies, btype='bandpass') # Results are correct

    # chandat_dnn = chandat_dnn.astype(float, copy=False) # REVIEW: necessary?

    rf_data = chandat_dnn.sum(axis=1)

    rf_data_filt = filtfilt(b, a, rf_data, axis=0, padtype='odd', padlen=3*(max(len(b),len(a))-1)) # Correct

    env = np_apply_along_axis(better_envelope, 0, rf_data_filt)

    np_divide(env, env.max(), out=env)
    clip_to_eps(env)
    # np_clip(env, np_spacing(1), None, out=env)
    env_dB = np_zeros_like(env)
    np_log10(env, out=env_dB)
    np_multiply(env_dB, 20, out=env_dB)

    # Upscale lateral sampling
    up_scale = 2
    up_scale_inverse = 1 / up_scale

    num_beams = env.shape[1]

    x = np_arange(1, num_beams+1)

    new_x = np_arange(1, num_beams+up_scale_inverse, up_scale_inverse)

    def curried_pchip(y):
        return pchip(x, y)(new_x)

    env_up = np_apply_along_axis(curried_pchip, 1, env)

    clip_to_eps(env_up)
    # np_clip(env_up, np_spacing(1), None, out=env_up)
    env_up_dB = np_zeros_like(env_up)
    np_log10(env_up, out=env_up_dB)
    np_multiply(env_up_dB, 20, out=env_up_dB)


    beam_position_x_up = np_linspace(beam_position_x.min(), beam_position_x.max(), num_beams)

    chandat_image_path = os_path_join(target_dirname, CHANDAT_IMAGE_SAVE_FNAME)
    savemat(chandat_image_path, {
        'rf_data': rf_data,
        'rf_data_filt': rf_data_filt,
        'env': env,
        'env_dB': env_dB,
        'env_up': env_up,
        'env_up_dB': env_up_dB,
        'beam_position_x_up': beam_position_x_up,
        'depth': chandat_dnn_obj['depth'],
    })


def clip_to_eps(array):
    '''
    Inplace clip of array to a min of the Matlab eps, which is usually
    2.220446049250313e-16 and equivalent to numpy.spacing(1)
    '''
    np_clip(array, EPS, None, out=array)
