from numpy import arange as np_arange
from numpy import zeros as np_zeros, ones as np_ones
from numpy import vectorize as np_vectorize
from numpy import broadcast_to as np_broadcast_to
from numpy import newaxis as np_newaxis
from numpy.fft import fft as np_fft, fftn as np_fftn, fft2 as np_fft2
from numpy.linalg import norm as np_norm
from scipy.signal import boxcar # pylint: disable=E0611

from lib.parse_matrix_part import parse_matrix_part
# Test variables
# num_rows, num_elements, num_beams = 1624, 65, 128
# len_each_section = 16;
# frac_overlap = 0.9;
# padding = 16;
# winInfo = {@rectwin};
# signal = np_zeros((num_rows, num_elements, num_beams))
# CHANDAT_FNAME = '/Users/zhanwenchen/Documents/projects/beam_nn/DNNs/test/scan_batteries/target_phantom_anechoic_cyst_2p5mm/target_1/chandat.mat'
# signal = loadmat(CHANDAT_FNAME)['chandat']


def stft(signal, len_each_section, frac_overlap, padding, win_info=boxcar):
    shift_length = round(len_each_section * (1. - frac_overlap)) # shift_length = 2
    len_fourier_transformed_signal = padding

    _, num_elements, num_beams = signal.shape

    zeroCrct = 0

    wVec_rectwin = boxcar(len_fourier_transformed_signal + zeroCrct)
    wVec = wVec_rectwin[zeroCrct//2:len(wVec_rectwin)-zeroCrct//2]

    allOvrlp = parse_matrix_part(signal, [len_each_section, 1, 1], [shift_length, 1, 1])

    num_rows_overlap = allOvrlp.shape[1]//(num_elements * num_beams)

    newShape = [len_each_section, num_rows_overlap, num_elements, num_beams]

    subOvrlp = allOvrlp.reshape(newShape)

    startLocs = np_arange(num_rows_overlap*shift_length, step=shift_length)
    # rep_mat = np_broadcast_to(wVec[:, np_newaxis, np_newaxis, np_newaxis], newShape)

    winOvrlp = subOvrlp * wVec[:, np_newaxis, np_newaxis,np_newaxis]

    # np_norm(winOvrlp)

    stft_array = np_fft(winOvrlp, padding, axis=0)


    freq = np_arange(padding)/padding

    # N = len_each_section
    # K = padding

    out = {}
    out['stft'] = stft_array
    out['freqs'] = freq
    out['startOffsets'] = startLocs
    out['len_each_section'] = len_each_section
    out['padding'] = padding
    out['win_info'] = win_info
    out['frac_overlap'] = frac_overlap
