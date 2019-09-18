from numpy import arange as np_arange, newaxis as np_newaxis, swapaxes as np_swapaxes
from numpy.fft import fft as np_fft
from scipy.signal import boxcar # pylint: disable=E0611

from lib.parse_matrix_part import parse_matrix_part

def stft(signal, len_each_section, frac_overlap, padding, win_info=boxcar):
    shift_length = round(len_each_section * (1. - frac_overlap)) # shift_length = 2

    _, num_elements, num_beams = signal.shape

    zeroCrct = 0

    wVec_rectwin = win_info(len_each_section + zeroCrct)
    wVec = wVec_rectwin[zeroCrct//2:len(wVec_rectwin)-zeroCrct//2]

    allOvrlp = parse_matrix_part(signal, [len_each_section, 1, 1], [shift_length, 1, 1])

    num_rows_overlap = allOvrlp.shape[1]//(num_elements * num_beams)

    newShape = [len_each_section, num_rows_overlap, num_elements, num_beams]

    subOvrlp = allOvrlp.reshape(newShape, order="F") # Matlab defaults to Fortran

    startLocs = np_arange(num_rows_overlap*shift_length, step=shift_length)

    winOvrlp = subOvrlp * wVec[:, np_newaxis, np_newaxis, np_newaxis]

    stft_array = np_fft(winOvrlp, padding, axis=0)

    freq = np_arange(padding)/padding

    out = {
        'stft': stft_array,
        'freqs': freq,
        'startOffsets': startLocs,
        'len_each_section': len_each_section,
        'padding': padding,
        'win_info': win_info,
        'frac_overlap': frac_overlap,
        'shift_length': shift_length,
    }

    return out
