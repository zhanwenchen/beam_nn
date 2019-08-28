# from scipy.io import loadmat
# from numpy import zeros as np_zeros, ones as np_ones
# from numpy import vectorize as np_vectorize

# Test variables
# num_rows, num_elements, num_beams = 1624, 65, 128
# len_each_section = 16;
# frac_overlap = 0.9;
# padding = 16;
# # winInfo = {@rectwin};
# # signal = np_zeros((num_rows, num_elements, num_beams))
# CHANDAT_FNAME = '/Users/zhanwenchen/Documents/projects/beam_nn/DNNs/test/scan_batteries/target_phantom_anechoic_cyst_2p5mm/target_1/chandat.mat'
# signal = loadmat(CHANDAT_FNAME)['chandat']
from numpy import ceil as np_ceil, \
                  arange as np_arange, \
                  prod as np_prod, \
                  zeros as np_zeros, \
                  ndim as np_ndim, \
                  asarray as np_asarray

def parse_matrix_part(matrix, szSub, ovSub):
    assert matrix.ndim == 3
    assert np_ndim(szSub) == 1
    assert len(szSub) == 3
    assert np_ndim(ovSub) == 1
    assert len(ovSub) == 3

    matrix_shape = np_asarray(matrix.shape, dtype=int)
    len_each_section, _, _ = szSub
    shift_length, _, _ = ovSub

    len_each_section_range = np_arange(len_each_section)

    matrix_shape = np_ceil((matrix_shape - szSub + 1)/ovSub).astype(int)
    num_rows_overlap, num_elements, num_beams = matrix_shape
    result_matrix = np_zeros((np_prod(szSub), np_prod(matrix_shape)))
    cnt = 0
    for i in range(num_beams):
        for j in range(num_elements):
            for k in range(num_rows_overlap):
                index_1 = len_each_section_range + k * shift_length
                index_2 = j
                index_3 = i
                tmp = matrix[index_1, index_2, index_3]
                result_matrix[:, cnt] = tmp
                cnt += 1

    return result_matrix
