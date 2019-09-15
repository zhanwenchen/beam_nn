from scipy.signal import remez, filtfilt, butter
from numpy import sqrt as np_sqrt

def better_envelope(rf_in):
    # cutoff == 0.2 and 0.8 from  fs = 20.832 MHz
    # f_cutoff = 0.2*(fs/2) = 2.0832MHz, 0.8*(fs/2) = 8.3328,
    # f0 = fs/4 = 5.208MHz fixed in Verasonics US systems
    cutoff_low = 0.2
    cutoff_high = 0.8
    # fs = 20.832
    num_taps = 10

    # This is equivalent to B=firpm(10,[.2 .8],[1 1],'Hilbert');
    coefficients = remez(num_taps+1, [cutoff_low/2, cutoff_high/2], [1], type='hilbert')

    Q = filtfilt(coefficients, 1, rf_in)

    envelope = np_sqrt(rf_in ** 2 + Q ** 2)

    b, a = butter(5, 0.25, btype='low')

    envelope_filtered = filtfilt(b, a, envelope, axis=0, padtype='odd', padlen=3*(max(len(b), len(a))-1))

    envelope_filtered.clip(min=0, out=envelope_filtered)

    return envelope_filtered
