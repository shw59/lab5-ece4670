import numpy as np
import scipy.io.wavfile as wav

def sdec():

    # Parameters
    N = 1024 # size of each OFDM symbol
    CP = 200 # cyclic prefix length
    FS = 44100 # sampling rate
    K = 350 # number of tones/frequencies carrying data per symbol
    
    # Number of symbols needed to processes all 200,000 bits
    NUM_SYMBOLS = int(np.ceil(200000 / K)) # 200000 / 350 = 571.4 so we need 572 symbols total
    # each symbol is N + CP = 1024 + 200 = 1224 samples long
    SYMBOL_LEN = N + CP

    # read rx.wav and convert from int32 back to float in range [-1, 1]
    _, received = wav.read('rx.wav')
    received = received / np.iinfo(np.int32).max

    # reconstruct the same tone indices as encoder with the center around bin 174 which corresponds to 7500 Hz
    center_bin = int(np.round(7500.0 * N / FS))
    valid_bins = np.arange(1, N // 2)
    distances = np.abs(valid_bins - center_bin)
    sorted_by_distance = valid_bins[np.lexsort((valid_bins, distances))]
    tone_idxs = np.sort(sorted_by_distance[:K])

    # build the sync symbol
    freq_sync = np.zeros(N, dtype=complex)
    
    # generate phase shifts so the cosine waves in the sync symbol don't become a high-energy impulse
    sync_phases = [ 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1
, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1 ]

    # assemble the frequency-domain synchronization symbol
    for k in tone_idxs:
        freq_sync[k] = sync_phases[k]
        freq_sync[N - k] = sync_phases[k]

    # convert to time domain
    time_sync = np.real(np.fft.ifft(freq_sync, norm='ortho'))

    # add cyclic prefix
    sync_symbol = np.concatenate([time_sync[-CP:], time_sync])

    # Find where the sync symbol starts in the received signal
    double_sync = np.concatenate([sync_symbol, sync_symbol])
    
    # Find the correlation between the received signal and the double sync
    correlation = np.correlate(received, double_sync, mode='full')
    
    # the highest point of the correlation is where the signal starts
    correlation = np.abs(correlation)
    peak_index = np.argmax(correlation)
    
    # Fix the channel delay and center it
    sync_start = peak_index - (len(double_sync) - 1) - (CP // 2)

    data_start = sync_start + len(double_sync)

    # Extract and decode each data symbol
    bits_out = np.zeros(NUM_SYMBOLS * K, dtype=int)

    # Get the first sync symbol body after cyclic prefix
    sync1_start = sync_start + CP
    sync1_body = received[sync1_start : sync1_start + N]
    sync1_fft = np.fft.fft(sync1_body, norm='ortho')

    # measure exact power at each of our tone indices in the sync symbol
    sync_powers = np.abs(sync1_fft[tone_idxs]) ** 2
    
    # list of thresholds for each frequency channel
    thresholds = sync_powers / 2.0

    # decode each data symbol
    for i in range(NUM_SYMBOLS):

        # find where this symbol starts
        symbol_start = data_start + i * SYMBOL_LEN

        # strip cyclic prefix by skipping first CP samples
        body_start = symbol_start + CP
        body = received[body_start : body_start + N]

        # take FFT
        symbol_fft = np.fft.fft(body, norm='ortho')

        # check power at each tone index
        for j in range(K):
            tone_index = tone_idxs[j]
            power = np.abs(symbol_fft[tone_index]) ** 2

            # OOK decision for each of the 350 tones
            if power > thresholds[j]:
                bits_out[i * K + j] = 1
            else:
                bits_out[i * K + j] = 0

    # return exactly 200,000 bits
    return bits_out[:200000]
