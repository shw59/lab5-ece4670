import numpy as np
import scipy.io.wavfile as wav

N = 1024
CP = 200
FS = 44100
K = 410 # active bins
BITS_PER_BIN = 4 # 2 real + 2 imag (16-QAM)
DATA_BINS = K - 1
BITS_PER_SYM = DATA_BINS * BITS_PER_BIN
NUM_SYMBOLS = int(np.ceil(200000 / BITS_PER_SYM))

# gray code mappings
def pam_to_bits(val):
    if val <= -2: return [0, 0]
    elif val <= 0: return [0, 1]
    elif val <= 2: return [1, 1]
    else: return [1, 0]

# select the K bins closest to 7500 Hz
valid_bins = np.arange(1, N // 2)
distances = np.abs(valid_bins - int(np.round(7500.0 * N / FS)))
tone_idxs = np.sort(valid_bins[np.argsort(distances)][:K])

# separate the 7.5 kHz frequency bin to act as our continuous pilot tone
pilot_k = int(np.round(7500.0 * N / FS))   # bin 174 = 7.5 kHz has best SNR
data_idxs = np.array([k for k in tone_idxs if k != pilot_k])


def dec():
    _, rx = wav.read('rx.wav')
    rx = rx / np.iinfo(np.int32).max

    # regenerate ideal double-sync template locally
    np.random.seed(4670)
    sync_phases = np.random.choice([1.0, -1.0], size=N)
    freq_sync = np.zeros(N, dtype=complex)
    for k in tone_idxs:
        freq_sync[k] = sync_phases[k]
        freq_sync[N-k] = sync_phases[k]
        
    time_sync = np.real(np.fft.ifft(freq_sync, norm='ortho'))
    sync_sym = np.concatenate([time_sync[-CP:], time_sync])
    double_sync = np.concatenate([sync_sym, sync_sym])

    # find exact start index via cross-correlation
    corr = np.abs(np.correlate(rx[:15000], double_sync, mode='valid'))
    sync_start = np.argmax(corr)

    # channel estimation from second sync symbol
    sync2_start = sync_start + (N + CP)
    sync2_body = rx[sync2_start + CP: sync2_start + N + CP]
    sync2_fft = np.fft.fft(sync2_body, norm='ortho')
    
    H = np.zeros(N, dtype=complex)
    for k in tone_idxs:
        H[k] = sync2_fft[k] / freq_sync[k]

    data_start = sync_start + 2 * (N + CP)
    
    # fast-phase tracking via pilot tones
    all_sym_ffts = []
    P_hats = []
    
    for i in range(NUM_SYMBOLS):
        start = data_start + i * (N + CP)
        body = rx[start + CP : start + N + CP]
        sym_fft = np.fft.fft(body, norm='ortho')
        all_sym_ffts.append(sym_fft)
        
        # equalize the pilot tone to see how much it rotated
        P_hats.append(sym_fft[pilot_k] / H[pilot_k])
        
    # extract the raw phase drift relative to the transmitted 3.0+0j
    raw_phases = np.angle(np.array(P_hats) / 3.0)
    
    # unwrap to safely handle accumulated shifts larger than pi (180 degrees)
    unwrapped_phases = np.unwrap(raw_phases)

    # decode data symbols
    bits_out = []
    for i in range(NUM_SYMBOLS):
        sym_fft = all_sym_ffts[i]
        
        # calculate the exact phase shift per frequency bin
        angle_per_bin = unwrapped_phases[i] / pilot_k

        for k in data_idxs:
            # zero-forcing equalization
            X_hat = sym_fft[k] / H[k]
            
            # reverse the spinning phase for this specific bin
            X_hat *= np.exp(-1j * k * angle_per_bin)
            
            # demodulate using standard PAM decision boundaries
            bits_out.extend(pam_to_bits(np.real(X_hat)))
            bits_out.extend(pam_to_bits(np.imag(X_hat)))

    return np.array(bits_out[:200000], dtype=int)