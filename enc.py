import numpy as np
import scipy.io.wavfile as wav

# --- SYSTEM PARAMETERS ---
N = 1024
CP = 200
FS = 44100
K = 410 # Active bins
BITS_PER_BIN = 4 # 2 real + 2 imag (16-QAM)
DATA_BINS = K - 1 # 349 bins for data
BITS_PER_SYM = DATA_BINS * BITS_PER_BIN
NUM_SYMBOLS = int(np.ceil(200000 / BITS_PER_SYM)) # 125 symbols

# Select the 400 bins closest to 7500 Hz
valid_bins = np.arange(1, N // 2)
distances = np.abs(valid_bins - int(np.round(7500.0 * N / FS)))
tone_idxs = np.sort(valid_bins[np.argsort(distances)][:K])

# --- GRAY CODE MAPPINGS ---
def bits_to_pam(b0, b1):
    if b0 == 0 and b1 == 0: return -3.0
    if b0 == 0 and b1 == 1: return -1.0
    if b0 == 1 and b1 == 1: return 1.0
    if b0 == 1 and b1 == 0: return 3.0

# Select the 400 bins closest to 7500 Hz
valid_bins = np.arange(1, N // 2)
distances = np.abs(valid_bins - int(np.round(7500.0 * N / FS)))
tone_idxs = np.sort(valid_bins[np.argsort(distances)][:K])

# Separate the 7.5 kHz frequency bin to act as our continuous pilot tone
pilot_k = int(np.round(7500.0 * N / FS))   # bin 174 = 7.5 kHz — best SNR
data_idxs = np.array([k for k in tone_idxs if k != pilot_k])

# --- ENCODER ---
def enc(bits):
    # Zero-pad bits to fit exactly into our OFDM symbols
    pad_len = NUM_SYMBOLS * BITS_PER_SYM - len(bits)
    bits_padded = np.concatenate([bits, np.zeros(pad_len, dtype=int)])
    
    tx_symbols = []

    # 1. Generate double PN Synchronization Symbol
    np.random.seed(4670)
    sync_phases = np.random.choice([1.0, -1.0], size=N)
    freq_sync = np.zeros(N, dtype=complex)
    for k in tone_idxs:
        freq_sync[k] = sync_phases[k]
        freq_sync[N-k] = sync_phases[k]
        
    time_sync = np.real(np.fft.ifft(freq_sync, norm='ortho'))
    sync_sym = np.concatenate([time_sync[-CP:], time_sync])
    tx_symbols.extend([sync_sym, sync_sym])

    # 2. Generate Data Symbols (16-QAM + 1 Pilot Tone)
    idx = 0
    for i in range(NUM_SYMBOLS):
        freq_data = np.zeros(N, dtype=complex)
        
        # Insert constant Pilot Tone
        freq_data[pilot_k] = 3.0 + 0j
        freq_data[N - pilot_k] = 3.0 - 0j
        
        # Insert Data Tones
        for k in data_idxs:
            r_val = bits_to_pam(bits_padded[idx], bits_padded[idx+1])
            i_val = bits_to_pam(bits_padded[idx+2], bits_padded[idx+3])
            idx += 4
            
            freq_data[k] = r_val + 1j * i_val
            freq_data[N-k] = r_val - 1j * i_val # Conjugate symmetry
            
        time_data = np.real(np.fft.ifft(freq_data, norm='ortho'))
        data_sym = np.concatenate([time_data[-CP:], time_data])
        tx_symbols.append(data_sym)

    # 3. Assemble and Normalize Power
    xraw = np.concatenate(tx_symbols)
    P_raw = np.mean(xraw**2)
    alpha = np.sqrt(0.0012 / P_raw)
    x_norm = xraw * alpha

    # 4. Save to WAV
    tmp = (x_norm * np.iinfo(np.int32).max).astype(np.int32)
    wav.write('tx.wav', FS, tmp)