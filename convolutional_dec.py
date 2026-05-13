import numpy as np
import scipy.io.wavfile as wav

# --- GRAY CODE MAPPINGS (16-QAM) ---
def pam_to_bits(val):
    if val <= -2: return [0, 0]
    elif val <= 0: return [0, 1]
    elif val <= 2: return [1, 1]
    else: return [1, 0]

# --- DECODER ---
def dec():

    # --- SYSTEM PARAMETERS ---
    N = 1024
    CP = 200
    FS = 44100
    K = 350 # Active bins (Reduced from 410 to prevent RLC attenuation errors)
    BITS_PER_BIN = 4 # 2 real + 2 imag (16-QAM)
    DATA_BINS = K - 1 # 349 bins for data, 1 for continuous pilot
    BITS_PER_SYM = DATA_BINS * BITS_PER_BIN # 1396 bits per symbol

    # Calculate symbols needed for the 400,004 convolutionally encoded bits
    CONV_BITS_LEN = (200000 + 2) * 2 # 400,004 bits
    NUM_SYMBOLS = int(np.ceil(CONV_BITS_LEN / BITS_PER_SYM)) # 287 symbols

    # Read rx.wav
    _, rx = wav.read('rx.wav')
    rx = rx / np.iinfo(np.int32).max

    # Select the 350 bins closest to 7500 Hz
    valid_bins = np.arange(1, N // 2)
    distances = np.abs(valid_bins - int(np.round(7500.0 * N / FS)))
    tone_idxs = np.sort(valid_bins[np.argsort(distances)][:K])

    # Separate the 7.5 kHz frequency bin to act as our continuous pilot tone
    pilot_k = int(np.round(7500.0 * N / FS))
    data_idxs = np.array([k for k in tone_idxs if k != pilot_k])

    # 1. Regenerate ideal double-sync template locally
    np.random.seed(4670)
    sync_phases = np.random.choice([1.0, -1.0], size=N)
    freq_sync = np.zeros(N, dtype=complex)
    for k in tone_idxs:
        freq_sync[k] = sync_phases[k]
        freq_sync[N-k] = sync_phases[k]
        
    time_sync = np.real(np.fft.ifft(freq_sync, norm='ortho'))
    sync_sym = np.concatenate([time_sync[-CP:], time_sync])
    double_sync = np.concatenate([sync_sym, sync_sym])

    # 2. Find Exact Start Index via Cross-Correlation (with CP//2 offset for ISI safety buffer)
    corr = np.correlate(rx[:15000], double_sync, mode='full')
    peak_index = np.argmax(np.abs(corr))
    sync_start = peak_index - (len(double_sync) - 1) - (CP // 2)

    # 3. Channel Estimation from second sync symbol body
    sync2_start = sync_start + (N + CP) + CP
    sync2_body = rx[sync2_start : sync2_start + N]
    sync2_fft = np.fft.fft(sync2_body, norm='ortho')
    
    H = np.zeros(N, dtype=complex)
    for k in tone_idxs:
        H[k] = sync2_fft[k] / freq_sync[k]

    data_start = sync_start + 2 * (N + CP)
    
    # 4. Fast-Phase Tracking via Pilot Tones
    all_sym_ffts = []
    P_hats = []
    
    for i in range(NUM_SYMBOLS):
        start = data_start + i * (N + CP)
        body = rx[start + CP : start + N + CP]
        sym_fft = np.fft.fft(body, norm='ortho')
        all_sym_ffts.append(sym_fft)
        
        # Equalize the pilot tone to see how much it rotated
        P_hats.append(sym_fft[pilot_k] / H[pilot_k])
        
    # Extract the raw phase drift relative to the transmitted 3.0+0j
    raw_phases = np.angle(np.array(P_hats) / 3.0)
    
    # Unwrap to safely handle accumulated shifts larger than pi (180 degrees)
    unwrapped_phases = np.unwrap(raw_phases)

    # 5. Extract encoded 16-QAM Data Symbols
    extracted_bits = []
    for i in range(NUM_SYMBOLS):
        sym_fft = all_sym_ffts[i]
        
        # Calculate the exact phase shift per frequency bin
        angle_per_bin = unwrapped_phases[i] / pilot_k

        for k in data_idxs:
            # Zero-Forcing Equalization
            X_hat = sym_fft[k] / H[k]
            
            # Reverse the spinning phase for this specific bin!
            X_hat *= np.exp(-1j * k * angle_per_bin)
            
            # Demodulate using standard PAM decision boundaries
            extracted_bits.extend(pam_to_bits(np.real(X_hat)))
            extracted_bits.extend(pam_to_bits(np.imag(X_hat)))

    # Strip any zero-padding the encoder added to fit the bits squarely into OFDM symbols
    conv_bits = extracted_bits[:CONV_BITS_LEN]

    # 6. VITERBI DECODING (Rate 1/2, Constraint Length 3)
    # State transitions mapped from: state = delay1 * 2 + delay2
    # Format: transitions = { current_state: { input_bit: (next_state, out0, out1) } }
    transitions = {
        0: {0: (0, 0, 0), 1: (2, 1, 1)}, # State 00
        1: {0: (0, 1, 1), 1: (2, 0, 0)}, # State 01
        2: {0: (1, 0, 1), 1: (3, 1, 0)}, # State 10
        3: {0: (1, 1, 0), 1: (3, 0, 1)}  # State 11
    }

    num_pairs = len(conv_bits) // 2
    costs = np.full(4, np.inf)
    costs[0] = 0 # The encoder always starts at state 00
    
    # Trellis matrices to trace our path back
    prev_states = np.zeros((num_pairs, 4), dtype=int)
    decoded_bits = np.zeros((num_pairs, 4), dtype=int)
    
    # Forward pass: compute minimum cost paths through the Trellis
    for t in range(num_pairs):
        r0 = conv_bits[2*t]
        r1 = conv_bits[2*t + 1]
        new_costs = np.full(4, np.inf)
        
        for s in range(4):
            if np.isinf(costs[s]): continue
            
            for b in (0, 1):
                next_s, out0, out1 = transitions[s][b]
                
                # Calculate Hamming distance (number of mismatched bits)
                cost = costs[s] + (r0 != out0) + (r1 != out1)
                
                # Save the cheapest route to this state
                if cost < new_costs[next_s]:
                    new_costs[next_s] = cost
                    prev_states[t, next_s] = s
                    decoded_bits[t, next_s] = b
                    
        costs = new_costs

    # Backward pass: Trace the winning path from the end back to the start
    final_bits = np.zeros(num_pairs, dtype=int)
    
    # The encoder padded the transmission with two 0s, flushing the shift registers back to state 00
    curr_state = 0 
    
    for t in range(num_pairs - 1, -1, -1):
        final_bits[t] = decoded_bits[t, curr_state]
        curr_state = prev_states[t, curr_state]
        
    # Return exactly 200,000 bits (dropping the 2 flush bits added at the end)
    return final_bits[:200000]