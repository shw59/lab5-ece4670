import numpy as np
import scipy.io.wavfile as wav
import os
from enc import enc
from dec import dec
from senc import senc
from sdec import sdec

def run_encode():
    print("\nGenerating 200,000 random bits...")
    original_bits = np.random.randint(0, 2, 200000)
    
    # Save the bits to the disk so the decoder can grade them later
    np.save('original_bits.npy', original_bits)
    
    print("Running encoder...")
    enc(original_bits)
    
    print("\n--- Encoding Complete ---")
    print("tx.wav has been successfully generated.")
    print("\nNow, run the channel simulation manually in your terminal.")
    print("Example command (for Windows):")
    print("python ccplay --prepause 2000 --postpause 2000 --channel audio0 tx.wav rx.wav")
    print("\nAfter rx.wav is generated, run this script again and choose 'Decode'.")

def run_sencode():
    print("\nGenerating 200,000 random bits...")
    original_bits = np.random.randint(0, 2, 200000)
    
    # Save the bits to the disk so the decoder can grade them later
    np.save('original_bits.npy', original_bits)
    
    print("Running encoder...")
    senc(original_bits)
    
    print("\n--- Encoding Complete ---")
    print("tx.wav has been successfully generated.")
    print("\nNow, run the channel simulation manually in your terminal.")
    print("Example command (for Windows):")
    print("python ccplay --prepause 2000 --postpause 2000 --channel audio0 tx.wav rx.wav")
    print("\nAfter rx.wav is generated, run this script again and choose 'Decode'.")    

def run_decode():
    if not os.path.exists('rx.wav'):
        print("\nError: 'rx.wav' not found! Please run the ccplay command in your terminal first.")
        return
    if not os.path.exists('original_bits.npy'):
        print("\nError: 'original_bits.npy' not found! Please run the Encode step first.")
        return
    if not os.path.exists('tx.wav'):
        print("\nError: 'tx.wav' not found! It is needed to calculate your power and data rate.")
        return

    print("\nRunning decoder...")
    decoded_bits = dec()
    
    print("Loading original bits to check for errors...")
    original_bits = np.load('original_bits.npy')
    
    # Calculate Bit Errors
    N = np.sum(original_bits != decoded_bits)
    
    # find where errors are clustered
    error_positions = np.where(original_bits != decoded_bits)[0]
    if len(error_positions) > 0:
        print(f"First error at bit position: {error_positions[0]}")
        print(f"Last error at bit position: {error_positions[-1]}")
        print(f"Errors in first 10000 bits: {np.sum(original_bits[:10000] != decoded_bits[:10000])}")
        print(f"Errors in last 10000 bits: {np.sum(original_bits[-10000:] != decoded_bits[-10000:])}")
        print(f"Errors in middle 10000 bits: {np.sum(original_bits[95000:105000] != decoded_bits[95000:105000])}")

    # Read tx.wav to calculate Power and Data Rate
    fs, X_tx_int = wav.read('tx.wav')
    X_tx = X_tx_int / np.iinfo(np.int32).max
    P = np.mean(X_tx**2)
    
    duration_sec = len(X_tx) / fs
    R = 200000 / duration_sec
    
    # Calculate Figure of Merit
    numerator = min(R, 3000000) * ((1 - N / 100000)**10)
    denominator = max(1, 800 * P)
    fom = numerator / denominator
    
    # Print the results
    print(f"\n--- Lab 5 High-Performance System Performance ---")
    print(f"tx.wav Duration : {duration_sec:.4f} seconds")
    print(f"Data Rate (R)   : {R:.2f} bps")
    print(f"Avg Power (P)   : {P:.6f}")
    print(f"Total Errors (N): {N} out of 200,000")
    print(f"\n>>> FIGURE OF MERIT: {fom:.2f} <<<")
    
    if P > 0.00125:
        print("\nWARNING: Your average power P exceeds 0.00125!")
        print("The max(1, 800*P) denominator is actively penalizing your score.")

def run_sdecode():
    if not os.path.exists('rx.wav'):
        print("\nError: 'rx.wav' not found! Please run the ccplay command in your terminal first.")
        return
    if not os.path.exists('original_bits.npy'):
        print("\nError: 'original_bits.npy' not found! Please run the Encode step first.")
        return
    if not os.path.exists('tx.wav'):
        print("\nError: 'tx.wav' not found! It is needed to calculate your power and data rate.")
        return

    print("\nRunning decoder...")
    decoded_bits = dec()
    
    print("Loading original bits to check for errors...")
    original_bits = np.load('original_bits.npy')
    
    # Calculate Bit Errors
    N = np.sum(original_bits != decoded_bits)
    
    # find where errors are clustered
    error_positions = np.where(original_bits != decoded_bits)[0]
    if len(error_positions) > 0:
        print(f"First error at bit position: {error_positions[0]}")
        print(f"Last error at bit position: {error_positions[-1]}")
        print(f"Errors in first 10000 bits: {np.sum(original_bits[:10000] != decoded_bits[:10000])}")
        print(f"Errors in last 10000 bits: {np.sum(original_bits[-10000:] != decoded_bits[-10000:])}")
        print(f"Errors in middle 10000 bits: {np.sum(original_bits[95000:105000] != decoded_bits[95000:105000])}")

    # Read tx.wav to calculate Power and Data Rate
    fs, X_tx_int = wav.read('tx.wav')
    X_tx = X_tx_int / np.iinfo(np.int32).max
    P = np.mean(X_tx**2)
    
    duration_sec = len(X_tx) / fs
    R = 200000 / duration_sec
    
    # Calculate Figure of Merit
    numerator = min(R, 10000) * ((1 - N / 100000)**10)
    denominator = max(1, 800 * P)
    fom = numerator / denominator
    
    # Print the results
    print(f"\n--- Lab 5 Standard-Compliant System Performance ---")
    print(f"tx.wav Duration : {duration_sec:.4f} seconds")
    print(f"Data Rate (R)   : {R:.2f} bps")
    print(f"Avg Power (P)   : {P:.6f}")
    print(f"Total Errors (N): {N} out of 200,000")
    print(f"\n>>> FIGURE OF MERIT: {fom:.2f} <<<")
    
    if P > 0.00125:
        print("\nWARNING: Your average power P exceeds 0.00125!")
        print("The max(1, 800*P) denominator is actively penalizing your score.")    

if __name__ == "__main__":
    print("=================================")
    print("    Lab 5 Testing Interface")
    print("=================================")
    print("1: Run Encoder (Generates tx.wav)")
    print("2: Run Decoder (Grades rx.wav)")
    print("=================================")
    
    choice = input("Enter 1 or 2, and put prefix 's' for standard-compliant or 'p' for high-performance (example: 's1'): ").strip()
    
    if choice == '1':
        run_encode()
    elif choice == '2':
        run_decode()
    else:
        print("Invalid choice. Exiting.")