import librosa
import numpy as np
import soundfile as sf
from scipy.fftpack import dct
import time

def logistic_map(x, r, n=1):
    """
    Implements the logistic map chaotic function
    x: initial value (0-1)
    r: parameter (typically 3.57-4.0 for chaotic behavior)
    n: number of iterations
    """
    if n > 1:
        sequence = []
        for _ in range(n):
            x = r * x * (1 - x)
            sequence.append(x)
        return np.array(sequence)
    else:
        return r * x * (1 - x)

def generate_key(x, r1, r2, key_length):
    """
    Generate a permutation key using dual chaotic maps
    """
    chaotic = []
    for i in range(0, key_length, 2):
        x = logistic_map(x, r1)
        chaotic.append(x)
        if i+1 < key_length:  # Ensure we don't exceed key_length
            y2 = logistic_map(x, r2)
            chaotic.append(y2)
            x = y2

    chaotic_array = np.array(chaotic[:key_length])  # Ensure correct length
    
    # Generate permutation sequence
    key = []
    for _ in range(key_length):
        if len(chaotic_array) == 0:
            break
        j = np.argwhere(chaotic_array == np.min(chaotic_array))[0][0]
        key.append(j % key_length)
        chaotic_array[j] = 1  # Mark as used
    
    return key

def permute_audio(audio_data, key):
    """
    Apply permutation to audio data using the key
    """
    permuted_audio = np.zeros_like(audio_data)
    key_length = len(key)
    
    if key_length > len(audio_data):
        key = key[:len(audio_data)]
    elif key_length < len(audio_data):
        # Extend key if needed
        key = np.resize(key, len(audio_data))
    
    for i, k in enumerate(key):
        if i < len(permuted_audio) and k < len(audio_data):
            permuted_audio[i] = audio_data[k]
    
    return permuted_audio

def generate_mask_key(audio_len):
    """
    Generate a bit mask key using multiple chaotic maps
    """
    r_values = [3.7, 3.8, 3.9, 3.6, 3.85, 3.95]
    x_values = [0.2, 0.4, 0.6, 0.3, 0.5, 0.7]
    
    # Generate chaotic sequences
    logistic_maps = []
    for x, r in zip(x_values, r_values):
        logistic_maps.append(logistic_map(x, r, audio_len))
    
    # Create binary arrays
    A = np.zeros((6, audio_len), dtype=np.int8)
    for i in range(6):
        A[i] = (logistic_maps[i] >= 0.5).astype(np.int8) if i < 3 else (logistic_maps[i] <= 0.5).astype(np.int8)
    
    # Create mask bits using XOR operations
    mask_bits = np.zeros((8, audio_len), dtype=np.int8)
    mask_bits[0] = A[0] ^ A[4] ^ A[2]
    mask_bits[1] = A[1] ^ A[3] ^ A[4]
    mask_bits[2] = A[2] ^ A[1] ^ A[5]
    mask_bits[3] = A[3] ^ A[1] ^ A[5]
    mask_bits[4] = A[4] ^ A[0] ^ A[2]
    mask_bits[5] = A[5] ^ A[4] ^ A[2]
    mask_bits[6] = A[0] ^ A[1] ^ A[2]
    mask_bits[7] = A[3] ^ A[4] ^ A[5]

    # Convert bit arrays to integers
    mask_key = np.packbits(mask_bits.T, axis=-1).astype(np.int16).flatten()
    
    # Ensure the mask is the right length
    if len(mask_key) < audio_len:
        mask_key = np.resize(mask_key, audio_len)
    else:
        mask_key = mask_key[:audio_len]
        
    return mask_key

def xor_mask(data, mask_key):
    """
    Apply XOR masking to the data
    """
    if len(mask_key) != len(data):
        mask_key = np.resize(mask_key, len(data))
    
    # Convert to 16-bit integers for bitwise operations
    int_data = (data * 32767).astype(np.int16)
    xor_result = np.bitwise_xor(int_data, mask_key)
    
    # Convert back to float for audio processing
    float_result = xor_result.astype(np.float32) / 32767
    return float_result

def encrypt_audio(file_path, output_path, initial_condition, r1, r2):
    """
    Encrypt audio using permutation, DCT transform and XOR masking
    """
    start_time = time.time()
    print("LOADING THE AUDIO")
    audio_data, sample_rate = librosa.load(file_path, sr=16000)  # Use 16000 Hz for PESQ compatibility
    key_length = len(audio_data)
    
    print("GENERATING THE FIRST KEY FOR ENCRYPTION")
    key = generate_key(initial_condition, r1, r2, key_length)
    print("GENERATING THE SECOND KEY FOR ENCRYPTION")
    key2 = permute_audio(key, key)
    print("GENERATING THE MASKING XOR KEY FOR ENCRYPTION")
    mask_key = generate_mask_key(key_length)
    
    print("STARTING TO ENCRYPT THE AUDIO")
    # Step 1: Apply first permutation
    permuted_audio = permute_audio(audio_data, key)
    
    # Step 2: Apply DCT transform to permuted audio
    dct_data = dct(permuted_audio, norm='ortho')
    
    # Step 3: Apply second permutation to DCT data
    final_permuted_audio = permute_audio(dct_data, key2)
    
    # Step 4: Apply XOR masking
    final_audio = xor_mask(final_permuted_audio, mask_key)
    
    sf.write(output_path, final_audio, sample_rate)
    end_time = time.time()
    print(f"ENCRYPTED AUDIO IS SAVED TO: {output_path}")
    print(f"Encryption completed in {end_time - start_time:.2f} seconds")

def main():
    # Direct variable declarations instead of using argparse
    file_path = "sample.wav"  # Input audio file path
    output_path = "output_encrypted_audio.wav"  # Output encrypted file path
    initial_condition = 0.5  # Initial condition for chaotic map
    r1 = 3.85312  # First bifurcation parameter
    r2 = 3.84521  # Second bifurcation parameter
    
    # Execute encryption
    encrypt_audio(file_path, output_path, initial_condition, r1, r2)

if __name__ == "__main__":
    main()