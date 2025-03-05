import numpy as np
from scipy.io.wavfile import write, read
import pywt  # For Discrete Wavelet Transform

# Logistic map to generate a chaotic sequence
def logistic_map(x, r, n):
    sequence = []
    for _ in range(n):
        x = r * x * (1 - x)
        sequence.append(x)
    return np.array(sequence)

# Function to generate mask key based on six logistic maps
def generate_mask_key(audio_len):
    # Initialize the six logistic maps
    r_values = [3.7, 3.8, 3.9, 3.6, 3.85, 3.95]
    x_values = [0.2, 0.4, 0.6, 0.3, 0.5, 0.7]
    logistic_maps = [logistic_map(x, r, audio_len) for x, r in zip(x_values, r_values)]
    
    # Convert real values to bits using the described conditions
    A = np.zeros((6, audio_len), dtype=np.int8)
    
    for i in range(6):
        if i < 3:
            A[i] = logistic_maps[i] >= 0.5  # For A1, A2, A3 (Condition 9)
        else:
            A[i] = logistic_maps[i] <= 0.5  # For A4, A5, A6 (Condition 10)
    
    # Generate 8-bit mask from the A values using XOR operations
    mask_bits = np.zeros((8, audio_len), dtype=np.int8)
    mask_bits[0] = A[0] ^ A[4] ^ A[2]  # Bit1 = A1 ⊕ A5 ⊕ A3
    mask_bits[1] = A[1] ^ A[3] ^ A[4]  # Bit2 = A2 ⊕ A4 ⊕ A5
    mask_bits[2] = A[2] ^ A[1] ^ A[5]  # Bit3 = A3 ⊕ A2 ⊕ A6
    mask_bits[3] = A[3] ^ A[1] ^ A[5]  # Bit4 = A4 ⊕ A2 ⊕ A6
    mask_bits[4] = A[4] ^ A[0] ^ A[2]  # Bit5 = A5 ⊕ A1 ⊕ A3
    mask_bits[5] = A[5] ^ A[4] ^ A[2]  # Bit6 = A6 ⊕ A5 ⊕ A3
    mask_bits[6] = A[0] ^ A[1] ^ A[2]  # Bit7 = A1 ⊕ A2 ⊕ A3
    mask_bits[7] = A[3] ^ A[4] ^ A[5]  # Bit8 = A4 ⊕ A5 ⊕ A6

    # Combine 8 bits into an integer mask
    mask_key = np.packbits(mask_bits.T, axis=-1).astype(np.int16).flatten()
    return mask_key

# Function to save keys to a text file
def save_key_to_file(key1, key2, mask_key, filename="key.txt"):
    with open(filename, "w") as f:
        f.write(','.join(map(str, key1)) + "\n")
        f.write(','.join(map(str, key2)) + "\n")
        f.write(','.join(map(str, mask_key)) + "\n")
    print(f"Keys saved to {filename}")

# Function to load keys from a text file
def load_key_from_file(filename="key.txt"):
    with open(filename, "r") as f:
        key1 = np.array(list(map(float, f.readline().split(','))))
        key2 = np.array(list(map(float, f.readline().split(','))))
        mask_key = np.array(list(map(float, f.readline().split(',')))).astype(np.int16)
    print(f"Keys loaded from {filename}")
    return key1, key2, mask_key

# XOR Masking function
def xor_mask(data, mask_key):
    return np.bitwise_xor(data.astype(np.int16), mask_key)

# Encryption function with DWT, second permutation, and masking
def encryptor(audio_file):
    # Load the audio file
    samplerate, audio = read(audio_file)
    
    # First permutation
    r = 3.999
    x0 = 0.5
    audio_len = len(audio)
    perm_key1 = logistic_map(x0, r, audio_len)
    perm_order1 = np.argsort(perm_key1)
    
    # Scramble the audio
    permuted_audio = audio[perm_order1]
    
    # Apply Discrete Wavelet Transform (DWT)
    coeffs = pywt.wavedec(permuted_audio, 'db1', level=2)
    wavelet_transformed = np.concatenate(coeffs)
    
    # Second permutation on wavelet coefficients
    perm_key2 = logistic_map(0.3, 3.8, len(wavelet_transformed))
    perm_order2 = np.argsort(perm_key2)
    scrambled_wavelet = wavelet_transformed[perm_order2]
    
    # Generate mask key and apply XOR masking
    mask_key = generate_mask_key(len(scrambled_wavelet))
    masked_audio = xor_mask(scrambled_wavelet, mask_key)
    
    # Save the masked, scrambled audio
    encrypted_filename = "encrypted_sample.wav"
    write(encrypted_filename, samplerate, masked_audio.astype(np.int16))
    print(f"Encrypted audio saved as {encrypted_filename}")
    
    # Save keys to file
    save_key_to_file(perm_order1, perm_order2, mask_key)

# Decryption function to reverse DWT, permutations, and masking
def decryptor(encrypted_file):
    # Load the encrypted audio file
    samplerate, masked_audio = read(encrypted_file)
    
    # Load the permutation keys and mask key
    perm_order1, perm_order2, mask_key = load_key_from_file()
    
    # Remove XOR masking
    unmasked_audio = xor_mask(masked_audio, mask_key)
    
    # Reverse the second permutation (wavelet domain)
    inverse_perm_order2 = np.argsort(perm_order2)
    unscrambled_wavelet = unmasked_audio[inverse_perm_order2]
    
    # Inverse Discrete Wavelet Transform (IDWT)
    wavelet_coeff_lengths = [len(arr) for arr in pywt.wavedec(np.zeros(len(unscrambled_wavelet)), 'db1', level=2)]
    coeffs_split = np.split(unscrambled_wavelet, np.cumsum(wavelet_coeff_lengths)[:-1])
    reconstructed_audio = pywt.waverec(coeffs_split, 'db1')
    
    # Reverse the first permutation (time domain)
    inverse_perm_order1 = np.argsort(perm_order1)
    decrypted_audio = reconstructed_audio[inverse_perm_order1]
    
    # Save the decrypted audio
    decrypted_filename = "decrypted_sample.wav"
    write(decrypted_filename, samplerate, decrypted_audio.astype(np.int16))
    print(f"Decrypted audio saved as {decrypted_filename}")

# Main function to provide options to encrypt or decrypt
def main():
    choice = input("Enter 'e' to encrypt or 'd' to decrypt: ").lower()
    if choice == 'e':
        encryptor("sample.wav")  # Encrypt sample.wav
    elif choice == 'd':
        decryptor("encrypted_sample.wav")  # Decrypt encrypted_sample.wav
    else:
        print("Invalid choice. Please enter 'e' or 'd'.")

if _name_ == "_main_":
    main()