import librosa
import numpy as np
import soundfile as sf
from scipy.fftpack import dct, idct

# Logistic map function
def logistic_map(x, r,n=1):
    if n >1:
        sequence = []
        for _ in range(n):
            x = r * x * (1 - x)
            sequence.append(x)
        return np.array(sequence)
    else:
        return r * x * (1 - x)
    

# Generate key using the same method as encryption
def generate_key(x, r1, r2, key_length):
    chaotic = []
    for i in range(0, key_length, 2):
        x = logistic_map(x, r1)
        chaotic.append(x)
        y2 = logistic_map(x, r2)
        chaotic.append(y2)
        x = y2

    chaotic_array = np.array(chaotic)
    key = []
    for _ in range(key_length):
        j = np.argwhere(chaotic_array == np.min(chaotic_array))[0][0]
        key.append(j %key_length)
        chaotic_array[j] = 1  # Replace smallest value to find the next
    return key

def xor_mask(data, mask_key):
    if len(mask_key) != len(data):
        mask_key = np.resize(mask_key, len(data))
    
    int_data = (data * 32767).astype(np.int16)
    xor_result = np.bitwise_xor(int_data, mask_key)
    float_result = xor_result.astype(np.float32) / 32767
    return float_result

# Permute the audio based on the generated key
def permute_audio(audio_data, key):
    permuted_audio = np.zeros_like(audio_data)
    key_length = len(key)

    # Ensure key length matches the length of the audio
    if key_length > len(audio_data):
        key = key[:len(audio_data)]
    
    for i, k in enumerate(key):
        permuted_audio[i] = audio_data[k]

    return permuted_audio

# Inverse permute an array based on a given key
def inverse_permute_data(data, key):
    inverse_permuted_data = np.zeros_like(data)
    for i, k in enumerate(key):
        inverse_permuted_data[k] = data[i]
    return inverse_permuted_data

# Generate mask bits using the same method as encryption
def generate_mask_key(audio_len):
    # Initialize the six logistic maps
    r_values = [3.7, 3.8, 3.9, 3.6, 3.85, 3.95]
    x_values = [0.2, 0.4, 0.6, 0.3, 0.5, 0.7]
    logistic_maps = [logistic_map(x, r, audio_len) for x, r in zip(x_values, r_values)]
    
    # Convert real values to bits using the described conditions
    A = np.zeros((6, audio_len), dtype=np.int8)
    
    for i in range(6):
        if i < 3:
            A[i] = logistic_maps[i] >= 0.5  
        else:
            A[i] = logistic_maps[i] <= 0.5  
    
    # Generate 8-bit mask from the A values using XOR operations
    mask_bits = np.zeros((8, audio_len), dtype=np.int8)
    mask_bits[0] = A[0] ^ A[4] ^ A[2]  
    mask_bits[1] = A[1] ^ A[3] ^ A[4]  
    mask_bits[2] = A[2] ^ A[1] ^ A[5]  
    mask_bits[3] = A[3] ^ A[1] ^ A[5]  
    mask_bits[4] = A[4] ^ A[0] ^ A[2] 
    mask_bits[5] = A[5] ^ A[4] ^ A[2]  
    mask_bits[6] = A[0] ^ A[1] ^ A[2]  
    mask_bits[7] = A[3] ^ A[4] ^ A[5]  

    # Combine 8 bits into an integer mask
    mask_key = np.packbits(mask_bits.T, axis=-1).astype(np.int16).flatten()
    return mask_key


def xor_mask(data, mask_key):
    return np.bitwise_xor(data.astype(np.int16), mask_key)

# Perform inverse DCT on the permuted coefficients
def apply_idct(dct_data):
    return idct(dct_data, norm='ortho')

# Load the audio file
def load_audio(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    return audio_data, sample_rate

# Save the audio file
def save_audio(file_path, audio_data, sample_rate):
    sf.write(file_path, audio_data, sample_rate)

# Decryption function
def decrypt_audio(file_path, output_path, initial_condition, r1, r2):
    # Load encrypted audio
    encrypted_data, sample_rate = load_audio(file_path)

    # Generate key based on encrypted data length
    key_length = len(encrypted_data)
    key = generate_key(initial_condition, r1, r2, key_length)
    key2 = permute_audio(key, key)
    mask_key = generate_mask_key(key_length)
    # print(key_length)
    # Apply inverse encryption to recover the audio signal
    # recovered_audio = dct(frecovered_audio)
    frecovered_audio = xor_mask(encrypted_data,mask_key)
    frecovered_audio2 = inverse_permute_data(frecovered_audio, key2)
    frecovered_audio3 = idct(frecovered_audio2, norm='ortho')
    finalRecoveredAudio = inverse_permute_data(frecovered_audio3,key)

    # Save the recovered audio
    save_audio(output_path, finalRecoveredAudio, sample_rate)
    print(f"Decrypted audio saved to {output_path}")

def main():
    file_path = 'output_encrypted_audio.wav'
    output_path = 'output_decrypted_audio.wav'

    initial_condition = 0.5
    r1 = 3.85312  
    r2 = 3.84521
    decrypt_audio(file_path, output_path, initial_condition, r1, r2)

main()