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

# Key generation using the logistic map
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

# Load the audio file
def load_audio(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    return audio_data, sample_rate

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

# Save the permuted audio to a file
def save_audio(file_path, audio_data, sample_rate):
    sf.write(file_path, audio_data, sample_rate)

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
    if len(mask_key) != len(data):
        mask_key = np.resize(mask_key, len(data))
    
    int_data = (data * 32767).astype(np.int16)
    xor_result = np.bitwise_xor(int_data, mask_key)
    float_result = xor_result.astype(np.float32) / 32767
    return float_result

# Encrypt audio using DCT and bitwise XOR
def encrypt_audio(file_path, output_path, initial_condition, r1, r2):
    # Load audio
    audio_data, sample_rate = load_audio(file_path)

    # Generate key based on audio length
    key_length = len(audio_data)
    key = generate_key(initial_condition, r1, r2, key_length)
    key2 = permute_audio(key,key)
    mask_key = generate_mask_key(key_length)
    # print(key_length)
    # Permute the audio using the generated key
    permuted_audio = permute_audio(audio_data, key)

    # Apply DCT to permuted audio
    dct_data = dct(permuted_audio, norm='ortho')
    finalPermutedAudio = permute_audio(dct_data,key2)
    finalAudio = xor_mask(finalPermutedAudio,mask_key)
    # finalAudio1 = idct(finalPermutedAudio)
    
    save_audio(output_path, finalAudio, sample_rate)
    print(f"Encrypted audio saved to {output_path}")

def main():
# Example usage for encryption
    file_path = 'sample.wav'
    output_path = 'output_encrypted_audio.wav'

# Parameters for the logistic map
    initial_condition = 0.5
    r1 = 3.85312  
    r2 = 3.84521  

# Encrypt the audio
    encrypt_audio(file_path, output_path, initial_condition, r1, r2)

main()