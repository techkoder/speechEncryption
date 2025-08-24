🔒 Speech Encryption using Logistic Map, Permutation & Masking
This project implements a speech encryption and decryption system based on chaotic maps, inspired by the research paper "A Speech Encryption based on Chaotic Maps" (IJCA, 2014).

The system encrypts .wav speech files using:

Key generation via the Logistic Map (chaos-based pseudorandom sequence).

Permutation & masking of speech samples to obscure intelligibility.

Symmetric decryption using the same chaotic keys.

📂 Project Structure
text
your-project-folder/
│
├── encrypter.py     # Encrypts the recording.wav input
├── decypter.py      # Decrypts the encrypted WAV file
├── .gitignore       # Git ignore rules
└── recording.wav    # Input file to be encrypted (user-provided)
Note: You must provide a recording.wav file in the same directory, or change the filename path inside encrypter.py.
The encryption key (x0) and chaotic parameter (r) are currently hardcoded in the scripts.

🛠 Requirements
Install the required Python libraries:

bash
pip install numpy librosa soundfile scipy
▶️ Usage
1. Encrypt speech
bash
python encrypter.py
Reads recording.wav (or your own file if path changed).

Produces an encrypted audio file (see encrypter.py for output filename).

2. Decrypt speech
bash
python decypter.py
Uses the same chaotic keys.

Produces a decrypted file (which should closely match the original speech).

⚙️ How It Works
Input: Reads recording.wav (mono or stereo, typically 16-bit PCM).

Key generation: Uses the Logistic Map equation:


where x0 (initial key) and r (control parameter) form the secret key.

Permutation: The sequence shuffles the order of speech samples.

Masking: Additional substitution (XOR/multiplicative chaos masking) is applied.

Output: The resulting encrypted signal is unintelligible without the key.

Decryption: Same process is reversed (requires exact x0 and r).

🔑 Key Notes
The secret key is defined by:<img width="736" height="138" alt="image" src="https://github.com/user-attachments/assets/bed190ad-2b15-4891-8dde-dd8643f40e55" />


Initial condition x0

Logistic map parameter r

Extremely sensitive: even tiny differences in these values will prevent proper decryption.

Current version: x0 and r are hardcoded – modify directly inside the Python scripts to experiment.

✅ Example Workflow
Place your speech file as recording.wav in the folder.

Run python encrypter.py → Generates encrypted audio.

Run python decypter.py → Recovers the original audio.

📚 References & Credits
This project is based on and inspired by the research paper:

Saad Najim Al Saad & Eman Hato, "A Speech Encryption based on Chaotic Maps",
International Journal of Computer Applications (0975 – 8887), Volume 93 – No.4, May 2014.

Additional concepts:

Chaotic Cryptography using Logistic Maps.

Permutation-substitution ciphers for audio encryption.

🚀 Future Enhancements
Allow custom input/output filenames via command-line arguments.

Support real-time speech encryption (microphone input).

Extend chaotic key generation methods (Tent map, Henon map, Cat map).

Add evaluation metrics (SNR, correlation, intelligibility tests).
