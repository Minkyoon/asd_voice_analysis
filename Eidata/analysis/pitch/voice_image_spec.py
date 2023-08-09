import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('/home/minkyoon/2023_social/1650-173552-0008.flac')

# Compute the Short-time Fourier Transform (STFT)
D = librosa.stft(y)

# Get the frequencies corresponding to the rows in D
freqs = librosa.fft_frequencies(sr=sr, n_fft=D.shape[0])

# Define the frequency range in Hz for human voice
min_freq = 85.0
max_freq = 255.0

# Find the indices of the bins within the human voice frequency range
indices = np.where((freqs >= min_freq) & (freqs <= max_freq))

# Select the portion of D corresponding to these bins
D_voice = D[indices]

# Convert the magnitude spectrogram to dB scale for all and voice only spectrograms
mag = np.abs(D)
log_mag = librosa.amplitude_to_db(mag, ref=np.max)

mag_voice = np.abs(D_voice)
log_mag_voice = librosa.amplitude_to_db(mag_voice, ref=np.max)

# Display the full spectrogram
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
librosa.display.specshow(log_mag, sr=sr, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Full Spectrogram (dB)')

# Display the voice range spectrogram
plt.subplot(2, 1, 2)
librosa.display.specshow(log_mag_voice, sr=sr, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Voice Range Spectrogram (dB)')
plt.tight_layout()
plt.show()



import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('/home/minkyoon/2023_social/1650-173552-0008.flac')

# Compute the Short-time Fourier Transform (STFT)
stft = librosa.stft(y)

# Define the frequency range in Hz for human voice
min_freq = 85.0
max_freq = 255.0

# Convert the magnitude spectrogram to dB scale
mag = np.abs(stft)
log_mag = librosa.amplitude_to_db(mag, ref=np.max)

# Display the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mag, sr=sr, x_axis='time', y_axis='hz')
plt.ylim(min_freq, max_freq) # Limit the frequency range to human voice
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (dB) within Human Voice Frequency Range')
plt.tight_layout()
plt.show()
