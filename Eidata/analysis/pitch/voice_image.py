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
D = D[indices]

# Convert the magnitude spectrogram to dB scale
mag = np.abs(D)
log_mag = librosa.amplitude_to_db(mag, ref=np.max)

# Display the spectrogram
librosa.display.specshow(log_mag, sr=sr, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (dB)')
plt.show()



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
D = D[indices]

# Convert the magnitude spectrogram to dB scale
mag = np.abs(D)
log_mag = librosa.amplitude_to_db(mag, ref=np.max)

# Display the spectrogram
librosa.display.specshow(log_mag, sr=sr, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (dB)')
plt.show()
