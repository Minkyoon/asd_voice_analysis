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

# Convert the frequencies to spectrogram bin indices
min_bin = librosa.hz_to_mel(min_freq)
max_bin = librosa.hz_to_mel(max_freq)

# Select the bins within the human voice frequency range
stft = stft[min_bin:max_bin, :]

# Convert the magnitude spectrogram to dB scale
mag = np.abs(stft)
log_mag = librosa.amplitude_to_db(mag, ref=np.max)

# Display the spectrogram
librosa.display.specshow(log_mag, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (dB)')
plt.show()
