import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
y, sr = librosa.load('/home/minkyoon/2023_social/1650-173552-0008.flac')

#Display the log-scaled spectrum
stft = librosa.stft(y)
mag = np.abs(stft)
log_mag = librosa.amplitude_to_db(mag, ref=np.max)
librosa.display.specshow(log_mag, sr=sr, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (dB)')
plt.show()

# Compute the MFCCs
print(type(y))

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
# Display the MFCCs
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.show()

# Compute the pitch
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)




# Human voice frequency range in Hz
min_freq = 85.0
max_freq = 255.0

# Create a mask for pitches in the human voice range
pitch_mask = (pitches >= min_freq) & (pitches <= max_freq)

# Filter out pitches and magnitudes that are outside the human voice frequency range
pitches_in_range = pitches[pitch_mask]
magnitudes_in_range = magnitudes[pitch_mask]

# Compute the mean pitch
mean_pitch = pitches_in_range[magnitudes_in_range > np.median(magnitudes_in_range)].mean()
print(f"Mean pitch: {mean_pitch:.2f}Hz")

# Compute the pitch variability
pitch_variability = np.std(pitches_in_range[magnitudes_in_range > np.median(magnitudes_in_range)])
print(f"Pitch variability: {pitch_variability:.2f}Hz")


pitch_range = pitches_in_range.max() - pitches_in_range.min()
print(f"Pitch range: {pitch_range:.2f}Hz")