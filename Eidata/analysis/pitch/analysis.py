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
# Compute the mean pitch
mean_pitch = pitches[(500 > magnitudes) & (magnitudes > np.median(magnitudes))].mean()
print(f"Mean pitch: {mean_pitch:.2f}Hz")

mean_pitch = pitches[magnitudes > np.median(magnitudes)].mean()
print(f"Mean pitch: {mean_pitch:.2f}Hz")

# Compute the duration
duration = librosa.get_duration(y=y, sr=sr)
print(f"Duration: {duration:.2f}s")

pitch_variability = np.std(pitches[magnitudes > np.median(magnitudes)])
print(f"Pitch variability: {pitch_variability:.2f}Hz")




