### 첫번째 발화 시점 그래프에 표시시간 까지 표시됨 3번째가지 dtect
import librosa
import numpy as np
from fastdtw import fastdtw
from pathlib import Path
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm
import os

def display_spectrum(y, sr, title="Spectrum", save_path=None):
    stft = librosa.stft(y)
    mag = np.abs(stft)
    log_mag = librosa.amplitude_to_db(mag, ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 6))

    # We set the y-axis to 'linear' to show frequencies in Hz
    spec_display = librosa.display.specshow(log_mag, sr=sr, x_axis='time', y_axis='linear', ax=ax)
    plt.title(title, pad=20)

    # Detect onset times
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    print(onset_frames)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(onset_times)

    # Draw vertical lines at the first three onset times
    for i, onset_time in enumerate(onset_times[:1]):
        plt.axvline(x=onset_time, color='r')

        # Add text labels for the onset times
        plt.text(onset_time, ax.get_ylim()[1], f"Start {i+1}: {onset_time:.2f}s", color='r')

    cbar = plt.colorbar(spec_display, format='%+2.0f dB', ax=ax, pad=0.01)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()

    # Return the times of the first three onsets (or fewer if there were not three onsets)
    return onset_times[:3]




ROOT_PATH = Path('/home/minkyoon/2023_social')
file_paths = list(ROOT_PATH.glob('*.flac'))
onset_times = []
for i, file_path in tqdm(enumerate(file_paths, start=1)):
    y, sr = librosa.load(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = f'{file_name}_time.png'
    onset_time = display_spectrum(y, sr, title=f"Spectrum of {file_name}", save_path=save_path)
    onset_times.append((file_name, onset_time))

# Now 'onset_times' is a list of tuples, where each tuple is ('file_name', onset_time)
print(onset_times)


i