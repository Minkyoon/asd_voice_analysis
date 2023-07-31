import librosa
import numpy as np
from fastdtw import fastdtw
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
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
    
    cbar = plt.colorbar(spec_display, format='%+2.0f dB', ax=ax, pad=0.01)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()

ROOT_PATH = Path('/home/minkyoon/2023_social')
file_paths = list(ROOT_PATH.glob('*.flac'))
for i, file_path in tqdm(enumerate(file_paths, start=1)):
    y, sr = librosa.load(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = f'/home/minkyoon/2023_social/{file_name}.png'
    display_spectrum(y, sr, title=f"Spectrum of {file_name}", save_path=save_path)
