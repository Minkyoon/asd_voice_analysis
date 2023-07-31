import librosa
import numpy as np
from fastdtw import fastdtw
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
def display_spectrum(y, sr, save_path=None):
    # Compute the spectrogram of the input signal ‘y’. The short-time Fourier transform (STFT) is a technique for transforming a signal in the time domain to the frequency domain.
    stft = librosa.stft(y)
    # Compute the absolute value of the STFT to get the amplitude of the frequency component.
    mag = np.abs(stft)
    # Converts the amplitude value to decibels (dB). This expresses the relative magnitude of the amplitudes on a logarithmic scale.
    log_mag = librosa.amplitude_to_db(mag, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    # sr = sampling rate
    spec_display = librosa.display.specshow(log_mag, sr=sr, x_axis='time', y_axis='linear', ax=ax)
    #plt.title(title)
    #plt.colorbar(spec_display, format=‘%+2.0f dB’, ax=ax)
    plt.tight_layout()
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()
ROOT_PATH = Path('/home/minkyoon/2023_social')
# .wav로 끝나는 파일 전부 로드
file_paths = list(ROOT_PATH.glob('*.flac'))
for i, file_path in tqdm(enumerate(file_paths, start=1)):
    y, sr = librosa.load(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = f'/home/minkyoon/2023_social/{file_name}.png'
    # display_spectrum(y, sr, f”Spectrum of {file_path}“, save_path=f”/home/joohyun/joohyun/mobile_project/EEE7221-Team5/Result_for_CNN/Sample1WAV/spectrum{i}.png”)
    display_spectrum(y, sr,save_path=save_path)
    
