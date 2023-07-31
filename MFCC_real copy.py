import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os

def display_mfcc(y, sr, title="MFCC", save_path=None):
    # MFCC를 계산합니다
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # MFCC를 dB 단위로 변환합니다
    mfccs_db = librosa.amplitude_to_db(mfccs)

    fig, ax = plt.subplots(figsize=(10, 4))

    # MFCC를 그립니다
    img = librosa.display.specshow(mfccs_db, sr=sr, x_axis='time', ax=ax)

    # 그래프에 색상 막대를 추가합니다
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # 그래프의 제목을 설정합니다
    ax.set(title=title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()

ROOT_PATH = Path('/home/minkyoon/2023_social')
# .flac으로 끝나는 모든 파일을 찾습니다
file_paths = list(ROOT_PATH.glob('*.flac'))

for i, file_path in tqdm(enumerate(file_paths, start=1)):
    # 각 오디오 파일을 로드합니다
    y, sr = librosa.load(file_path)

    # 저장할 파일의 이름을 생성합니다
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = f'/home/minkyoon/2023_social/{file_name}_mfcc.png'

    # MFCC를 그리고 파일로 저장합니다
    display_mfcc(y, sr, title=f"MFCC of {file_name}", save_path=save_path)
