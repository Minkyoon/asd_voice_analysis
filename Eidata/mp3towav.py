import av
import numpy as np
import soundfile as sf

def convert_mp3_to_wav(mp3_path, wav_path):
    input_container = av.open(mp3_path)
    audio_stream = next(s for s in input_container.streams if s.type == 'audio')
    audio_frames = []

    for packet in input_container.demux(audio_stream):
        for frame in packet.decode():
            audio_frames.append(frame.to_ndarray().flatten())
    
    audio_data = np.concatenate(audio_frames)
    sf.write(wav_path, audio_data, audio_stream.rate)

convert_mp3_to_wav("/home/minkyoon/2023_social/data/ebs/short.mp3", "/home/minkyoon/2023_social/data/ebs/short.wav")




## ffmpeg 관리자 권한이 없어서 사용못함
from pydub import AudioSegment

sound = AudioSegment.from_mp3("/home/minkyoon/2023_social/data/ebs/opinion.mp3")
sound.export("/home/minkyoon/2023_social/data/ebs/opinion.wav", format="wav")
