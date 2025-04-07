import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

# 创建总进度条
total_steps = 5  # 总共5个主要步骤：加载音频、STFT计算、梅尔频谱计算、色度频谱计算、可视化
main_pbar = tqdm(total=total_steps, desc="总体进度", position=0)

audio_path = '/data/home/luyj/content/lab2.wav' 
print("正在加载音频文件...")
with tqdm(total=100, desc="加载音频", position=1, leave=False) as pbar:
    y, sr = librosa.load(audio_path, sr=22050)
    pbar.update(100)
main_pbar.update(1)

print("正在计算短时傅里叶变换...")
with tqdm(total=100, desc="计算STFT", position=1, leave=False) as pbar:
    D = librosa.stft(y, n_fft=2048, hop_length=1024, window='hamming')
    pbar.update(50)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    pbar.update(50)
main_pbar.update(1)

# 计算梅尔频谱
print("正在计算梅尔频谱...")
with tqdm(total=100, desc="计算梅尔频谱", position=1, leave=False) as pbar:
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    pbar.update(50)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    pbar.update(50)
main_pbar.update(1)

# 计算色度频谱
print("正在计算色度频谱...")
with tqdm(total=100, desc="计算色度频谱", position=1, leave=False) as pbar:
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=1024)
    pbar.update(100)
main_pbar.update(1)

# 可视化声谱图、梅尔频谱、色度频谱
print("正在生成可视化图表...")
with tqdm(total=100, desc="生成图表", position=1, leave=False) as pbar:
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
    pbar.update(20)

    # 声谱图
    img0 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax[0])
    ax[0].set_title('Spectrogram with Hamming Window')
    fig.colorbar(img0, ax=ax[0], format='%+2.0f dB')
    pbar.update(20)

    # 梅尔频谱
    img1 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax[1])
    ax[1].set_title('Mel Spectrogram')
    fig.colorbar(img1, ax=ax[1], format='%+2.0f dB')
    pbar.update(20)

    # 色度频谱
    img2 = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=ax[2])
    ax[2].set_title('Chroma Spectrogram')
    fig.colorbar(img2, ax=ax[2])
    pbar.update(20)

    plt.tight_layout()
    plt.savefig('/data/home/luyj/content/audio_analysis.png')  # 保存图像到指定路径
    pbar.update(20)
main_pbar.update(1)

main_pbar.close()
print("处理完成！图像已保存到: /data/home/luyj/content/audio_analysis.png")