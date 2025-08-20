import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import dct
import numpy as np
# 加载真实语音文件
waveform, sample_rate = torchaudio.load(torchaudio.utils.download_asset(
    "tutorial-assets/steam-train-whistle-daniel_simon.wav"))
waveform = waveform[0:1, :]  # ✅ 保证是单通道（shape: 1 × N）

# 取前2秒
waveform = waveform[:, :sample_rate * 2]

# 提取 log-mel 能量
mel_spec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=400,
    hop_length=160,
    n_mels=40
)
db_transform = torchaudio.transforms.AmplitudeToDB()

logmel = db_transform(mel_spec_transform(waveform)).squeeze(0).T  # shape: (frames, mels)

# DCT 得到 MFCC（保留前13维）
mfcc = dct(logmel.numpy(), type=2, norm='ortho', axis=1)[:, :13]

# 画热力图，00 在左下角
plt.figure(figsize=(3, 2), dpi=150)
sns.heatmap(
    mfcc.T[::-1],
    cmap="magma",
    xticklabels=20,
    yticklabels=np.arange(12, -1, -1)
)
plt.title("MFCCs from Real Audio (Train Whistle)")
plt.xlabel("Frame Index (Time)")
plt.ylabel("MFCC Coefficient Index")
plt.tight_layout()
plt.savefig('mfcc.png')
plt.show()