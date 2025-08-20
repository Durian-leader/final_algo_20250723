import math
import random
from pathlib import Path

import torch
import torchaudio
import soundfile as sf  # 仅用于保存 .wav，可替换 torchaudio.save

TARGET_SR = 8000                       # 目标采样率
SNR_RANGE = (-5, 20)                   # dB
EPS = 1e-10

def rms(signal: torch.Tensor) -> float:
    """计算 RMS 能量（单声道 Tensor）"""
    return torch.sqrt(torch.mean(signal ** 2) + EPS).item()

def scale_noise_to_snr(speech, noise, snr_db):
    """把噪声缩放到指定 SNR"""
    ps = rms(speech)
    pn = rms(noise)
    k = ps / (pn * 10 ** (snr_db / 20))
    return noise * k

def mix_one_pair(speech_wav, noise_wav, snr_db):
    """返回混合波形、语音标签（1）、静音标签（0）"""
    # ① 采样率统一 & 转单声道
    speech, sr_s = torchaudio.load(speech_wav)
    noise,  sr_n = torchaudio.load(noise_wav)

    if sr_s != TARGET_SR:
        speech = torchaudio.functional.resample(speech, sr_s, TARGET_SR)
    if sr_n != TARGET_SR:
        noise = torchaudio.functional.resample(noise, sr_n, TARGET_SR)

    speech = torch.mean(speech, dim=0)          # -> [T]
    noise  = torch.mean(noise,  dim=0)

    # ② 对齐长度：截断或循环填充噪声
    if noise.numel() < speech.numel():
        # 重复噪声补长
        repeat = math.ceil(speech.numel() / noise.numel())
        noise = noise.repeat(repeat)[:speech.numel()]
    else:
        # 随机裁切同长度
        start = random.randint(0, noise.numel() - speech.numel())
        noise = noise[start:start + speech.numel()]

    # ③ SNR 缩放 & 混合
    noise = scale_noise_to_snr(speech, noise, snr_db)
    mixture = speech + noise

    # ④ 防削波
    max_amp = mixture.abs().max().item()
    if max_amp > 1.0:
        mixture = mixture / max_amp * 0.99

    return mixture

def generate_dataset(
    speech_dir: Path,
    noise_dir: Path,
    out_wav_dir: Path,
    mixes_per_utt: int = 3,
):
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    speech_files = list(speech_dir.rglob("*.wav"))
    noise_files  = list(noise_dir.rglob("*.wav"))

    for spk_idx, speech_path in enumerate(speech_files):
        for n in range(mixes_per_utt):
            noise_path = random.choice(noise_files)
            snr = random.uniform(*SNR_RANGE)

            mix= mix_one_pair(speech_path, noise_path, snr)

            mix_name = f"{speech_path.stem}_mix{n}_snr{int(snr)}dB.wav"

            sf.write(out_wav_dir / mix_name, mix.numpy(), TARGET_SR, subtype="PCM_16")

            print(f"[✓] {mix_name}  (SNR={snr:.1f} dB)")

# ------------------ 执行示例 ------------------
if __name__ == "__main__":
    generate_dataset(
        speech_dir=Path("数据集/备用数据/music"),
        noise_dir=Path("数据集/备用数据/env"),
        out_wav_dir=Path("dataset/mixed_wav/music-env"),
        mixes_per_utt=5,          # 每条语音生成 5 个不同 SNR & 噪声
    )