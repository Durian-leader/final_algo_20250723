import os
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
import torchaudio
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# 加载动态库
mfcc_lib = ctypes.CDLL("./mfcc_c/libmfcc.so")

# 配置函数参数类型
mfcc_lib.extract_mfcc_from_float.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # 输入帧
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")   # 输出MFCC
]

# 参数设置
sample_rate = 8000
frame_duration = 0.032  # 32ms
frame_size = int(sample_rate * frame_duration)

step_size = 1
mfcc_dim = 13

    

def extract_mfcc_per_file(wav_path: Path, step_size):
    wav_path = Path(wav_path)
    
    try:
        waveform, sr = torchaudio.load(wav_path)
        assert sr == sample_rate, f"Sample rate mismatch: expected {sample_rate}, got {sr}"
        signal = waveform[0].numpy()

        # 分帧
        num_frames = (len(signal) - frame_size) // step_size + 1
        mfcc_features = np.zeros((num_frames, mfcc_dim), dtype=np.float32)

        for i in range(num_frames):
            start = i * step_size
            frame = signal[start:start + frame_size].astype(np.float32)
            output = np.zeros(mfcc_dim, dtype=np.float32)
            mfcc_lib.extract_mfcc_from_float(frame, output)
            mfcc_features[i, :] = output

        # 保存为 .npy
        save_path = wav_path.with_suffix(".npy")
        np.save(save_path, mfcc_features.T)
        return mfcc_features.T

    except Exception as e:
        print(f"[ERROR] {wav_path}: {e}")
        return None
def extract_mfcc_for_folder(wav_dir: Path, step_size:int = 256):
    wav_paths = list(Path(wav_dir).rglob("*.wav"))

    print(f"Total WAV files: {len(wav_paths)}")

    args_list = [(wav_path, step_size) for wav_path in wav_paths]

    with Pool() as pool:
        results = list(tqdm(pool.starmap(extract_mfcc_per_file, args_list), total=len(args_list)))



if __name__ == "__main__":
    wav_dir = "dataset"
    extract_mfcc_for_folder(Path(wav_dir))

