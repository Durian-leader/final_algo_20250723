import torch
import argparse
def extract_mfcc_windows(A, window_size=30, window_step=1):
    """
    A: [mfcc_len, time_len] 的 MFCC 特征
    返回: [N, window_size, mfcc_len]
    """
    mfcc_len, time_len = A.shape
    windows = []

    for start in range(0, time_len - window_size + 1, window_step):
        end = start + window_size
        mfcc_window = A[:, start:end].T  # [window_size, mfcc_len]
        windows.append(mfcc_window)

    return torch.stack(windows)  # [N, window_size, mfcc_len]



def window_soft_labels(probs, window_size=30, window_step=1):
    """
    probs: [prob_seq_len] 的软标签序列
    返回: [N, window_size]，每个窗口内的标签子序列
    """
    prob_seq_len = probs.shape[0]
    result = []
    
    for start in range(0, prob_seq_len - window_size + 1, window_step):
        end = start + window_size
        window = probs[start:end]
        result.append(window)

    return torch.stack(result)  # [N, window_size]



from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from multiprocessing import Pool, cpu_count
import os
# 顶层函数，必须放在模块最顶层
def process_file_wrapper(args):
    return process_file(*args)

def process_file(wav_path, window_size=30, window_step=1):
    txt_path = wav_path.with_suffix(".txt")
    npy_path = wav_path.with_suffix(".npy")
    try:
        mfcc_np = np.load(npy_path)
        mfcc_tensor = torch.tensor(mfcc_np, dtype=torch.float32)

        with open(txt_path, "r") as f:
            probs = [float(line.strip().split(",")[2]) for line in f]
        probs_tensor = torch.tensor(probs, dtype=torch.float32)

        time_len = mfcc_tensor.shape[1]
        if time_len < window_size:
            return None, None, f"{wav_path} - too short (length={time_len})"

        features = extract_mfcc_windows(mfcc_tensor, window_size, window_step)
        labels = window_soft_labels(probs_tensor, window_size, window_step)

        N = min(features.shape[0], labels.shape[0])
        return features[:N], labels[:N], None
    except Exception as e:
        skipped_len = mfcc_tensor.shape[1] if 'mfcc_tensor' in locals() else "unknown"
        return None, None, f"{wav_path} - error: {e} (length={skipped_len})"


def process_dataset(wav_dir, window_size=30, window_step=1):
    wav_paths = list(Path(wav_dir).rglob("*.wav"))
    args = [(path, window_size, window_step) for path in wav_paths]

    input_samples = []
    soft_labels = []
    skipped_files = []

    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_file_wrapper, args), total=len(args)))

    for features, labels, error in results:
        if error:
            skipped_files.append(error)
        elif features is not None:
            input_samples.append(features)
            soft_labels.append(labels)

    if skipped_files:
        print("\nSkipped files:")
        for item in skipped_files:
            print(item)
        with open(Path(wav_dir) / "skipped_files.txt", "w") as f:
            for item in skipped_files:
                f.write(item + "\n")

    if not input_samples:
        mfcc_dim = 13
        return torch.empty(0, window_size, mfcc_dim), torch.empty(0, window_size)

    return torch.cat(input_samples, dim=0), torch.cat(soft_labels, dim=0)


import torch
from pathlib import Path
def compute_and_concat_diff(input_path: str, output_path: str):
    """
    加载 PyTorch tensor，对时间维（dim=1）做一阶差分并补0，再在特征维（dim=2）拼接原始特征和差分特征。
    最后将拼接结果保存到 output_path。
    """
    tensor = torch.load(input_path)  # shape: (N, T, D)
    print(f"Loaded tensor shape: {tensor.shape}")

    # 差分
    diff_tensor = torch.diff(tensor, dim=1)  # shape: (N, T-1, D)

    # 在时间维前面补零，恢复 shape: (N, T, D)
    pad = torch.zeros((tensor.shape[0], 1, tensor.shape[2]),
                      dtype=diff_tensor.dtype, device=diff_tensor.device)
    diff_tensor_padded = torch.cat([pad, diff_tensor], dim=1)

    # 在特征维拼接原始和差分特征 → shape: (N, T, 2D)
    concat_tensor = torch.cat([tensor, diff_tensor_padded], dim=2)

    print(f"Concatenated tensor shape: {concat_tensor.shape}")
    torch.save(concat_tensor, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset to extract MFCC windows and labels.")
    parser.add_argument("wav_dir", type=str, help="Directory containing .wav, .npy, and .txt files")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("output_file_name1", type=str)
    parser.add_argument("output_file_name2", type=str)
    parser.add_argument("--window_size", type=int, default=30, help="Window size for MFCC slicing")
    parser.add_argument("--window_step", type=int, default=1, help="Step size for MFCC slicing")

    args = parser.parse_args()

    print(f"Processing directory: {args.wav_dir}")
    print(f"Window size: {args.window_size}, Step: {args.window_step}")

    inputs, labels = process_dataset(args.wav_dir, args.window_size, args.window_step)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    torch.save(inputs, output_dir / args.output_file_name1)
    torch.save(labels, output_dir / args.output_file_name2)

    print(f"Saved inputs to {output_dir / args.output_file_name1}")
    print(f"Saved labels to {output_dir / args.output_file_name2}")