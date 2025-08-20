import matplotlib.pyplot as plt
import numpy as np

def plot_mfcc(mfcc_np: np.ndarray, title="MFCC", save_path=None):
    """
    可视化 MFCC 特征

    Args:
        mfcc_np (np.ndarray): MFCC 特征数组，形状通常是 [n_mfcc, 时间帧数]
        title (str): 图的标题
        save_path (str or Path): 如果指定，则保存图像；否则显示图像
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc_np, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel("Time Frame")
    plt.ylabel("MFCC Coefficient")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()