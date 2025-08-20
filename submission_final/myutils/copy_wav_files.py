import os
import shutil

def copy_all_wavs(source_dir, target_dir):
    """
    将 source_dir 中所有子目录下的 .wav 文件复制到 target_dir 中（不保留目录结构）
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)

                # 如果有重名文件可考虑加索引或跳过
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(file)
                    i = 1
                    while os.path.exists(os.path.join(target_dir, f"{base}_{i}{ext}")):
                        i += 1
                    target_path = os.path.join(target_dir, f"{base}_{i}{ext}")

                shutil.copy2(source_path, target_path)
                print(f"复制: {source_path} -> {target_path}")

if __name__ == '__main__':
    # 修改以下路径为你的源目录和目标目录
    source_directory = '/home/lidonghaowsl/develop/VeriSilicon_Cup_Competition/DS-CNN/V4_soft_label_mfcc_2d/dataset_remixed/source'
    target_directory = '/home/lidonghaowsl/develop/VeriSilicon_Cup_Competition/DS-CNN/V4_soft_label_mfcc_2d/dataset_remixed/labeled'

    copy_all_wavs(source_directory, target_directory)
