import os
import shutil

def copy_files_to_subfolders(target_dir: str, txt_path: str):
    """
    根据 txt 文件中列出的路径和目标子目录，将文件复制到目标目录中的对应子目录下。

    Args:
        target_dir (str): 目标根目录路径。
        txt_path (str): 包含源文件路径和子目录名的 txt 文件路径。
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep=",")
            if len(parts) < 2:
                print(f"跳过格式错误的行: {line.strip()}")
                continue
            src_path, subfolder = parts[0], parts[1]
            if not os.path.exists(src_path):
                print(f"源文件不存在: {src_path}")
                continue

            dest_dir = os.path.join(target_dir, subfolder)
            os.makedirs(dest_dir, exist_ok=True)

            dest_path = os.path.join(dest_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)
            print(f"已复制: {src_path} -> {dest_path}")