import os
import wave
from collections import defaultdict
import librosa

def calculate_category_durations(dataset_path):
    """
    计算数据集中每个类别（子文件夹）的音频总时长
    
    参数:
        dataset_path: 数据集根目录路径
        
    返回:
        一个字典，键为类别名称，值为该类别的总时长（秒）
    """
    category_durations = defaultdict(float)
    
    # 遍历所有子文件夹
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        
        # 确保是目录
        if not os.path.isdir(category_path):
            continue
            
        # 遍历该类别下的所有wav文件
        for root, _, files in os.walk(category_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        # 方法1：使用wave库
                        with wave.open(file_path, 'r') as wf:
                            # 获取帧数和帧率
                            frames = wf.getnframes()
                            rate = wf.getframerate()
                            duration = frames / float(rate)
                        
                        # 或者使用方法2：librosa库
                        # duration = librosa.get_duration(path=file_path)
                        
                        category_durations[category] += duration
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {e}")
    
    # 打印结果
    print("各类别音频时长统计（秒）:")
    for category, duration in category_durations.items():
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        print(f"{category}: {duration:.2f}秒 ({hours}小时 {minutes}分钟 {seconds}秒)")
        
    return category_durations


import os
import wave
import shutil
import random
import math

def sample_audio_files(dataset_path, category, target_duration, output_folder):
    """
    从指定类别中随机抽取指定总时长的音频文件并复制到目标文件夹
    
    参数:
        dataset_path: 数据集根目录路径
        category: 要抽取的类别名称
        target_duration: 目标总时长（秒）
        output_folder: 输出文件夹路径
    
    返回:
        实际抽取的总时长（秒）
    """
    category_path = os.path.join(dataset_path, category)
    
    # 确保目标文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 收集类别中的所有wav文件及其时长
    audio_files = []
    
    for root, _, files in os.walk(category_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                
                try:
                    with wave.open(file_path, 'r') as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        duration = frames / float(rate)
                    
                    audio_files.append((file_path, duration))
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
    
    # 随机打乱文件列表
    random.shuffle(audio_files)
    
    # 选择文件直到达到目标时长
    selected_files = []
    current_duration = 0
    
    for file_path, duration in audio_files:
        selected_files.append((file_path, duration))
        current_duration += duration
        
        if current_duration >= target_duration:
            break
    
    # 复制所选文件到目标文件夹
    print(f"从类别 '{category}' 中抽取 {len(selected_files)} 个文件:")
    
    for i, (file_path, duration) in enumerate(selected_files):
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_folder, filename)
        
        # 如果文件名重复，添加序号
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            dest_path = os.path.join(output_folder, f"{base}_{i}{ext}")
        
        shutil.copy2(file_path, dest_path)
        print(f"  - {filename}: {duration:.2f}秒")
    
    print(f"总抽取时长: {current_duration:.2f}秒，目标时长: {target_duration:.2f}秒")
    
    return current_duration

# 使用示例
if __name__ == "__main__":
    # 数据集路径
    dataset_path = "/path/to/your/dataset"
    
    # 1. 计算所有类别的总时长
    category_durations = calculate_category_durations(dataset_path)
    
    # 2. 从指定类别中随机抽取音频文件
    category_to_sample = "your_category_name"  # 替换为您要抽取的类别名
    target_duration = 3600  # 目标时长（秒），例如1小时
    output_folder = "/path/to/output/folder"
    
    sampled_duration = sample_audio_files(
        dataset_path, 
        category_to_sample, 
        target_duration, 
        output_folder
    )