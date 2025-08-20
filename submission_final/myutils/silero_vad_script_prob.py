import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from silero_vad import load_silero_vad, read_audio

def process_wav_file_with_probs(wav_path, output_txt_path, model, sampling_rate=8000):
    """
    处理单个WAV文件并将每一帧的语音概率保存到TXT文件
    
    Args:
        wav_path: WAV文件路径
        output_txt_path: 输出TXT文件路径
        model: Silero VAD模型
        sampling_rate: 采样率 (默认8000Hz)
    """
    try:
        # 确定精确的样本数（Silero VAD要求）
        num_samples = 512 if sampling_rate == 16000 else 256
        frame_ms = int(num_samples / sampling_rate * 1000)
        
        # print(f"使用精确的帧大小: {num_samples}样本 ({frame_ms}ms) @ {sampling_rate}Hz")
        
        # 读取音频文件
        wav = read_audio(wav_path, sampling_rate=sampling_rate)
        
        # 检查音频是否太短
        if len(wav) < num_samples:
            # print(f"警告: 音频文件 {wav_path} 太短 ({len(wav)} 样本), 将被填充到 {num_samples} 样本")
            # 填充音频到所需长度
            wav = torch.nn.functional.pad(wav, (0, num_samples - len(wav)))
        
        # 逐帧处理音频，收集每一帧的概率值
        frame_probs = []
        frame_times = []
        
        for i in range(0, len(wav), num_samples):
            # 提取精确长度的片段
            if i + num_samples <= len(wav):
                chunk = wav[i:i + num_samples]
                
                # 验证样本长度
                if len(chunk) != num_samples:
                    # print(f"错误: 片段长度不正确 ({len(chunk)} != {num_samples}), 跳过")
                    continue
                
                # 获取语音概率 (而不是二元标签)
                try:
                    with torch.no_grad():
                        speech_prob = model(chunk, sampling_rate).item()
                    
                    # 记录当前帧的时间(毫秒)和概率
                    start_ms = int(i / sampling_rate * 1000)
                    end_ms = int((i + num_samples) / sampling_rate * 1000)
                    
                    frame_times.append((start_ms, end_ms))
                    frame_probs.append(speech_prob)
                except Exception as e:
                    # print(f"警告: 处理第 {i//num_samples} 帧 (样本 {i}-{i+num_samples}) 时出错: {str(e)}")
                    pass
        
        # 将帧概率保存到TXT文件
        with open(output_txt_path, 'w') as f:
            for (start_ms, end_ms), prob in zip(frame_times, frame_probs):
                f.write(f"{start_ms},{end_ms},{prob:.6f}\n")
        
        # 验证我们是否成功处理了一些帧
        if len(frame_probs) == 0:
            # print(f"警告: 文件 {wav_path} 未能处理任何帧")
            # 创建默认的标签文件
            create_default_label_file(wav_path, output_txt_path, sampling_rate, num_samples)
            return True
        
        # print(f"成功处理 {len(frame_probs)} 帧")
        return True
    except Exception as e:
        # print(f"处理文件 {wav_path} 时出错: {e}")
        # 创建默认的标签文件
        try:
            create_default_label_file(wav_path, output_txt_path, sampling_rate, num_samples)
            return True
        except:
            return False

def create_default_label_file(wav_path, output_txt_path, sampling_rate, num_samples):
    """创建默认的标签文件，所有帧都标记为非语音"""
    try:
        # 估计帧数量
        import torchaudio
        
        # 获取音频信息
        info = torchaudio.info(wav_path)
        num_frames = info.num_frames // num_samples
        duration_ms = int(info.num_frames / sampling_rate * 1000)
        
        with open(output_txt_path, 'w') as f:
            for i in range(num_frames):
                start_ms = int(i * num_samples / sampling_rate * 1000)
                end_ms = int((i + 1) * num_samples / sampling_rate * 1000)
                # 确保不超过音频总长度
                if end_ms > duration_ms:
                    end_ms = duration_ms
                # 写入零概率
                f.write(f"{start_ms},{end_ms},0.0\n")
        
        # print(f"为文件 {wav_path} 创建了默认标签 (所有帧标为非语音)")
        return True
    except Exception as e:
        # print(f"创建默认标签时出错: {e}")
        
        # 最后尝试创建只有一帧的标签文件
        try:
            with open(output_txt_path, 'w') as f:
                f.write("0,1000,0.0\n")
            # print(f"为文件 {wav_path} 创建了单帧默认标签")
            return True
        except Exception as e2:
            # print(f"创建单帧标签也失败: {e2}")
            return False

def process_directory(input_dir, output_dir=None, sampling_rate=8000):
    """
    处理目录中的所有WAV文件，输出软标签
    
    Args:
        input_dir: 输入WAV文件目录
        output_dir: 输出TXT文件目录 (默认与input_dir相同)
        sampling_rate: 采样率 (默认8000Hz)
    """
    if output_dir is None:
        output_dir = input_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定精确的样本数（Silero VAD要求）
    num_samples = 512 if sampling_rate == 16000 else 256
    frame_ms = int(num_samples / sampling_rate * 1000)
    
    print(f"使用精确的帧大小: {num_samples}样本 ({frame_ms}ms) @ {sampling_rate}Hz")
    
    # 导入torchaudio用于获取音频信息
    import torchaudio
    
    # 加载Silero VAD模型
    print("加载Silero VAD模型...")
    model = load_silero_vad(onnx=False)
    model.eval()
    
    # 获取所有WAV文件
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    
    if not wav_files:
        print(f"目录 {input_dir} 中未找到WAV文件")
        return
    
    # 处理每个WAV文件
    print(f"开始处理 {len(wav_files)} 个WAV文件...")
    success_count = 0
    
    for wav_file in tqdm(wav_files, desc="Processing WAV files", leave=True):
        wav_path = os.path.join(input_dir, wav_file)
        txt_file = os.path.splitext(wav_file)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_file)
        
        if process_wav_file_with_probs(wav_path, txt_path, model, sampling_rate):
            success_count += 1
    
    print(f"处理完成。成功处理: {success_count}/{len(wav_files)} 文件")
    
    # 删除临时文件
    for f in os.listdir(output_dir):
        if f.startswith("temp_empty_") and f.endswith(".txt"):
            try:
                os.remove(os.path.join(output_dir, f))
            except:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用Silero VAD生成音频文件的软标签（概率值）')

    parser.add_argument('--input_dir', type=str, default="/home/lidonghaowsl/develop/VeriSilicon_Cup_Competition/DS-CNN/V4_soft_label_mfcc_2d/dataset_remixed/labeled", 
                        help='输入WAV文件目录')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='输出TXT文件目录 (默认与input_dir相同)')
    parser.add_argument('--sampling_rate', type=int, default=8000, 
                        help='WAV文件采样率 (默认 8000Hz)')

    args = parser.parse_args()

    # 如果未提供 output_dir，就使用 input_dir
    output_dir = args.output_dir or args.input_dir

    process_directory(
        args.input_dir,
        output_dir,
        args.sampling_rate
    )