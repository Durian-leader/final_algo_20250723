import os
import torch
import argparse
from tqdm import tqdm
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

def process_wav_file(wav_path, output_txt_path, model, sampling_rate=8000, threshold=0.5):
    """
    处理单个WAV文件并将语音段时间戳保存到TXT文件
    
    Args:
        wav_path: WAV文件路径
        output_txt_path: 输出TXT文件路径
        model: Silero VAD模型
        sampling_rate: 采样率 (默认8000Hz)
        threshold: 语音检测阈值 (默认0.5)
    """
    try:
        # 读取音频文件
        wav = read_audio(wav_path, sampling_rate=sampling_rate)
        
        # 获取语音时间戳
        speech_timestamps = get_speech_timestamps(
            wav, 
            model, 
            sampling_rate=sampling_rate,
            threshold=threshold,
            return_seconds=False
        )
        
        with open(output_txt_path, 'w') as f:
            for ts in speech_timestamps:
                f.write(f"{ts['start']},{ts['end']}\n") 
                    
        return True
    except Exception as e:
        print(f"处理文件 {wav_path} 时出错: {e}")
        return False

def process_directory(input_dir, output_dir=None, sampling_rate=8000, threshold=0.5):
    """
    处理目录中的所有WAV文件
    
    Args:
        input_dir: 输入WAV文件目录
        output_dir: 输出TXT文件目录 (默认与input_dir相同)
        sampling_rate: 采样率 (默认8000Hz)
        threshold: 语音检测阈值 (默认0.5)
    """
    if output_dir is None:
        output_dir = input_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载Silero VAD模型
    print("加载Silero VAD模型...")
    model = load_silero_vad(onnx=False)
    
    # 获取所有WAV文件
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    
    if not wav_files:
        print(f"目录 {input_dir} 中未找到WAV文件")
        return
    
    # 处理每个WAV文件
    print(f"开始处理 {len(wav_files)} 个WAV文件...")
    success_count = 0
    
    for wav_file in tqdm(wav_files):
        wav_path = os.path.join(input_dir, wav_file)
        txt_file = os.path.splitext(wav_file)[0] + '_hard' + '.txt'
        txt_path = os.path.join(output_dir, txt_file)
        
        if process_wav_file(wav_path, txt_path, model, sampling_rate, threshold):
            success_count += 1
    
    print(f"处理完成。成功处理: {success_count}/{len(wav_files)} 文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用Silero VAD标注WAV文件中的语音段')

    parser.add_argument('--input_dir', type=str, default='/home/lidonghaowsl/develop/VeriSilicon_Cup_Competition/calib_wav', help='输入WAV文件目录 (默认 data/wav)')
    parser.add_argument('--output_dir', type=str, default=None, help='输出TXT文件目录 (默认与input_dir相同)')
    parser.add_argument('--sampling_rate', type=int, default=8000, help='WAV文件采样率 (默认 8000Hz)')
    parser.add_argument('--threshold', type=float, default=0.5, help='语音检测阈值 (默认 0.5)')

    args = parser.parse_args()

    # 如果未提供 output_dir，就使用 input_dir
    output_dir = args.output_dir or args.input_dir

    process_directory(
        args.input_dir,
        output_dir,
        args.sampling_rate,
        args.threshold
    )