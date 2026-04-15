"""
資料準備腳本：從 LibriSpeech 開放資料集下載真人語音
完全不依賴 torchaudio，改用 requests + librosa + soundfile
"""
import os
import io
import tarfile
import requests
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
DOWNLOAD_PATH = "../data/librispeech.tar.gz"
EXTRACT_PATH = "../data/librispeech_raw"

def download_librispeech():
    if os.path.exists(EXTRACT_PATH):
        print("LibriSpeech 資料已存在，跳過下載。")
        return
    os.makedirs("../data", exist_ok=True)
    print(f"正在從 OpenSLR 下載真人高清語音資料集 (約 300MB)...")
    response = requests.get(LIBRISPEECH_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(DOWNLOAD_PATH, 'wb') as f, tqdm(
        desc="下載進度", total=total_size, unit='iB', unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    print("下載完成！正在解壓縮...")
    with tarfile.open(DOWNLOAD_PATH, 'r:gz') as tar:
        tar.extractall("../data/librispeech_raw")
    os.remove(DOWNLOAD_PATH)
    print("解壓完成！")

def prepare_data(output_dir, num_samples=1000, duration=3, sr=22050):
    mix_dir = os.path.join(output_dir, "mix")
    vocal_dir = os.path.join(output_dir, "vocals")
    os.makedirs(mix_dir, exist_ok=True)
    os.makedirs(vocal_dir, exist_ok=True)

    # 搜集所有 flac 音訊檔
    flac_files = []
    for root, dirs, files in os.walk(EXTRACT_PATH):
        for f in files:
            if f.endswith('.flac'):
                flac_files.append(os.path.join(root, f))

    if not flac_files:
        print("找不到音訊檔，改用程序性合成資料！")
        generate_fallback(output_dir, num_samples, duration, sr)
        return

    print(f"找到 {len(flac_files)} 個真人語音檔，開始切割成 {num_samples} 筆訓練對...")
    samples_needed = int(duration * sr)
    t = np.linspace(0, duration, samples_needed)
    
    for i in tqdm(range(num_samples), desc="生成訓練對"):
        # 隨機選一個語音檔
        src_file = flac_files[i % len(flac_files)]
        
        try:
            audio, file_sr = librosa.load(src_file, sr=sr, mono=True)
        except Exception:
            audio = np.zeros(samples_needed)
        
        # 裁切或補零
        if len(audio) > samples_needed:
            start = np.random.randint(0, len(audio) - samples_needed)
            vocal_crop = audio[start:start + samples_needed]
        else:
            vocal_crop = np.pad(audio, (0, max(0, samples_needed - len(audio))))
        
        # 生成程序性伴奏背景（混合多頻率低頻波 + 輕微噪音）
        f_bg = np.random.uniform(60, 200)
        f_bg2 = np.random.uniform(100, 300)
        background = (0.3 * np.sin(2 * np.pi * f_bg * t) + 
                      0.2 * np.sin(2 * np.pi * f_bg2 * t) + 
                      0.15 * np.random.randn(samples_needed))
        
        # 混音
        mix = vocal_crop + background
        
        # 正規化
        if np.max(np.abs(vocal_crop)) > 0:
            vocal_crop = vocal_crop / (np.max(np.abs(vocal_crop)) + 1e-8)
        if np.max(np.abs(mix)) > 0:
            mix = mix / (np.max(np.abs(mix)) + 1e-8)
        
        sf.write(os.path.join(vocal_dir, f"sample_{i:04d}.wav"), vocal_crop, sr)
        sf.write(os.path.join(mix_dir, f"sample_{i:04d}.wav"), mix, sr)

    print(f"\n成功！1000 筆真人語音訓練資料已保存至 '{output_dir}'")

def generate_fallback(output_dir, num_samples, duration, sr):
    """若下載失敗，退回至程序性合成資料"""
    mix_dir = os.path.join(output_dir, "mix")
    vocal_dir = os.path.join(output_dir, "vocals")
    t = np.linspace(0, duration, sr * duration)
    for i in tqdm(range(num_samples)):
        f1, f2 = np.random.uniform(200, 800, 2)
        vocals = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)
        f3 = np.random.uniform(50, 150)
        noise = 0.4 * np.random.randn(len(t)) + 0.5 * np.sin(2 * np.pi * f3 * t)
        mix = vocals + noise
        vocals = vocals / (np.max(np.abs(vocals)) + 1e-8)
        mix = mix / (np.max(np.abs(mix)) + 1e-8)
        sf.write(os.path.join(vocal_dir, f"sample_{i:04d}.wav"), vocals, sr)
        sf.write(os.path.join(mix_dir, f"sample_{i:04d}.wav"), mix, sr)

if __name__ == "__main__":
    download_librispeech()
    prepare_data("../data", num_samples=1000)
