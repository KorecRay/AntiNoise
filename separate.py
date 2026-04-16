import os
import torch
import soundfile as sf
import argparse
from src.audio_utils import process_audio, get_spectrogram, reconstruct_audio
from src.model import SpectrogramUNet
from src.model_v2 import SpectrogramUNetv2

def separate(input_file, output_file, model_path):
    # 強制判斷 CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- AI 分離處理器 (GPU:{torch.cuda.is_available()}) ---")
    
    # [模型自動變身邏輯] 根據路徑自動切換架構
    if "M2" in model_path:
        print(">>> 偵測到 M2 架構 (Self-Attention 版本)...")
        model = SpectrogramUNetv2()
    else:
        print(">>> 使用 M1 標準架構...")
        model = SpectrogramUNet()

    try:
        model = model.to(device)
    except Exception:
        device = torch.device('cpu')
        model = model.to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f">>> 成功加載權重: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"Error: 權重加載失敗! {e}")
        return
        
    model.eval()
    
    print(f"Processing audio: {input_file}")
    # 確保以 16000Hz 處理 (這是訓練時採用的頻率)
    y = process_audio(input_file, sr=16000)
    mag, phase = get_spectrogram(y)
    
    # 格式化為 (Batch, Channel, Freq, Time)
    mag_tensor = torch.FloatTensor(mag).unsqueeze(0).unsqueeze(0).to(device)
    
    print("Separating vocals...")
    with torch.no_grad():
        mask = model(mag_tensor)
        
    mask = mask.squeeze().cpu().numpy()
    pred_mag = mag * mask
    
    print("Reconstructing audio signal...")
    y_recon = reconstruct_audio(pred_mag, phase)
    
    sf.write(output_file, y_recon, 16000)
    print(f"Success! Separated vocal saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate vocals using our U-Net AI.")
    parser.add_argument("input_file", type=str, help="Path to the mixed audio file")
    parser.add_argument("--output_file", type=str, default="output_vocal.wav", help="Output path")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    
    args = parser.parse_args()
    separate(args.input_file, args.output_file, args.model)
