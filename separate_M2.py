import os
import torch
import soundfile as sf
import argparse
import numpy as np
from src.audio_utils import process_audio, get_spectrogram, reconstruct_audio
from src.model_v2 import SpectrogramUNetv2

def separate_m2(input_file, output_file, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- M2 Separation Engine (Attention) ---")
    
    model = SpectrogramUNetv2().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f">>> M2 Weights Loaded: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"Error: {e}")
        return
        
    model.eval()
    # M2 使用 16000Hz 與 256 Hop
    y = process_audio(input_file, sr=16000)
    mag, phase = get_spectrogram(y, n_fft=1024, hop_length=256)
    
    # [尺寸對齊] 確保長寬都是 8 的倍數
    h, w = mag.shape
    new_h = (h // 8) * 8
    new_w = (w // 8) * 8
    mag = mag[:new_h, :new_w]
    phase = phase[:new_h, :new_w]
    
    mag_tensor = torch.FloatTensor(mag).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mask = model(mag_tensor)
        
    mask = mask.squeeze().cpu().numpy()
    pred_mag = mag * mask
    y_recon = reconstruct_audio(pred_mag, phase, n_fft=1024, hop_length=256)
    
    sf.write(output_file, y_recon, 16000)
    print(f"Success! Result saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    separate_m2(args.input_file, args.output_file, args.model)
