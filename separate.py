import os
import torch
import soundfile as sf
import argparse
from src.audio_utils import process_audio, get_spectrogram, reconstruct_audio
from src.model import SpectrogramUNet

def separate(input_file, output_file, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    
    # Initialize and Load Model
    model = SpectrogramUNet()
    try:
        model = model.to(device)
        # Test hardware compatibility
        if device.type == 'cuda':
            dummy = torch.randn(1, 1, 64, 64).to(device)
            _ = model(dummy)
    except Exception:
        print("⚠️ GPU 不支援，切換至 CPU 模式運算...")
        device = torch.device('cpu')
        model = model.to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: 找不到模型權重 {model_path}。請先執行訓練 (train.py)。")
        return
        
    model.eval()
    
    print(f"Processing audio: {input_file}")
    y = process_audio(input_file)
    mag, phase = get_spectrogram(y)
    
    # Format shape to (Batch, Channel, Height, Width) for the model format
    mag_tensor = torch.FloatTensor(mag).unsqueeze(0).unsqueeze(0).to(device)
    
    print("Separating vocals...")
    with torch.no_grad():
        mask = model(mag_tensor)
        
    # Apply soft mask
    mask = mask.squeeze().cpu().numpy()
    pred_mag = mag * mask # Extract vocals by scaling down non-vocal frequencies
    
    print("Reconstructing audio time-domain signal...")
    y_recon = reconstruct_audio(pred_mag, phase)
    
    sf.write(output_file, y_recon, 22050)
    print(f"Success! Separated vocal saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate vocals using out U-Net AI.")
    parser.add_argument("input_file", type=str, help="Path to the mixed audio file")
    parser.add_argument("--output_file", type=str, default="output_vocal.wav", help="Output path (eg: vocal.wav)")
    parser.add_argument("--model", type=str, default="models/unet_vocal_separator.pth", help="Path to model weights")
    
    args = parser.parse_args()
    separate(args.input_file, args.output_file, args.model)
