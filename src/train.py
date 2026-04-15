import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VocalSeparationDataset
from model import SpectrogramUNet
from tqdm import tqdm

def train(data_dir, epochs=5, batch_size=16, lr=1e-3):
    # 強制檢查 CUDA 兼容性或退回 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initial device check: {device}")
    
    # 建立模型測試 GPU 兼容性
    model = SpectrogramUNet()
    try:
        model = model.to(device)
        # 測試一個小運算來確保顯卡真的能跑 (針對 RTX 50 系列的特殊保護)
        if device.type == 'cuda':
            dummy = torch.randn(1, 1, 64, 64).to(device)
            _ = model(dummy)
    except Exception as e:
        print(f"⚠️ GPU 暫不支援您的顯卡架構 (sm_120)，已自動退回到 CPU 模式以確保運行。\n錯誤訊息: {e}")
        device = torch.device('cpu')
        model = model.to(device)

    print(f"Training using device: {device}")

    criterion = nn.L1Loss() # Mean Absolute Error on spectrograms often yields less artifacts
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Load Dataset
    print(f"Loading dataset from: {data_dir}")
    dataset = VocalSeparationDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"Found {len(dataset)} training samples. Starting epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (mix, vocal) in pbar:
            mix, vocal = mix.to(device), vocal.to(device)
            
            optimizer.zero_grad()
            
            # Predict the mask and apply it to the mixed magnitude
            predicted_mask = model(mix)
            pred_vocal = mix * predicted_mask
            
            # Loss is calculated against the ground truth vocal spectrogram
            loss = criterion(pred_vocal, vocal)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
    # Save Model Weights
    os.makedirs("../models", exist_ok=True)
    save_path = "../models/unet_vocal_separator.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights saved to {save_path}")

if __name__ == "__main__":
    # Change data_dir to match absolute or relative path
    # Assuming this script is run from inside the src/ folder
    # 為了成果報告快速產出，我們預設跑 5 個 Epoch 即可
    train("../data", epochs=5)

