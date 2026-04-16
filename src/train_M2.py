import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VocalSeparationDataset
from model_v2 import SpectrogramUNetv2
from tqdm import tqdm

def train(data_dir, noise_dir=None, augment=False, epochs=50, batch_size=16, lr=1e-3, save_name="unet_m2", num_noises=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- M2 Version Training (Attention + Multi-Noise) ---")
    print(f"Device: {device} | Noises per sample: {num_noises}")
    
    # 初始化 M2 模型
    model = SpectrogramUNetv2().to(device)
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 模型路徑邏輯 (存入 models/M2)
    os.makedirs("models/M2", exist_ok=True)
    save_path = f"models/M2/{save_name}.pth"
    
    if os.path.exists(save_path):
        print(f">>> Detected existing M2 weights: {save_name}.pth")
        choice = input(">>> [R]esume or [O]verwrite? [R/O]: ").strip().lower()
        if choice == 'r':
            model.load_state_dict(torch.load(save_path, map_location=device))
    
    # [M2 Flagship 參數] 使用 16000Hz 與 256 Hop (高精度譜圖)
    dataset = VocalSeparationDataset(data_dir, noise_dir=noise_dir, sr=16000, hop_length=256, augment=augment, num_noises=num_noises)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"Total samples: {len(dataset)}")
    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            curr_lr = optimizer.param_groups[0]['lr']
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"M2 Epoch {epoch+1}/{epochs}")
            for _, (mix, vocal) in pbar:
                mix, vocal = mix.to(device), vocal.to(device)
                optimizer.zero_grad()
                pred_mask = model(mix)
                pred_vocal = mix * pred_mask
                loss = criterion(pred_vocal, vocal)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{curr_lr:.6f}"})
                
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            torch.save(model.state_dict(), save_path)
            print(f"M2 Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {curr_lr:.6f}")

    except KeyboardInterrupt:
        print("\n>>> Interrupted. Saving...")
        torch.save(model.state_dict(), save_path)

    print(f"\nM2 Training finished. Weights: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--noise_dir", type=str, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_name", type=str, default="unet_m2")
    parser.add_argument("--num_noises", type=int, default=1)
    args = parser.parse_args()
    
    train(args.data_dir, args.noise_dir, args.augment, args.epochs, args.batch_size, 1e-3, args.save_name, args.num_noises)
