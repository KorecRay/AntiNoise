import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VocalSeparationDataset
from model import SpectrogramUNet
from tqdm import tqdm

def train(data_dir, noise_dir=None, augment=False, epochs=50, batch_size=16, lr=1e-3, save_name="unet_vocal_separator"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initial device check: {device}")
    
    model = SpectrogramUNet()
    try:
        model = model.to(device)
        if device.type == 'cuda':
            dummy = torch.randn(1, 1, 64, 64).to(device)
            _ = model(dummy)
    except Exception as e:
        print(f"⚠️ GPU 暫不支援，已自動退回到 CPU。錯誤: {e}")
        device = torch.device('cpu')
        model = model.to(device)

    print(f"Training using device: {device}")

    criterion = nn.L1Loss()
    # 加入 weight_decay 減少過擬合
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # [核心升級] 學習率排程：若 5 輪 Loss 沒降，則將學習率除以 2
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 模型路徑邏輯 (存入 models/M1)
    os.makedirs("models/M1", exist_ok=True)
    save_path = f"models/M1/{save_name}.pth"
    
    if os.path.exists(save_path):
        print(f"--- ⚠️ 偵測到既有模型權重：{save_name}.pth ---")
        choice = input(">>> 想要 [R]繼續訓練 (Resume) 還是 [O]重新訓練並覆蓋 (Overwrite)? [R/O]: ").strip().lower()
        if choice == 'r':
            try:
                model.load_state_dict(torch.load(save_path, map_location=device))
                print(">>> 已載入舊權重")
            except Exception as e:
                print(f">>> 載入失敗，將重新開始。原因: {e}")
        else:
            print(">>> 已選擇重新訓練，將覆蓋舊檔案。")
    
    # Load Dataset
    print(f"Loading dataset from: {data_dir}")
    # [M1 Legacy 參數] 使用原本的 22050Hz 與 512 Hop
    dataset = VocalSeparationDataset(data_dir, noise_dir=noise_dir, sr=22050, hop_length=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"Found {len(dataset)} training samples. Starting epochs...")
    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            # 顯示當前學習率以便觀察
            current_lr = optimizer.param_groups[0]['lr']
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (mix, vocal) in pbar:
                mix, vocal = mix.to(device), vocal.to(device)
                
                optimizer.zero_grad()
                
                predicted_mask = model(mix)
                pred_vocal = mix * predicted_mask
                
                loss = criterion(pred_vocal, vocal)
                loss.backward()
                
                # 梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})
                
            avg_loss = epoch_loss / len(dataloader)
            
            # 更新排程器
            scheduler.step(avg_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # [每輪保存] 每跑完一輪就存一次，這樣意外中斷也不怕
            torch.save(model.state_dict(), save_path)
            
            # [提早結束提示] 如果 LR 已經微小到極限，提醒使用者
            if current_lr < 1e-7:
                print(">>> [提示] 學習率已達微小極限，模型已收斂，您可以按 Ctrl+C 結束訓練。")

    except KeyboardInterrupt:
        print("\n\n>>> ⚠️ 偵測到手動中斷訓練！")
        choice = input(">>> 是否要保存目前的訓練成果？ [Y/N]: ").strip().lower()
        if choice == 'y':
            torch.save(model.state_dict(), save_path)
            print(f">>> 成果已安全存至 {save_path}，下次可繼續疊加功力。")
        else:
            print(">>> 已放棄本次變更。")

    print(f"Training session ended. Final weights: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train U-Net Vocal Separator")
    parser.add_argument("--data_dir", type=str, default="../data", help="Path to data directory")
    parser.add_argument("--noise_dir", type=str, default=None, help="Optional noise directory")
    parser.add_argument("--augment", action="store_true", help="Enable SpecAugment")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    parser.add_argument("--save_name", type=str, default="unet_vocal_separator", help="Name of the model file")
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        noise_dir=args.noise_dir,
        augment=args.augment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_name=args.save_name
    )

