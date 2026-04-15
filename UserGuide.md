# AI 音訊聲源分離系統：使用指南 (User Guide)

本專案提供一套基於 U-Net 與 SpecAugment 技術的聲源分離解決方案。支援高效能 NVIDIA GPU 加速，特別針對 **Blackwell 架構 (RTX 50 系列)** 進行優化。

---

## 快速環境配置

### 1. 建立虛擬環境 (建議使用 Python 3.12)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. 安裝核心依賴 (針對 RTX 5080 / Blackwell GPU)
由於 RTX 50 系列採用 sm_120 架構，請務必安裝以下 Nightly 版本以獲得硬體加速支援 (已實機測試通過)：
```powershell
# 安裝支援 Blackwell 之 PyTorch Nightly (cu128)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# 安裝音訊處理與其餘套件
pip install librosa soundfile tqdm requests
```

---

## 資料準備

本系統支援「動態混音」模式。建議將資料放置於 `data/` 目錄：
- `data/vocals/`: 存放純淨的人聲 WAV 檔案 (推薦使用 LibriSpeech)。
- `data/noises/`: (選用) 存放真實環境噪音 WAV 檔案 (咖啡廳、風聲、街道音等)。

---

## 訓練模型

### 基本訓練 (靜態混音)
```powershell
python src/train.py --data_dir ../data --epochs 50
```

### 進階訓練 (動態混音 + 隨機 SNR + SpecAugment)
**推薦模式**：使用此模式可獲得最強的泛化能力，避免模型「走捷徑」。
```powershell
# --noise_dir: 指定真實噪音來源
# --augment: 開啟頻譜遮蔽增強 (SpecAugment)
python src/train.py --data_dir ../data --noise_dir ../data/noises --augment --epochs 100
```

---

## 執行聲源分離 (推論)

輸入一個包含背景音的 WAV 檔案，輸出分離後的人聲：
```powershell
python separate.py input_file.wav --output clean_voice.wav --model models/unet_vocal_separator.pth
```

---

## 硬體相容性檢查

本專案內建 Blackwell (sm_120) 自動相容機制。

若要手動確認 GPU 狀態，可點擊執行：
```powershell
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---
