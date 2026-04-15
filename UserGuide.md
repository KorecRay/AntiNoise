# 🎤 Vocal Separation AI 專案使用指南

本專案是一個基於 **STFT U-Net** 架構的人聲分離 AI，專門設計於本機端執行（支援 NVIDIA RTX 50 系列顯卡 GPU 加速）。

---

## 階段一：環境配置

發揮 **RTX 50架構** 算力，請使用 Python 3.12 與虛擬環境 (不穩定)：

1. **建立虛擬環境**：
   ```powershell
   python -m venv venv
   ```
2. **安裝 GPU 版 PyTorch (Nightly 版本以支援最新架構)**：
   ```powershell
   .\venv\Scripts\python.exe -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
   ```
3. **安裝其餘相依套件**：
   ```powershell
   .\venv\Scripts\python.exe -m pip install librosa numpy scipy soundfile tqdm matplotlib
   ```
*** 若不支援 sm120 會自動退回至 CPU 運行***

---

## 階段二：資料準備

在訓練之前，我們需要 1,000 筆訓練樣本。我們使用自動合成腳本來產生對照組：

*   **執行指令**：
    ```powershell
    cd src
    ..\venv\Scripts\python.exe generate_procedural_data.py
    ```
*   **說明**：此腳本會在 `data/mix` 與 `data/vocals` 資料夾中產生 1,000 個配對的音訊檔。

---

## 階段三：模型訓練

這是專案的核心，AI 會學習如何從混音中識別並提取人聲特徵：

*   **執行指令**：
    ```powershell
    cd src
    ..\venv\Scripts\python.exe train.py
    ```
*   **重點**：
    1.  確認出現 `Training using device: cuda`（代表正在使用 GPU）。
    2.  觀察 `tqdm` 進度條每一輪 (Epoch) 的進度與 `Loss` (誤差值) 的變化。
*   **輸出位置**：訓練結束後，模型權重會存至 `models/unet_vocal_separator.pth`。

---

## 階段四：實際使用

當模型訓練完成後，您可以拿任何 WAV 檔案來進行實際的人聲分離：

*   **執行指令**：
    ```powershell
    # 請回到專案根目錄執行
    cd d:\Coding\EMER\AI
    .\venv\Scripts\python.exe separate.py [您的音樂路徑.wav] --output_file separated_vocal.wav
    ```
*   **說明**：AI 會讀取模型權重，對您的音樂進行頻譜遮罩 (Soft Masking) 運算，最後導出純淨的人聲檔。

---

## 🛠️ 疑難排解 (Troubleshooting)

*   **CUDA Error: no kernel image...**：
    *   代表您的顯卡太新 (RTX 50 系列)，目前的穩定版 PyTorch 尚未收錄其驅動二進位檔。
    *   **解決方案**：請務必如階段一所示安裝 `Nightly` 版本。
*   **ImportError: attempted relative import...**：
    *   請確保您是在 `src` 目錄內執行腳本，或使用 `python -m src.train` 方式啟動。

---
