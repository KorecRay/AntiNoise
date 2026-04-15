import os
import torch
import numpy as np
from torch.utils.data import Dataset
from audio_utils import process_audio, get_spectrogram, apply_spec_augment

class VocalSeparationDataset(Dataset):
    def __init__(self, data_dir, noise_dir=None, segment_duration=3, sr=22050, augment=False):
        """
        Args:
            data_dir (str): Directory containing 'vocals' folders.
            noise_dir (str): Optional directory containing real-world noise .wav files.
            segment_duration (int): Duration in seconds to trim/pad audio.
            sr (int): Sample rate.
            augment (bool): Whether to apply SpecAugment.
        """
        self.data_dir = data_dir
        self.noise_dir = noise_dir # 實作建議：改用真實噪音源
        self.sr = sr
        self.segment_samples = segment_duration * sr
        self.augment = augment
        
        self.vocal_dir = os.path.join(data_dir, "vocals")
        self.mix_dir = os.path.join(data_dir, "mix") # 備用：讀取預設合成好的混音
        
        # 讀取人聲檔案清單
        self.files = [f for f in os.listdir(self.vocal_dir) if f.endswith('.wav')]
        
        # 讀取噪音檔案清單 (如果有的話)
        self.noise_files = []
        if self.noise_dir and os.path.exists(self.noise_dir):
            self.noise_files = [f for f in os.listdir(self.noise_dir) if f.endswith(('.wav', '.flac'))]
        
    def __len__(self):
        return len(self.files)
        
    def _pad_or_trim(self, audio):
        """Ensures all audio clips are the exact same length."""
        if len(audio) > self.segment_samples:
            # Random crop
            start = np.random.randint(0, len(audio) - self.segment_samples)
            return audio[start:start+self.segment_samples]
        else:
            # Pad with zeros
            return np.pad(audio, (0, self.segment_samples - len(audio)))
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        vocal_path = os.path.join(self.vocal_dir, filename)
        vocal_audio = self._pad_or_trim(process_audio(vocal_path, self.sr))
        
        # --- 實作動態數據增強管線 (Data Augmentation Pipeline) ---
        if self.noise_files:
            # 隨機挑選一段真實噪音進行混音
            noise_idx = np.random.randint(0, len(self.noise_files))
            noise_path = os.path.join(self.noise_dir, self.noise_files[noise_idx])
            noise_audio = self._pad_or_trim(process_audio(noise_path, self.sr))
            
            # 您建議的：隨機設定信噪比在 -5dB 到 15dB 之間，防止模型「走捷徑」
            snr = np.random.uniform(-5, 15)
            clean_power = np.mean(vocal_audio**2) + 1e-10
            noise_power = np.mean(noise_audio**2) + 1e-10
            scale = np.sqrt(clean_power / (noise_power * (10**(snr/10))))
            mix_audio = vocal_audio + noise_audio * scale
        else:
            # 備用方案：若無噪音目錄，則讀取預先製作好的 mix
            mix_path = os.path.join(self.mix_dir, filename)
            mix_audio = self._pad_or_trim(process_audio(mix_path, self.sr))
        
        # 振幅頻譜圖處理
        mix_mag, _ = get_spectrogram(mix_audio)
        vocal_mag, _ = get_spectrogram(vocal_audio)
        
        # 實作 SpecAugment 遮蔽部分頻寬，增加模型泛化能力
        if self.augment:
            mix_mag = apply_spec_augment(mix_mag)
        
        # 轉換為 PyTorch Tensors
        mix_tensor = torch.FloatTensor(mix_mag).unsqueeze(0)
        vocal_tensor = torch.FloatTensor(vocal_mag).unsqueeze(0)
        
        return mix_tensor, vocal_tensor
