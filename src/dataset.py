import os
import random
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset
from audio_utils import process_audio, mix_vocal_noise, get_spectrogram, apply_spec_augment

class VocalSeparationDataset(Dataset):
    def __init__(self, data_dir, noise_dir=None, sr=22050, duration=5.0, n_fft=1024, hop_length=512, num_noises=1, augment=False):
        self.vocal_dir = os.path.join(data_dir, 'vocals')
        self.noise_dir = noise_dir
        self.sr = sr
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_noises = num_noises
        self.augment = augment
        
        if not os.path.exists(self.vocal_dir):
            raise FileNotFoundError(f"Missing folder: {self.vocal_dir}")
        self.files = [f for f in os.listdir(self.vocal_dir) if f.endswith('.wav')]
        
        self.noise_files = []
        if self.noise_dir and os.path.exists(self.noise_dir):
            self.noise_files = [os.path.join(self.noise_dir, f) for f in os.listdir(self.noise_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.files)

    def _pad_or_trim(self, audio):
        target_len = int(self.sr * self.duration)
        if len(audio) < target_len:
            return np.pad(audio, (0, target_len - len(audio)))
        return audio[:target_len]

    def __getitem__(self, idx):
        vocal_path = os.path.join(self.vocal_dir, self.files[idx])
        vocal_audio = self._pad_or_trim(process_audio(vocal_path, self.sr))
        
        if self.noise_files:
            final_noise = np.zeros_like(vocal_audio)
            for _ in range(self.num_noises):
                n_path = random.choice(self.noise_files)
                current_noise = self._pad_or_trim(process_audio(n_path, self.sr))
                final_noise += current_noise
            snr = random.uniform(5, 15)
            mix_audio = mix_vocal_noise(vocal_audio, final_noise, snr)
        else:
            mix_audio = vocal_audio
            
        v_mag, _ = get_spectrogram(vocal_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        m_mag, _ = get_spectrogram(mix_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # [最終尺寸修正] 極重要！
        # 強制裁切為 8 的倍數，確保 3 層 MaxPool 都不會產生奇數
        # 高度: 513 -> 512
        # 寬度: 可能出現 313, 157 等奇數 -> 強制裁切到最接近的 8 的倍數
        h, w = v_mag.shape
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        
        v_mag = v_mag[:new_h, :new_w]
        m_mag = m_mag[:new_h, :new_w]
        
        if self.augment:
            m_mag = apply_spec_augment(m_mag)
            
        v_mag = np.log1p(v_mag)[np.newaxis, ...]
        m_mag = np.log1p(m_mag)[np.newaxis, ...]
        
        return torch.from_numpy(m_mag).float(), torch.from_numpy(v_mag).float()
