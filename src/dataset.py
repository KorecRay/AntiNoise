import os
import torch
import numpy as np
from torch.utils.data import Dataset
from audio_utils import process_audio, get_spectrogram

class VocalSeparationDataset(Dataset):
    def __init__(self, data_dir, segment_duration=3, sr=22050):
        """
        Args:
            data_dir (str): Directory containing 'mix' and 'vocals' folders.
            segment_duration (int): Duration in seconds to trim/pad audio.
            sr (int): Sample rate.
        """
        self.data_dir = data_dir
        self.sr = sr
        self.segment_samples = segment_duration * sr
        
        self.mix_dir = os.path.join(data_dir, "mix")
        self.vocal_dir = os.path.join(data_dir, "vocals")
        
        # Load all audio files that have a match in both directories
        self.files = [f for f in os.listdir(self.mix_dir) if f.endswith('.wav')]
        
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
        mix_path = os.path.join(self.mix_dir, filename)
        vocal_path = os.path.join(self.vocal_dir, filename)
        
        # Load and pad/trim time domain audio
        mix_audio = self._pad_or_trim(process_audio(mix_path, self.sr))
        vocal_audio = self._pad_or_trim(process_audio(vocal_path, self.sr))
        
        # Convert to Spectrogram Multi-Dimensional Tensors
        mix_mag, _ = get_spectrogram(mix_audio)
        vocal_mag, _ = get_spectrogram(vocal_audio)
        
        # Convert to PyTorch Tensors (Add Channel Dimension needed for CNNs)
        mix_tensor = torch.FloatTensor(mix_mag).unsqueeze(0)
        vocal_tensor = torch.FloatTensor(vocal_mag).unsqueeze(0)
        
        return mix_tensor, vocal_tensor
