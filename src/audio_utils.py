import librosa
import numpy as np

N_FFT = 1024
HOP_LENGTH = 512

def process_audio(file_path, sr=22050):
    """Loads audio file and returns it as a mono-channel numpy array."""
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return y

def get_spectrogram(y):
    """
    Computes STFT of audio signal.
    Returns:
        magnitude: Absolute magnitude of the spectrogram
        phase: Phase of the spectrogram (used for reconstruction)
    """
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude, phase = librosa.magphase(stft)
    return magnitude, phase

def reconstruct_audio(magnitude, phase):
    """結合振幅譜與相位譜，透過 ISTFT 反轉換回時域波形。"""
    stft_matrix = magnitude * phase
    y_recon = librosa.istft(stft_matrix, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return y_recon

def apply_spec_augment(magnitude, num_mask=1, freq_mask_param=20, time_mask_param=20):
    """
    實作 SpecAugment：隨機遮蔽部分頻率與時間軸。
    強迫模型根據上下文推測遺失的音訊特徵，提升泛化能力。
    """
    cloned = magnitude.copy()
    num_freqs, num_frames = cloned.shape

    # 頻率遮蔽 (Frequency Masking)
    for _ in range(num_mask):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, num_freqs - f)
        cloned[f0:f0+f, :] = 0

    # 時間遮蔽 (Time Masking)
    for _ in range(num_mask):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, num_frames - t)
        cloned[:, t0:t0+t] = 0
        
    return cloned
