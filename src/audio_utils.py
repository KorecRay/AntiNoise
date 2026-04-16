import librosa
import numpy as np

# 專案統一參數
N_FFT = 1024

def process_audio(file_path, sr=22050):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return y

def get_spectrogram(y, n_fft=1024, hop_length=512):
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)
    return magnitude, phase

def reconstruct_audio(magnitude, phase, n_fft=1024, hop_length=512):
    stft_matrix = magnitude * phase
    y_recon = librosa.istft(stft_matrix, n_fft=n_fft, hop_length=hop_length)
    return y_recon

def mix_vocal_noise(vocal, noise, snr_db):
    if len(noise) < len(vocal):
        noise = np.pad(noise, (0, len(vocal) - len(noise)), mode='wrap')
    else:
        noise = noise[:len(vocal)]
    v_rms = np.sqrt(np.mean(vocal**2) + 1e-10)
    n_rms = np.sqrt(np.mean(noise**2) + 1e-10)
    target_n_rms = v_rms / (10**(snr_db / 20))
    scale = target_n_rms / n_rms
    mixed = vocal + noise * scale
    if np.max(np.abs(mixed)) > 1.0:
        mixed = mixed / np.max(np.abs(mixed))
    return mixed

def apply_spec_augment(magnitude, num_mask=1, freq_mask_param=20, time_mask_param=20):
    cloned = magnitude.copy()
    num_freqs, num_frames = cloned.shape
    for _ in range(num_mask):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, num_freqs - f)
        cloned[f0:f0+f, :] = 0
    for _ in range(num_mask):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, num_frames - t)
        cloned[:, t0:t0+t] = 0
    return cloned
