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
    """
    Takes magnitude and phase spectrograms, combines them, 
    and returns to time-domain audio.
    """
    stft_matrix = magnitude * phase
    y_recon = librosa.istft(stft_matrix, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return y_recon
