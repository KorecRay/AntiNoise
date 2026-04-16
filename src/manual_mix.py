import os
import numpy as np
import soundfile as sf
import librosa

def manual_mix(vocal_path, noise_path, output_path, snr_db=5):
    print(f"Mixing: \n Vocal: {os.path.basename(vocal_path)} \n Noise: {os.path.basename(noise_path)}")
    
    # Load audio
    vocal, sr = librosa.load(vocal_path, sr=16000, mono=True)
    noise, _ = librosa.load(noise_path, sr=16000, mono=True)
    
    # Pad or trim noise to match vocal length
    if len(noise) < len(vocal):
        noise = np.pad(noise, (0, len(vocal) - len(noise)), mode='wrap')
    else:
        noise = noise[:len(vocal)]
        
    # Calculate energies
    vocal_energy = np.sum(vocal**2) / len(vocal)
    noise_energy = np.sum(noise**2) / len(noise)
    
    # Calculate required noise power for target SNR
    # SNR = 10 * log10(P_signal / P_noise)
    # P_noise = P_signal / (10^(SNR/10))
    target_noise_energy = vocal_energy / (10**(snr_db / 10))
    
    # Scale noise
    scaling_factor = np.sqrt(target_noise_energy / (noise_energy + 1e-10))
    mixed = vocal + noise * scaling_factor
    
    # Normalize to avoid clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
        
    sf.write(output_path, mixed, sr)
    print(f"\n[✔] Mix complete! Saved to: {output_path}")
    print(f"    SNR: {snr_db}dB (Lower dB means LOUDER noise)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocal", type=str, required=True)
    parser.add_argument("--noise", type=str, required=True)
    parser.add_argument("--snr", type=float, default=5)
    parser.add_argument("--out", type=str, default="test_mix.wav")
    args = parser.parse_args()
    
    manual_mix(args.vocal, args.noise, args.out, args.snr)
