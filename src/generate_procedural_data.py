import os
import numpy as np
import soundfile as sf
from tqdm import tqdm

def generate(output_dir, num_samples=1000, duration=3, sr=22050):
    mix_dir = os.path.join(output_dir, "mix")
    vocal_dir = os.path.join(output_dir, "vocals")
    os.makedirs(mix_dir, exist_ok=True)
    os.makedirs(vocal_dir, exist_ok=True)
    
    print(f"Generating {num_samples} simulated data pairs skipping TorchAudio dependencies...")
    t = np.linspace(0, duration, sr * duration)
    for i in tqdm(range(num_samples)):
        f1, f2 = np.random.uniform(200, 800, 2)
        vocals = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)
        f3 = np.random.uniform(50, 150)
        noise = 0.4 * np.random.randn(len(t)) + 0.5 * np.sin(2 * np.pi * f3 * t)
        mix = vocals + noise
        
        vocals = vocals / np.max(np.abs(vocals)) if np.max(np.abs(vocals)) > 0 else vocals
        mix = mix / np.max(np.abs(mix)) if np.max(np.abs(mix)) > 0 else mix
        
        sf.write(os.path.join(vocal_dir, f"sample_{i:04d}.wav"), vocals, sr)
        sf.write(os.path.join(mix_dir, f"sample_{i:04d}.wav"), mix, sr)
    print("Dummy Generation complete!")

if __name__ == "__main__":
    generate("../data", 1000)
