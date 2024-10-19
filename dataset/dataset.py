import matplotlib.pyplot as plt
import os

from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torchaudio
import torch
import torch.nn as nn
from torch.nn.functional import pad

import random
import sys
import os
sys.path.append(os.curdir)

from models.VisResMAE import VisResMAE


class AudioDataset(Dataset):
    def __init__(self, data_dir, target_sr, sample_len, transformation):

        super().__init__()
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.sample_len = sample_len
        self.transformation = transformation
        self.audio_files = os.listdir(self.data_dir)
        
        # Calculate normalization stats
        self.min_val = float('inf')
        self.max_val = float('-inf')
        for idx in range(len(self.audio_files)):
            signal = self._load_and_transform(idx)
            self.min_val = min(self.min_val, signal.min().item())
            self.max_val = max(self.max_val, signal.max().item())
    
    def _load_and_transform(self, idx):
        audio_sample_path = self.get_audio_path(idx)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self.resample(signal, sr)
        signal = self.to_mono(signal)
        signal = self.padding(signal)
        signal = self.truncate(signal)
        signal = self.transformation(signal)
        return signal
    
    def __getitem__(self, idx):
        signal = self._load_and_transform(idx)
        # Normalize to [-1, 1] range
        signal = 2 * (signal - self.min_val) / (self.max_val - self.min_val) - 1
        return signal
    
    def __len__(self):
        return len(self.audio_files)

    def get_audio_path(self, idx):
        filename = self.audio_files[idx]
        return os.path.join(self.data_dir, filename)

    def resample(self, signal, sr):
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            signal = resampler(signal)
        return signal

    def to_mono(self, signal):
        if signal.shape[0] != 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def padding(self, signal):
        len_signal = signal.shape[1]
        if len_signal < self.sample_len:
            n_missing = self.sample_len - len_signal
            padding = (0, n_missing)
            signal = pad(signal, padding)
        return signal

    def truncate(self, signal):
        if signal.shape[1] > self.sample_len:
            signal = signal[:, :self.sample_len]
        return signal
    

if __name__ == "__main__": 

    def mask_spectrogram(spectrogram, max_mask_size=50, deterministic=False, seed=42):
        if deterministic:
            random.seed(seed)
        B, C, freq_bins, time_frames = spectrogram.shape
        max_possible_mask_size = min(max_mask_size, freq_bins, time_frames)
        mask_size = random.randint(1, max_possible_mask_size)
        freq_start = random.randint(0, freq_bins - mask_size)
        time_start = random.randint(0, time_frames - mask_size)
        spectrogram[:, :, freq_start:freq_start+mask_size, time_start:time_start+mask_size] = 0
        return spectrogram

    audio_dir = '/Users/inigoparra/Desktop/speech'

    transforms = Compose([
        torchaudio.transforms.Resample(orig_freq=16000, new_freq=16000),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=256,                 # Small window for better temporal precision
            hop_length=64,             # High overlap to capture temporal structure
            n_mels=64,                 # Mel bands (128 is common for speech)
            f_min=80,                  # Capture low-frequency components like voice
            f_max=8000,                # Respect Nyquist limit at 8 kHz sample rate
            norm='slaney',
            mel_scale='slaney',
            pad_mode='reflect'
        ),
        torchaudio.transforms.AmplitudeToDB()
    ])

    dataset = AudioDataset(
        data_dir=audio_dir,
        target_sr=16000,
        sample_len=64000,
        transformation=transforms
    )

    original = dataset[10]
    print(original.shape)

    original = original.unsqueeze(1)
    print(f'Original Shape: {original.shape}')
    autoencoder = VisResMAE(in_dims=1, out_dims=1, kernel_size=3, activation='leaky', alpha=0.1)
    masked = mask_spectrogram(original, max_mask_size=50, deterministic=False)
    x, emb = autoencoder(original)
    print(f'Embedding Shape: {emb.shape}')
    print(f'Predicted Shape: {x.shape}')

    emb_transform = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1)
    emb = emb_transform(emb)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(original.squeeze().detach().numpy(), cmap='inferno', origin='lower')
    plt.title('Mel Spectrogram - Original')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')

    plt.subplot(1, 3, 2)
    plt.imshow(x.squeeze().detach().numpy(), cmap='inferno', origin='lower')
    plt.title('Mel Spectrogram - Processed')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')

    plt.subplot(1, 3, 3)
    plt.imshow(emb.squeeze().detach().numpy(), cmap='inferno', origin='lower')
    plt.title('Latent Representation')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')

    plt.tight_layout()
    plt.show()