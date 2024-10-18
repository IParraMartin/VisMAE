import torch
import torchaudio
from torchvision.transforms import Compose
from torch.utils.data import random_split, DataLoader

import yaml
import sys
import os
sys.path.append(os.curdir)

from models.VisResMAE import VisResMAE
from dataset.dataset import AudioDataset
from loop import train


with open('/Users/inigoparra/Desktop/VisMAE/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def make_deterministic(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

make_deterministic(config['seed'])

transforms = Compose([
    torchaudio.transforms.Resample(orig_freq=44100, new_freq=config['target_sr']),
    torchaudio.transforms.MelSpectrogram(
        sample_rate=config['target_sr'],
        n_fft=256,                          # Small window for better temporal precision
        hop_length=64,                      # High overlap to capture temporal structure
        n_mels=64,                          # Mel bands (128 is common for speech)
        f_min=60,                           # Capture low-frequency components like voice
        f_max=4000,                         # Respect Nyquist limit at 8 kHz sample rate
    ),
    torchaudio.transforms.AmplitudeToDB()
])

dataset = AudioDataset(
    data_dir=config['audio_dir'],
    target_sr=config['target_sr'],
    sample_len=config['sample_len'],
    transformation=transforms
)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
criterion = torch.nn.MSELoss()
autoencoder = VisResMAE(**config['model_config'])
optim = torch.optim.Adam(
    params=autoencoder.parameters(), 
    lr=config['lr'],
    betas=(config['beta_1'], config['beta_2']),
    weight_decay=config['weight_decay']
)

train_size = int(0.8 * len(dataset))
val_size = int(len(dataset) - train_size)
train_split, val_split = random_split(dataset=dataset, lengths=[train_size, val_size])
train_dataloader = DataLoader(dataset=train_split, batch_size=config['batch_size'], shuffle=True)
val_dataloader = DataLoader(dataset=val_split, batch_size=config['batch_size'], shuffle=False)

train(
    device=device,
    model=autoencoder,
    epochs=config['epochs'],
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    criterion=criterion,
    optim=optim,
    log=config['log']
)