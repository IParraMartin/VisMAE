import torch

import wandb
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from utilities.mask import *
import io
import sys
import os
sys.path.append(os.curdir)


def tensor_to_image(tensor):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(tensor, aspect='auto', origin='lower', cmap='magma')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))

def save_checkpoints(model, model_name, save_to):
   os.makedirs(os.path.join(save_to, 'checkpoints'), exist_ok=True)
   torch.save(model.state_dict(), os.path.join(save_to, 'checkpoints', f'{model_name}.pt'))

masking = SpectrogramMasking(mask_ratio=0.5, patch_size=16)


def train(device, model, epochs, train_dataloader, val_dataloader, optim, log, save_epochs, save_path):

    print(f'\nLogging to wandb: {log}')
    if log:
        global_step = 0
        wandb.init(project='VisResAE')
        wandb.watch(model, log='all', log_freq=1)

    print('\nTraining model...')
    model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5)

    print(f'\nUsing {scheduler.__class__.__name__} schedulling.')
    print(f'\nTraining on {device}')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_steps = 0
        for idx_batch, signal in enumerate(train_dataloader):
            original_signal = signal.to(device)
            masked_signal, mask = masking(original_signal.clone())
            out, _ = model(masked_signal)

            loss = masked_mse_loss(out, original_signal, mask)
            total_train_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            train_steps += 1

            if (idx_batch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] - Train Step [{idx_batch+1}/{len(train_dataloader)}] - Loss: {loss.item():.3f}')

            if log:
                global_step += 1
                wandb.log({
                    'train_steps': global_step,
                    'train_loss': loss.item()
                })

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for idx_batch, signal in enumerate(val_dataloader):
                original_signal = signal.to(device)
                masked_signal, mask = masking(original_signal.clone())
                out, _ = model(masked_signal)

                loss = masked_mse_loss(out, original_signal, mask)
                total_val_loss += loss.item()

                if (idx_batch + 1) % 5 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}] - Validation Step [{idx_batch+1}/{len(val_dataloader)}] - Loss: {loss.item():.3f}')
                
                original_img = tensor_to_image(original_signal[0].squeeze().cpu().detach().numpy())
                masked_img = tensor_to_image(masked_signal[0].squeeze().cpu().detach().numpy())
                reconstructed_img = tensor_to_image(out[0].squeeze().cpu().detach().numpy())

                if log:
                    wandb.log({
                        'original_spectrogram': wandb.Image(original_img, caption='Original Spectrogram'),
                        'masked_spectrogram': wandb.Image(masked_img, caption='Masked Spectrogram'),
                        'reconstructed_spectrogram': wandb.Image(reconstructed_img, caption='Reconstructed Spectrogram'),
                    })

        avg_val_loss = total_val_loss / len(val_dataloader)
        scheduler.step(avg_val_loss)

        if log:
            wandb.log({
                'epoch': epoch,
                'avg_train_loss': avg_train_loss,
                'avg_train_loss': avg_val_loss
            })

        if epoch + 1 % save_epochs == 0:
            save_checkpoints(model=model, model_name=f'melmae_{epoch+1}', save_to=save_path)

    print('Finished Training.')

    if log:
        wandb.finish()
