import random

def mask_spectrogram(spectrogram, max_mask_size=50):
    
    """
    Applies a random square mask to a section of the spectrogram by setting it to zero.

    Parameters:
    - spectrogram: Tensor of shape [batch_size, channels, freq_bins, time_frames]
    - max_mask_size: Maximum size of the square mask (in bins/frames)
    """
    
    B, C, freq_bins, time_frames = spectrogram.shape
    # Determine the maximum possible mask size given the spectrogram dimensions
    max_possible_mask_size = min(max_mask_size, freq_bins, time_frames)
    # Randomly determine the size of the square mask
    mask_size = random.randint(1, max_possible_mask_size)
    # Randomly choose the starting point for the mask
    freq_start = random.randint(0, freq_bins - mask_size)
    time_start = random.randint(0, time_frames - mask_size)
    # Apply the square mask
    spectrogram[:, :, freq_start:freq_start+mask_size, time_start:time_start+mask_size] = 0
    return spectrogram