import torch

class SpectrogramMasking:

    def __init__(self, mask_ratio=0.5, patch_size=16):

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
    def __call__(self, spec):

        B, C, F, T = spec.shape
        
        # Calculate number of patches, rounding up
        n_patches_f = (F + self.patch_size - 1) // self.patch_size
        n_patches_t = (T + self.patch_size - 1) // self.patch_size
        
        # Pad the input to be divisible by patch_size
        pad_f = n_patches_f * self.patch_size - F
        pad_t = n_patches_t * self.patch_size - T
        
        if pad_f > 0 or pad_t > 0:
            spec = torch.nn.functional.pad(spec, (0, pad_t, 0, pad_f))
        
        n_patches = n_patches_f * n_patches_t
        
        # Create patch mask
        n_masked = int(self.mask_ratio * n_patches)
        mask = torch.zeros(B, n_patches)
        for i in range(B):
            perm = torch.randperm(n_patches)
            mask[i, perm[:n_masked]] = 1
            
        # Expand mask to full size
        mask = mask.reshape(B, 1, n_patches_f, n_patches_t)
        mask = mask.repeat_interleave(self.patch_size, dim=2)
        mask = mask.repeat_interleave(self.patch_size, dim=3)
        
        # Cut back to original size
        mask = mask[:, :, :F, :T]
        
        # Move mask to same device as input
        mask = mask.to(spec.device)
        
        # Apply mask (replace masked regions with zeros)
        masked_spec = spec[:, :, :F, :T] * (1 - mask)
        
        return masked_spec, mask


def masked_mse_loss(pred, target, mask):
    loss = (pred - target) ** 2
    loss = loss * mask
    return loss.mean()
