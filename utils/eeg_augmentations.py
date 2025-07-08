
import torch
import numpy as np
from scipy import signal
import random

class EEGAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def time_shift_batch(self, x, max_shift_ms=50, fs=250):
        """
        Time shift for batch of samples
        x shape: (batch_size, 1, channels, time_points)
        """
        if np.random.random() > self.prob:
            return x
            
        batch_size = x.shape[0]
        max_shift_samples = int(max_shift_ms * fs / 1000)
        
        # Generate different shift for each sample in batch
        shifts = np.random.randint(-max_shift_samples, max_shift_samples + 1, size=batch_size)
        
        x_shifted = x.copy()
        
        for i, shift in enumerate(shifts):
            if shift != 0:
                # Apply shift to single sample
                x_shifted[i] = np.roll(x[i], shift, axis=2)  # axis=2 for (1, channels, time)
                
                # Zero out wrapped portion
                if shift > 0:
                    x_shifted[i, :, :, :shift] = 0
                else:
                    x_shifted[i, :, :, shift:] = 0
        
        return x_shifted
    
    def add_noise_batch(self, x, noise_factor=0.02):
        """
        Add noise to batch
        x shape: (batch_size, 1, channels, time_points)
        """
        if np.random.random() > self.prob:
            return x
            
        # Calculate noise per sample and channel
        noise_std = noise_factor * np.std(x, axis=3, keepdims=True)  # axis=3 for time
        noise = np.random.normal(0, noise_std, x.shape)
        return x + noise
    
    def amplitude_scale_batch(self, x, scale_range=(0.8, 1.2)):
        """
        Scale amplitude for batch
        x shape: (batch_size, 1, channels, time_points)
        """
        if np.random.random() > self.prob:
            return x
            
        batch_size, _, channels, _ = x.shape
        
        # Different scaling per sample and channel
        scales = np.random.uniform(
            scale_range[0], scale_range[1], 
            (batch_size, 1, channels, 1)  # Broadcast-compatible shape
        )
        return x * scales
    
    def channel_dropout_batch(self, x, dropout_prob=0.1):
        """
        Channel dropout for batch
        x shape: (batch_size, 1, channels, time_points)
        """
        if np.random.random() > self.prob:
            return x
            
        batch_size, _, channels, time_points = x.shape
        n_drop = int(channels * dropout_prob)
        
        if n_drop == 0:
            return x
            
        x_aug = x.copy()
        
        for i in range(batch_size):
            # Different dropout pattern per sample
            drop_channels = np.random.choice(channels, n_drop, replace=False)
            
            for ch in drop_channels:
                # Interpolate from neighbors
                neighbors = [max(0, ch-1), min(channels-1, ch+1)]
                x_aug[i, 0, ch, :] = np.mean(x_aug[i, 0, neighbors, :], axis=0)
        
        return x_aug
    
    def __call__(self, x):
        """
        Apply augmentations to batch
        x shape: (batch_size, 1, channels, time_points)
        """
        x_aug = x.copy()
        
        x_aug = self.time_shift_batch(x_aug)
        x_aug = self.add_noise_batch(x_aug)
        x_aug = self.amplitude_scale_batch(x_aug)
        x_aug = self.channel_dropout_batch(x_aug)
        
        return x_aug
