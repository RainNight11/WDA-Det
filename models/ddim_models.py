# models/ddim_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DDIM(nn.Module):
    def __init__(self, model, timesteps=1000, beta_schedule='linear'):
        super(DDIM, self).__init__()
        self.model = model
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, timesteps)
        else:
            raise NotImplementedError("Only linear beta schedule is supported")

    def forward(self, x_start):
        batch_size = x_start.shape[0]
        x_t = x_start
        for t in reversed(range(self.timesteps)):
            epsilon = self.model(x_t, t)  # Model generates epsilon (predicted noise)
            
            alpha_t = 1 - self.betas[t]
            sigma_t = torch.sqrt(self.betas[t])
            
            x_t = (x_t - epsilon * sigma_t) / torch.sqrt(alpha_t)
            x_t = torch.clamp(x_t, -1.0, 1.0)
        
        return x_t
