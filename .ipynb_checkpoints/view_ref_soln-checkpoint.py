#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[10]:


# Loading data
x = np.load('x.npy') # x coordinates discretization vector

Nx = x.shape[0]
print("Num of discretization of x coordinates: ",Nx)

u0_train = np.load('u0_train_data.npy')
u0_test  = np.load('u0_test_data.npy')

Ntrain = u0_train.shape[0]
Ntest  = u0_test.shape[0]
print("Trainning data size:", Ntrain)
print("Testing data size:", Ntest)

u_train = np.load('u_train_data.npy')
u_test  = np.load('u_test_data.npy')

# ========== 单元格输出 ==========
# 流输出 (stdout/stderr):
# Num of discretization of x coordinates:  100
# Trainning data size: 2000
# Testing data size: 2000
# ==============================


# In[3]:


# Plotting a few training samples
fig,axs = plt.subplots(2,2,figsize=(10,10))
axs = axs.flatten()
for s in range (4):
    axs[s].plot(x,u0_train[s],label='u(x,0)')
    axs[s].plot(x,u_train[s],label='u(x,T)')
    axs[s].legend()
plt.savefig('result/samples.png',bbox_inches='tight')

# ========== 单元格输出 ==========
# 显示数据:
# <Figure size 1000x1000 with 4 Axes>
# ==============================


# ## Score-Based Conditional Diffusion Model

# In[16]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import tqdm # Use standard tqdm if notebook tqdm not available


# In[17]:


# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== 单元格输出 ==========
# 流输出 (stdout/stderr):
# Using device: cuda
# ==============================


# ### DM Architechture

# In[19]:


#!/usr/bin/env python
# coding: utf-8

# # ==========================================================================================
# # Generic Conditional Score-Based Diffusion Model Module
# #
# # Description:
# # This script provides a self-contained, problem-agnostic implementation of a
# # conditional score-based diffusion model. It is designed to solve inverse problems
# # by learning the conditional score function s(x_t, y, t) directly from paired data (x, y),
# # as described in "Unifying and extending Diffusion Models through PDEs for solving
# # Inverse Problems" (arXiv:2504.07437).
# # ==========================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
import functools
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# --- 1. SDE and Time Embedding Definition (Problem-Agnostic) ---

class GaussianFourierProjection(nn.Module):
    """
    Encodes scalar time-steps into a high-dimensional feature vector
    using a set of fixed (non-trainable) random Fourier features.
    """
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

def marginal_prob_std_fn(t, sigma):
    """
    Computes the standard deviation of p(x_t|x_0) for the VP-SDE.
    This corresponds to sigma_t in the notation p(x_t|x_0) = N(x_t; alpha_t*x_0, sigma_t^2*I).
    """
    t = torch.as_tensor(t)
    return torch.sqrt((sigma**(2 * t) - 1.) / (2. * np.log(sigma)))

def diffusion_coeff_fn(t, sigma):
    """Computes the diffusion coefficient g(t) for the VP-SDE."""
    t = torch.as_tensor(t)
    return sigma**t

# --- 2. Conditional Score Network (Problem-Agnostic) ---

class ConditionalScoreNet(nn.Module):
    """
    A time-dependent, conditional score-based model.
    It learns the score of the conditional distribution p_t(x|y), denoted as s(x, y, t).
    """
    def __init__(self, marginal_prob_std, x_dim, y_dim, hidden_depth=3, embed_dim=256):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        
        # Time and condition embedding layers
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        # A simple linear layer to embed the condition y
        self.condition_embed = nn.Linear(y_dim, embed_dim)
        
        # Main network layers
        self.input_layer = nn.Linear(x_dim, embed_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(hidden_depth)
        ])
        self.output_layer = nn.Linear(embed_dim, x_dim)
        
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x, y, t):
        # 1. Generate embeddings for time and condition
        t_embedding = self.act(self.time_embed(t))
        y_embedding = self.act(self.condition_embed(y))
        
        # 2. Combine embeddings (simple addition)
        combined_embedding = t_embedding + y_embedding
        
        # 3. Process x through the network, modulated by the combined embedding
        h = self.act(self.input_layer(x))
        for layer in self.hidden_layers:
            # Add embedding and apply residual connection
            h = self.act(layer(h) + combined_embedding)
        
        out = self.output_layer(h)
        
        # 4. Scale output by the marginal standard deviation (a key part of the design)
        out = out / self.marginal_prob_std(t)[:, None]
        return out

# --- 3. Loss Function (Problem-Agnostic) ---

def conditional_loss_fn(model, x, y, marginal_prob_std, eps=1e-5):
    """
    The loss function for training the conditional score model.
    Corresponds to the denoising score matching loss for p(x|y).
    """
    # Sample random time steps
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    
    # Sample noise and perturb data
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None]
    
    # Get model's score prediction
    score = model(perturbed_x, y, random_t)
    
    # Calculate loss
    loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=1))
    return loss

# --- 4. Training Infrastructure (Problem-Agnostic) ---

class PairedDataset(Dataset):
    """A dataset for (x, y) pairs."""
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def train_conditional_model(
    score_model,
    x_data,
    y_data,
    lr=1e-4,
    batch_size=256,
    n_epochs=500,
    device='cpu'
):
    """Trains the conditional score model."""
    dataset = PairedDataset(x_data.to(device), y_data.to(device))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = Adam(score_model.parameters(), lr=lr)
    score_model.train()
    
    tqdm_epoch = tqdm(range(n_epochs), desc="Training Epochs")
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x_batch, y_batch in data_loader:
            loss = conditional_loss_fn(score_model, x_batch, y_batch, score_model.marginal_prob_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x_batch.shape[0]
            num_items += x_batch.shape[0]
        
        tqdm_epoch.set_description(f"Epoch {epoch}: Avg Loss = {avg_loss / num_items:.4f}")

    score_model.eval()
    return score_model

# --- 5. Sampler (Problem-Agnostic) ---

def conditional_sampler(
    score_model,
    marginal_prob_std,
    diffusion_coeff,
    y_cond,
    batch_size,
    x_dim,
    num_steps=500,
    device='cpu',
    eps=1e-3
):
    """
    Generates samples from the conditional distribution p(x|y) using the trained score model.
    Implements the reverse-time SDE sampler (Euler-Maruyama method).
    """
    score_model.eval()
    t_init = torch.ones(batch_size, device=device)
    
    # Start from pure noise
    init_x = torch.randn(batch_size, x_dim, device=device) * marginal_prob_std(t_init)[:, None]
    
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    
    x = init_x
    y_cond_batch = y_cond.repeat(batch_size, 1).to(device)

    with torch.no_grad():
        for time_step in tqdm(time_steps, desc="Conditional Sampling"):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            
            # The core change: directly use the conditional score model
            score = score_model(x, y_cond_batch, batch_time_step)
            
            # Euler-Maruyama update
            mean_x = x + (g**2)[:, None] * score * step_size
            noise = torch.randn_like(x)
            x = mean_x + g[:, None] * torch.sqrt(step_size) * noise
            
    return mean_x



# In[ ]:




