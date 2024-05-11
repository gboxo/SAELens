import torch
import torch.nn.functional as F

import einops
from torch import nn

from dataclasses import dataclass
from typing import Any, Optional, cast, NamedTuple



## KL divergence

def gaussain_kl_divergence(mu,log_var):
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=-1).mean()
    return kld




# Resampling

def gaussian_reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    gaussian = eps.mul(std).add_(mu)
    return gaussian     