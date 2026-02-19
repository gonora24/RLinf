import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class GaussianTanhPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims=(256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        low: float | torch.Tensor | None = -1.0,
        high: float | torch.Tensor | None = 1.0,
    ):
        super().__init__()

        # Simple MLP encoder
        layers = []
        last_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.net = nn.Sequential(*layers)

        # Heads for mean and log_std
        self.mean_layer = nn.Linear(last_dim, action_dim)
        self.log_std_layer = nn.Linear(last_dim, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Action bounds
        if low is None:
            low = -1.0
        if high is None:
            high = 1.0

        # Convert to tensors for vector operations
        low = torch.tensor(low, dtype=torch.float32)
        high = torch.tensor(high, dtype=torch.float32)

        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Args:
            obs: [B, obs_dim]
            deterministic: if True, use mean action; else sample with rsample().

        Returns:
            action:    [B, action_dim], in [low, high]
            log_prob: [B, 1], log π(a|s) under this tanh-squashed Gaussian
            mean:     [B, action_dim], pre-tanh mean for diagnostics/eval
        """
        h = self.net(obs)

        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        dist = Normal(mean, std)

        if deterministic:
            # mean action before tanh
            x = mean
        else:
            # reparameterized sample for backprop
            x = dist.rsample()

        # log prob of pre-tanh sample
        log_prob = dist.log_prob(x).sum(dim=-1, keepdim=True)

        # tanh squashing
        y = torch.tanh(x)

        # scale to [low, high]
        action = y * self.action_scale + self.action_bias

        # Tanh correction term: log|det d(tanh^{-1}(a))/da|
        # Similar to RLinf flow actor’s tanh correction [2].
        # 1 - tanh(x)^2 = sech^2(x)
        # For tanh-squashed Gaussian, log_prob -= sum log(1 - y^2 + eps)
        eps = 1e-6
        tanh_correction = torch.sum(
            torch.log(1.0 - y.pow(2) + eps), dim=-1, keepdim=True
        )
        log_prob = log_prob - tanh_correction

        return action, log_prob, mean