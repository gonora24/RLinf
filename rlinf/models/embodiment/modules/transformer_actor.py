import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionChunkTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 10,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        img_channels: int = 3,
        img_size: int = 64,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        low: float = -1.0,
        high: float = 1.0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.low = low
        self.high = high

        # Positional embeddings for action chunk, learnable better for smaller sequences
        self.pos_emb = nn.Parameter(torch.randn(chunk_size, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Each action token predicts mean and log_std
        self.out = nn.Linear(d_model, action_dim * 2)

    def forward(self, state_features: torch.Tensor, image_features: torch.Tensor):
        """
        Args:
            state_features: (B, state_dim)
            image_features: (B, image_dim)
        Returns:
            mu: (B, chunk_size, action_dim)
            std: (B, chunk_size, action_dim)
        """
        B = state_features.shape[0]

        # Shape [B, state_dim + image_dim]
        context = torch.cat([state_features, image_features], dim=-1)  # (B, state_dim + image_dim)

        # Repeat context for each action token + add positional embedding
        x = context.unsqueeze(1) + self.pos_emb.unsqueeze(0)  # (B, chunk_size, d_model)

        h = self.transformer(x)  # (B, chunk_size, d_model)

        out = self.out(h)  # (B, chunk_size, action_dim * 2)
        mu, log_std = out.chunk(2, dim=-1)
        # Optionally clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        return mu, std

    def sample(self, state: torch.Tensor, image: torch.Tensor, deterministic: bool = False):
        """
        Sample a chunk of actions in parallel (batch + chunk) with correct log_prob.
        Args:
            state: [B, state_dim]
            image: [B, C, H, W]
            deterministic: Whether to use mean
        Returns:
            actions: [B, chunk_size, action_dim]
            log_prob: [B]  # summed over chunk and action dim
        """
        mean, std = self.forward(state, image)  # [B, chunk_size, action_dim]

        if deterministic:
            # deterministic policy: use mean
            actions = torch.tanh(mean)
            log_prob = torch.zeros(state.shape[0], device=state.device)
            return actions, log_prob

        # Stochastic sampling
        normal = torch.distributions.Normal(mean, std)
        base_dist = torch.distributions.Independent(normal, 1)  # treats last dim as multivariate

        # rsample for gradient flow
        x = base_dist.rsample()  # [B, chunk_size, action_dim]
        y = torch.tanh(x)

        # log_prob with tanh correction
        log_prob = base_dist.log_prob(x)  # [B, chunk_size]
        # Jacobian: sum over action_dim
        log_prob -= torch.sum(torch.log(1 - y.pow(2) + 1e-7), dim=-1)  # still [B, chunk_size]

        # sum over chunk dimension to get single log_prob per batch
        log_prob = log_prob.sum(dim=1)  # [B]

        # optional rescale to [low, high]
        if self.low is not None and self.high is not None:
            scale = (self.high - self.low) / 2.0
            shift = (self.high + self.low) / 2.0
            actions = y * scale + shift
            log_prob -= torch.sum(torch.log(scale * torch.ones_like(y)), dim=[1,2])

        else:
            actions = y

        return actions, log_prob


class AutoregressiveActionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        image_dim: int,
        action_dim: int,
        chunk_size: int = 8,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        low: float = -1.0,
        high: float = 1.0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.d_model = d_model
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.low = low
        self.high = high
        
        self.context_proj = nn.Linear(state_dim + image_dim, d_model, bias=False)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out = nn.Linear(d_model, action_dim * 2)

    def _dist(self, h):
        out = self.out(h)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mu, std

    def _sample_tanh(self, mu, std):
        normal = torch.distributions.Normal(mu, std)
        base = torch.distributions.Independent(normal, 1)

        x = base.rsample()
        y = torch.tanh(x)

        log_prob = base.log_prob(x)
        log_prob -= torch.sum(torch.log(1 - y.pow(2) + 1e-7), dim=-1)

        if self.low is not None and self.high is not None:
            scale = (self.high - self.low) / 2.0
            shift = (self.high + self.low) / 2.0
            y = y * scale + shift
            log_prob -= self.action_dim * torch.log(
                torch.tensor(scale, device=y.device)
            )

        return y, log_prob

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: [B, state_dim + image_dim]
        Returns:
            mu: [B, chunk_size, action_dim]
            std: [B, chunk_size, action_dim]
        """
        B = features.shape[0]
        context = self.context_proj(features).unsqueeze(1)  # [B, 1, state_dim + image_dim] as memory for transformer
        # Shape [B, 1, state_dim + image_dim] -> Shape [B, 1, d_model]
        h = self.transformer(context)  # [B, chunk_size, d_model]
        out = self.out(h).squeeze(1)  # [B, action_dim*2]
        mu, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mu, std

    def sample(self, features: torch.Tensor, deterministic: bool = False):
        """Sample a chunk of actions with correct log_prob.
        Args:
            features: [B, state_dim + image_dim]
            deterministic: Whether to use mean
        Returns:
            actions: [B, chunk_size, action_dim]
            log_prob: [B]  # summed over chunk and action dim
        """
        B = features.shape[0]
        # Encode context
        context = self.context_proj(features).unsqueeze(1)  # [B, 1, state_dim + image_dim] as memory for transformer

        actions = []
        log_probs = []

        # Initialize first token
        # tokens = self.start_token.expand(B, 1, -1)
        tokens = context

        for t in range(self.chunk_size):
            tgt = tokens + self.pos_emb[:, :tokens.size(1)]

            mask = torch.triu(
                torch.ones(tgt.size(1), tgt.size(1), device=tgt.device),
                diagonal=1,
            ).bool()

            # h = self.transformer(tgt, context, tgt_mask=mask)
            h = self.transformer(tgt, mask=mask, is_causal=True)
            h_last = h[:, -1]

            mu, std = self._dist(h_last)

            if deterministic:
                action = torch.tanh(mu)
                log_prob = torch.zeros(B, device=tgt.device)
            else:
                action, log_prob = self._sample_tanh(mu, std)

            actions.append(action)
            log_probs.append(log_prob)

            # IMPORTANT: append, don't overwrite
            new_token = self.action_proj(action).unsqueeze(1)
            tokens = torch.cat([tokens, new_token], dim=1)

        # Stack actions and log_probs
        actions = torch.stack(actions, dim=1)  # [B, chunk_size, action_dim]
        log_probs = torch.stack(log_probs, dim=1).mean(dim=1)  # mean over chunk → [B]

        return actions, log_probs