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
        action_dim: int,
        chunk_size: int = 8,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        img_channels: int = 3,
        img_size: int = 64,
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

        self.action_proj = nn.Linear(action_dim, d_model)
        self.start_token = nn.Parameter(torch.zeros(d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*2,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.out = nn.Linear(d_model, action_dim * 2)

    def forward(self, state_features: torch.Tensor, image_features: torch.Tensor):
        """
        Args:
            state_features: [B, state_dim]
            image_features: [B, image_dim]
        Returns:
            mu: [B, chunk_size, action_dim]
            std: [B, chunk_size, action_dim]
        """
        B = state_features.shape[0]
        context = torch.cat([state_features, image_features], dim=-1)  # [B, state_dim + image_dim]
        context = context.unsqueeze(1)  # [B, 1, state_dim + image_dim] as memory for transformer
        # Shape [B, 1, state_dim + image_dim] -> Shape [B, 1, d_model]
        h = self.transformer(context)  # [B, chunk_size, d_model]
        out = self.out(h).squeeze(1)  # [B, action_dim*2]
        mu, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mu, std

    def sample(self, state_features: torch.Tensor, image_features: torch.Tensor, deterministic: bool = False):
        """Sample a chunk of actions with correct log_prob.
        Args:
            state: [B, state_dim]
            image: [B, C, H, W]
            deterministic: Whether to use mean
        Returns:
            actions: [B, chunk_size, action_dim]
            log_prob: [B]  # summed over chunk and action dim
        """
        B = state_features.shape[0]
        # Encode context
        context = torch.cat([state_features, image_features], dim=-1)  # [B, state_dim + image_dim]
        context = context.unsqueeze(1)  # [B, 1, state_dim + image_dim] as memory for transformer

        actions = []
        log_probs = []

        # Initialize first token
        prev_token = self.start_token.unsqueeze(0).repeat(B, 1)  # [B, 1, d_model]

        for t in range(self.chunk_size):
            # Embed previous action token
            token_embed = prev_token.unsqueeze(1)  # [B,1,d_model]

            # Transformer decoder: token attends to context (memory)
            h = self.transformer(token_embed, context)  # [B,1,d_model]
            out = self.out(h).squeeze(1)  # [B, action_dim*2]
            mu, log_std = out.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = log_std.exp()

            # Sample / deterministic
            if deterministic:
                action = torch.tanh(mu)
                log_prob = torch.zeros(B, device=state_features.device)
            else:
                # Create base Normal distribution (wrapped as multivariate via Independent)
                normal = torch.distributions.Normal(mu, std)
                base_dist = torch.distributions.Independent(normal, 1)

                # Sample from base distribution (pre-tanh space)
                x_t = base_dist.rsample()  # [B, output_dim], supports gradient flow

                # Manually apply tanh transform
                y_t = torch.tanh(x_t)  # [B, output_dim], in [-1, 1]

                # Compute log_prob (CleanRL style)
                # 1. Base distribution log_prob (in pre-tanh space)
                # 2. Subtract Jacobian correction: log|det J| = sum(log(1 - y_i^2))
                log_prob = base_dist.log_prob(x_t)  # [B], already summed
                log_prob -= torch.sum(
                    torch.log(1 - y_t.pow(2) + 1e-7), dim=-1
                )  # Jacobian correction

                # Optional: rescale to [low, high]
                if self.low is not None and self.high is not None:
                    scale_factor = (self.high - self.low) / 2.0
                    shift = (self.high + self.low) / 2.0
                    action = y_t * scale_factor + shift
                    # Subtract log(scale_factor) from log_prob
                    log_prob -= torch.sum(
                        torch.log(abs(scale_factor) * torch.ones_like(y_t)), dim=-1
                    )
                else:
                    action = y_t

            actions.append(action)
            log_probs.append(log_prob)

            # Set prev_token for next step (autoregressive)
            # Project sampled action back to d_model
            prev_token = self.action_proj(action)

        # Stack actions and log_probs
        actions = torch.stack(actions, dim=1)  # [B, chunk_size, action_dim]
        log_probs = torch.stack(log_probs, dim=1).sum(dim=1)  # sum over chunk → [B]

        return actions, log_probs