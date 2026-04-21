import torch
import torch.nn as nn


class SeqQFuncTorch(nn.Module):
    def __init__(
        self,
        state_dim,
        image_dim,
        action_dim,
        n_embed,
        n_heads,
        n_layer,
        action_horizon,
        dropout_rate,
        num_q_heads=1,
    ):
        super().__init__()
        self.context_proj = nn.Linear(state_dim + image_dim, n_embed, bias=False)
        self.action_embed = nn.Linear(action_dim, n_embed, bias=False)
        self.pos_embed = nn.Embedding(1 + action_horizon, n_embed)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embed,
            nhead=n_heads,
            dim_feedforward=4 * n_embed,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.norm = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, num_q_heads)

    def forward(self, state_features, image_features, actions):
        """
        Args:
            state_features:  [B, state_dim]
            image_features:  [B, image_dim]
            actions:         [B, T, action_dim]
        Returns:
            q_values:        [B, T, num_q_heads]
        """
        B, T, _ = actions.shape

        context = torch.cat([state_features, image_features], dim=-1)  # [B, state_dim+image_dim]
        context_emb = self.context_proj(context).unsqueeze(1)          # [B, 1, n_embed]
        action_emb = self.action_embed(actions)                        # [B, T, n_embed]
        seq = torch.cat([context_emb, action_emb], dim=1)              # [B, 1+T, n_embed]

        pos = self.pos_embed(torch.arange(1 + T, device=seq.device))   # [1+T, n_embed]
        x = seq + pos

        x = self.transformer(x)
        x = self.norm(x)
        x = self.head(x)            # [B, 1+T, out_dim]
        return x[:, 1:, :]          # drop context token, keep action Q-values
