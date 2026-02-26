import torch
import torch.nn as nn

from rlinf.models.embodiment.modules.encoder import Encoder
from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.models.embodiment.modules.utils import init_mlp_weights, make_mlp


class MLPCritic(nn.Module):
    def __init__(self, 
    num_images_in_input, 
    cnn_features, 
    cnn_strides, 
    state_dim, 
    state_latent_dim, 
    critic_hidden_dims, 
    num_q_heads,
    action_dim):
        super(MLPCritic, self).__init__()
        self.num_images_in_input = num_images_in_input
        # Image encoders (one per camera view)
        self.encoders = nn.ModuleList()
        encoder_out_dim = 0
        for img_id in range(num_images_in_input):
            self.encoders.append(
                Encoder(cnn_features, cnn_strides, out_features=50) # from dsrl
            )
            encoder_out_dim += self.encoders[img_id].out_features

        # # State encoder

        self.state_proj = nn.Sequential(
            *make_mlp(
                in_channels=state_dim,
                mlp_channels=[
                    state_latent_dim,
                ],
                act_builder=nn.Tanh,
                last_act=True,
                use_layer_norm=True,
            )
        )
        init_mlp_weights(self.state_proj, nonlinearity="tanh")

        self.q_head = MultiQHead(hidden_size=encoder_out_dim + state_latent_dim, 
                                              action_feature_dim=action_dim, 
                                              hidden_dims=critic_hidden_dims, 
                                              output_dim=1,
                                              train_action_encoder=False,
                                              num_q_heads=num_q_heads)

    def forward(self, images, states, actions):
        states = states.to(torch.float32)
        visual_features = []
        for img_id in range(self.num_images_in_input):
            visual_features.append(self.encoders[img_id](images[img_id])) # [B, H, W, D]
        visual_features = torch.cat(visual_features, dim=1) # [B, D]
        state_features = self.state_proj(states) # [B, D]
        features = torch.cat([visual_features, state_features], dim=1) # [B, D]
        return self.q_head(features, actions) # [B, 1]