import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.modules.utils import init_mlp_weights

class Encoder(nn.Module):
    """
    Encoder module for DSRL Implementation
    """
    def __init__(self, cnn_features=(32, 32, 32, 32), cnn_strides=(2, 1, 1, 1), input_channels=3, out_features=50):
        super(Encoder, self).__init__()
        
        self.features = cnn_features
        self.strides = cnn_strides
        self.out_features = out_features
        # We need to build the layers explicitly
        layers = []
        in_channels = input_channels
        
        for feat, stride in zip(self.features, self.strides):
            layers.append(
                nn.Conv2d(
                    in_channels, 
                    out_channels=feat, 
                    kernel_size=3, 
                    stride=stride, 
                    padding=0  # 'VALID' in JAX means no padding
                )
            )
            layers.append(nn.ReLU())
            in_channels = feat
        self.main = nn.Sequential(*layers)
        self.mlp = nn.Sequential(
            nn.Linear(self.features[-1], out_channels=self.out_features), # hardcoded from dsrl
            nn.LayerNorm(self.out_features),
            nn.Tanh()
        )
        init_mlp_weights(self.mlp, nonlinearity="tanh")

        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        x = self.main(x)

        x = self.mlp(x)
        return x