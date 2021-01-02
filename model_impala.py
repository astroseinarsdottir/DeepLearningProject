import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_train import make_env, Storage, orthogonal_init


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.layers = nn.Sequential(nn.ReLU(),

                                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=3, stride=1),
                                    nn.Dropout(),
                                    # Potentially add batch norm

                                    nn.ReLU(),

                                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=3, stride=1),
                                    nn.Dropout(),
                                    # Potentially add batch norm
                                    )

    def forward(self, x):
        residual = x
        x = self.layers(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()

        self.layers = nn.Sequential(
            # First iteration of sequrence, 16
            nn.Conv2d(in_channels=in_channels, out_channels=32,
                      kernel_size=3, stride=1),
            nn.MaxPool2d(stride=2, kernel_size=3),
            ResidualBlock(in_channels=16, out_channels=16),
            ResidualBlock(in_channels=16, out_channels=16),

            # Second iteration of sequence, 32
            nn.Conv2d(in_channels=in_channels, out_channels=32,
                      kernel_size=3, stride=1),
            nn.MaxPool2d(stride=2, kernel_size=3),
            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32),

            # Third iteration of sequence, 32
            nn.Conv2d(in_channels=in_channels, out_channels=32,
                      kernel_size=3, stride=1),
            nn.MaxPool2d(stride=2, kernel_size=3),
            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32),

            Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=feature_dim)
        )

        self.apply(orthogonal_init)

    def forward(self, x):
        return self.layers(x)


class Policy(nn.Module):
    def __init__(self, encoder, feature_dim, num_actions):
        super().__init__()
        self.encoder = encoder

        self.policy = orthogonal_init(
            nn.Linear(feature_dim, num_actions), gain=.01)
        self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

    def act(self, x):
        with torch.no_grad():
            x = x.cuda().contiguous()
            dist, value = self.forward(x)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu(), log_prob.cpu(), value.cpu()

    def forward(self, x):
        x = self.encoder(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(1)
        dist = torch.distributions.Categorical(logits=logits)

        return dist, value
