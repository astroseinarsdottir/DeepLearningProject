import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_train import make_env, Storage, orthogonal_init


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(ch)

    def forward(self, x):
        residual = x
        out = F.relu(x, inplace=True)
        out = self.conv1(x)
        out = self.batch_norm(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.batch_norm(out)
        out = out + residual
        # out = out.view(x.size(0), -1)
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()

        self.layers = nn.Sequential(
            # First iteration of sequrence, 16
            nn.Conv2d(in_channels=in_channels, out_channels=16,
                      kernel_size=3, stride=1),
            nn.MaxPool2d(stride=2, kernel_size=3),
            ResidualBlock(16),
            ResidualBlock(16),

            # Second iteration of sequence, 32
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1),
            nn.MaxPool2d(stride=2, kernel_size=3),
            ResidualBlock(32),
            ResidualBlock(32),

            # Third iteration of sequence, 32
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1),
            nn.MaxPool2d(stride=2, kernel_size=3),
            ResidualBlock(32),
            ResidualBlock(32),

            Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=800, out_features=feature_dim)
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
