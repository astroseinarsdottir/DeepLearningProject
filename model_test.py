import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32,
                      kernel_size=8, stride=4), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=4, stride=2), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            Flatten(),
            torch.nn.Dropout(p= 0.3, inplace= False),
            nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
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
        # self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)
        
        self.value = nn.Sequential(
            nn.Linear(feature_dim, 64,bias=True), nn.Tanh(),
            nn.Linear(64, 64,bias=True), nn.Tanh(),
            nn.Linear(64, 1,bias=True),nn.ReLU()
        )
        

        
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
