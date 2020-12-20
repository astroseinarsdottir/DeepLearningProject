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
    def __init__(self, encoder, feature_dim, num_actions, hidden_dim):
        super().__init__()
        self.encoder = encoder
        
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )

    def act(self, x):
        with torch.no_grad():
            x = x.cuda().contiguous()
            dist = self.forward(x)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu(), log_prob.cpu()

    def forward(self, x):
        x = self.encoder(x)
        logits = self.policy(x)

        dist = torch.distributions.Categorical(logits=logits)

        return dist

class Critic(nn.Module):
    def __init__(self, encoder, state_dim, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        with torch.no_grad():
            x = x.cuda().contiguous()
            x = self.encoder(x)
        return self.net(x)