import torch
import torch.nn as nn
import torch.nn.functional as F

def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
	"""Orthogonal weight initialization: https://arxiv.org/abs/1312.6120"""
	if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
		nn.init.orthogonal_(module.weight.data, gain)
		nn.init.constant_(module.bias.data, 0)
	return module

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
            nn.BatchNorm1d(num_features=feature_dim),
            nn.Linear(feature_dim, 128,bias=False), nn.ReLU(),
            torch.nn.Dropout(p= 0.5, inplace= False),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 128,bias=False), nn.ReLU(),
            torch.nn.Dropout(p= 0.5, inplace= False),
            nn.Linear(128, 1)
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
