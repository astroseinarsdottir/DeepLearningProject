import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Policy, Flatten, Encoder
from utils import make_env, Storage, orthogonal_init
import imageio

# Hyperparameters
total_steps = 8e6
num_envs = 32
num_levels = 10
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01


# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
n_input = eval_env.observation_space.shape[0]
num_actions = eval_env.action_space.n
in_channels = 3
encoder = Encoder(in_channels, n_input)
policy = Policy(encoder, n_input, num_actions)
policy.load_state_dict(torch.load('checkpoint.pt'))
policy.eval()
for _ in range(512):

    # Use policy
    action, log_prob, value = policy.act(obs)

    # Take step in environment
    obs, reward, done, info = eval_env.step(action)
    total_reward.append(torch.Tensor(reward))

    # Render environment and store
    frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
    frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('vid.mp4', frames, fps=25)
