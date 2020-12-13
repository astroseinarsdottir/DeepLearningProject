import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Policy, Flatten, Encoder
from utils import make_env, Storage, orthogonal_init
import imageio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pt_file", type=str, default="checkpoint.pt", help="", required=False)
args = parser.parse_args()

# Hyperparameters
num_envs = 32
num_levels = 1
checkpoint_file = args.pt_file

# Make evaluation environment
eval_env = make_env(num_envs, start_level=10000, num_levels=num_levels, env_name="coinrun")
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
n_input = eval_env.observation_space.shape[0]
num_actions = eval_env.action_space.n
in_channels = 3
encoder = Encoder(in_channels, n_input)
policy = Policy(encoder, n_input, num_actions)
policy.load_state_dict(torch.load(checkpoint_file))
policy.eval()
policy.cuda()

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
