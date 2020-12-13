import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Policy, Flatten, Encoder
from utils import make_env, Storage, orthogonal_init
import imageio
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="", help="", required=False)
args = parser.parse_args()
run_name = args.run_name

# Hyperparameters
total_steps = 8e6
num_envs = 32
num_levels = 2
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01


grid_param = [[1,1]]

# Make evaluation environment
eval_env = make_env(num_envs, start_level=0, num_levels=0, env_name="coinrun")
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
n_input = eval_env.observation_space.shape[0]
num_actions = eval_env.action_space.n
in_channels = 3
encoder = Encoder(in_channels, n_input)
policy = Policy(encoder, n_input, num_actions)
policy.cuda()
policy.load_state_dict(torch.load(run_name+'/checkpoint.pt'))
policy.eval()

for _ in range(1024):
    # Use policy
    action, log_prob, value = policy.act(obs)

    # Take step in environment
    obs, reward, done, info = eval_env.step(action)
    total_reward.append(torch.Tensor(reward))

    # Render environment and store
    frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
    frames.append(frame)

# Calculate average return
stacked_total_reward = torch.stack(total_reward).sum(0)
total_reward = stacked_total_reward.mean(0)

print(stacked_total_reward)
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave(run_name+'.mp4', frames, fps=25)


validation_file_name = "validations.csv"
df_current = pd.DataFrame({'Run name': run_name, 'Average_Reward': total_reward.item()}, index=[0])

    
if not os.path.exists(validation_file_name):
    df_current.to_csv(validation_file_name, index=False)
else:
    df_loaded = pd.read_csv(validation_file_name, delimiter=',')
    df_save = df_loaded.append(df_current, ignore_index = True) 
    df_save.to_csv(validation_file_name, index=False)
