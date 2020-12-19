import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_train import make_env, Storage, orthogonal_init, saveArrayAsCSV, saveTensorAsCSV
from model_tuned import Flatten, Encoder, Policy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# Used to saved the mean reward
steps_score = []
steps_score_full = []

d = {'Step': [], 'Average_Reward': []}
df_reward = pd.DataFrame(data=d)

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="Run_"+str(np.random.random_integers(1,1e5)), help="", required=False)
parser.add_argument("--total_steps", type=float, default=8e6, help="", required=False)
parser.add_argument("--num_envs", type=int, default=32, help="", required=False)
parser.add_argument("--num_levels", type=int, default=10, help="", required=False)
parser.add_argument("--num_steps", type=int, default=256, help="", required=False)
parser.add_argument("--num_epochs", type=int, default=3, help="", required=False)
parser.add_argument("--batch_size", type=int, default=512, help="", required=False)
parser.add_argument("--eps", type=float, default=0.2, help="", required=False)
parser.add_argument("--grad_eps", type=float, default=0.5, help="", required=False)
parser.add_argument("--value_coef", type=float, default=0.5, help="", required=False)
parser.add_argument("--entropy_coef", type=float, default=0.01, help="", required=False)
args = parser.parse_args()

# Hyperparameters
run_name = args.run_name
total_steps = args.total_steps
num_envs = args.num_envs
num_levels = args.num_levels
num_steps = args.num_steps
num_epochs = args.num_epochs
batch_size = args.batch_size
eps = args.eps
grad_eps = args.grad_eps
value_coef = args.value_coef
entropy_coef = args.entropy_coef

if not os.path.exists(run_name):
    os.makedirs(run_name)
    

save_step = 1

# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels, env_name="coinrun")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space.n)


n_input = env.observation_space.shape[0]
num_actions = env.action_space.n
in_channels = 3
# Define network
encoder = Encoder(in_channels, n_input)
policy = Policy(encoder, n_input, num_actions)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5, weight_decay=0.0001)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(env.observation_space.shape, num_steps, num_envs)

# Run training
obs = env.reset()
step = 0

param_string = str(run_name)+"\nTotal steps: "+str(total_steps)+"\nNum envs: "+str(num_envs)+"\nNum actions: "+str(num_actions)+"\nN levels: "+str(num_levels)+"\nN epochs: "+str(num_epochs)+"\nBatch size: "+str(batch_size)+"\neps: "+str(eps)+"\ngrad_eps: "+str(grad_eps)+"\nValue coef: "+str(value_coef)+"\nEntropy coef: "+str(entropy_coef)

with open(run_name+'/infos.txt', 'w') as f:
    f.write(param_string)


while step < total_steps:

    # Use policy to collect data for num_steps steps
    policy.eval()
    for _ in range(num_steps):
        # Use policy
        action, log_prob, value = policy.act(obs)

        # Take step in environment
        next_obs, reward, done, info = env.step(action)

        # Store data
        storage.store(obs, action, reward, done, info, log_prob, value)

        # Update current observation
        obs = next_obs

    # Add the last observation to collected data
    _, _, value = policy.act(obs)
    storage.store_last(obs, value)

    # Compute return and advantage
    storage.compute_return_advantage()

    # Optimize policy
    policy.train()
    for epoch in range(num_epochs):

        # Iterate over batches of transitions
        generator = storage.get_generator(batch_size)
        for batch in generator:
            b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

            # Get current policy outputs
            new_dist, new_value = policy(b_obs)
            new_log_prob = new_dist.log_prob(b_action)

            # Clipped policy objective
            ratio = torch.exp(new_log_prob - b_log_prob)
            pi_1 = ratio * b_advantage
            pi_2 = ratio.clamp(1.0 - eps, 1.0 + eps) * b_advantage
            pi_loss = -torch.min(pi_1, pi_2).mean()

            # Clipped value function objective
            clipped_val = b_value + (new_value - b_value).clamp( -eps, eps) 
            val_s_1 = torch.pow(new_value - b_returns, 2)
            val_s_2 = torch.pow(clipped_val - b_returns, 2) 
            value_loss = 0.5 * torch.max(val_s_1, val_s_2).mean()

            # Entropy loss - Should read up on how other people are handeling this
            entropy_loss = new_dist.entropy().mean()

            # Backpropagate losses
            loss = pi_loss + value_coef * value_loss - entropy_coef * entropy_loss
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

            # Update policy
            optimizer.step()
            optimizer.zero_grad()

    # Update stats
    step += num_envs * num_steps
    

    
    df_current = {'Step': step, 'Average_Reward': storage.get_reward().item()}
    df_reward = df_reward.append(df_current, ignore_index = True) 
    #steps_score_full.append(storage.get_full_reward())
    print(f"Step: {step}\tMean reward: {storage.get_reward()}")
    #plt.plot(df_reward.Step, df_reward.Average_Reward)
    
    

    
 
    #fig = sns_plot.get_figure()
    #fig.savefig(run_name+'/last_captured_reward_step_.png')
    if (step > save_step):
        save_step += 30000
        #saveArrayAsCSV(steps_score, run_name,"average")
        df_reward.to_csv(run_name+'/reward.csv')
        #saveTensorAsCSV(steps_score_full, run_name,"full")
        df_reward.plot(x ='Step', y='Average_Reward', kind = 'line')
        plt.ylabel("Reward")
        plt.xlabel("Training step")
        plt.savefig(run_name+'/last_captured_reward_step_.png', format="png")
        plt.show()
        plt.close()



print("Completed training!")
saveArrayAsCSV(steps_score, run_name,"average")
saveTensorAsCSV(steps_score_full, run_name,"full")
torch.save(policy.state_dict(), run_name+'/'+"checkpoint.pt")
