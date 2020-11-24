import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
from model import Flatten, Encoder, Policy

# Hyperparameters
total_steps = 8e6
num_envs = 64
num_levels = 10
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01

# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)


n_input = env.observation_space.shape[0]
num_actions = env.action_space.n
in_channels = 3
# Define network
encoder = Encoder(in_channels, n_input)
policy = Policy(encoder, n_input, num_actions)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

# Run training
obs = env.reset()
step = 0
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
            pi_2 = ratio.clamp(1. - eps, 1. + eps) * b_advantage
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
    print(f'Step: {step}\tMean reward: {storage.get_reward()}')

print('Completed training!')
torch.save(policy.state_dict(), 'checkpoint.pt')
