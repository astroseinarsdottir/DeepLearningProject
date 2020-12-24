import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_train import make_env, Storage, orthogonal_init, saveArrayAsCSV, saveTensorAsCSV
from model import Flatten, Encoder, Policy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

class Train:
    
    def __init__(
        self,
        run_name,
        total_steps,
        num_envs,
        num_levels,
        num_steps,
        num_epochs,
        batch_size,
        grad_eps,
        value_coef,
        entropy_coef,
        eps,
        distribution_mode
        ):
        # Hyperparameters
        self.run_name = run_name
        self.total_steps = total_steps
        self.num_envs = num_envs
        self.num_levels = num_levels
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.grad_eps = grad_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef        
        self.eps = eps
        
        self.step = 0
        self.distribution_mode = distribution_mode
        
        self.env = make_env(num_envs, num_levels=num_levels, env_name="coinrun", distribution_mode=distribution_mode)
        self.obs = self.env.reset()
        
        print("Observation space:", self.env.observation_space)
        print("Action space:", self.env.action_space.n)


        self.n_input = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.in_channels = 3
        
        # Define network
        self.encoder = Encoder(self.in_channels, self.n_input)
        self.policy = Policy(self.encoder, self.n_input, self.num_actions)
        self.policy.cuda()

        # Define optimizer
        # these are reasonable values but probably not optimals
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4, eps=1e-5, weight_decay=0.0001)

        # Define temporary storage
        # we use this to collect transitions during each iteration
        self.storage = Storage(self.env.observation_space.shape, self.num_steps, self.num_envs)
        
        
        # For tracking reward during training
        self.save_step = 1
        
        
        # Used to saved the mean reward
        self.steps_score = []
        self.steps_score_full = []

        d = {'Step': [], 'Average_Reward': []}
        self.df_reward = pd.DataFrame(data=d)

        # Create a folder for this run if there is not already one
        if not os.path.exists(self.run_name):
            os.makedirs(self.run_name)
    
    def __init__(
        self,
        run_name,
        total_steps,
        num_envs,
        num_levels,
        num_steps,
        num_epochs,
        batch_size,
        grad_eps,
        value_coef,
        entropy_coef,
        eps
        ):
        # Hyperparameters
        self.run_name = run_name
        self.total_steps = total_steps
        self.num_envs = num_envs
        self.num_levels = num_levels
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.grad_eps = grad_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef        
        self.eps = eps
        
        self.step = 0
        
        self.env = make_env(num_envs, num_levels=num_levels, env_name="coinrun")
        self.obs = self.env.reset()
        
        print("Observation space:", self.env.observation_space)
        print("Action space:", self.env.action_space.n)


        self.n_input = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.in_channels = 3
        
        # Define network
        self.encoder = Encoder(self.in_channels, self.n_input)
        self.policy = Policy(self.encoder, self.n_input, self.num_actions)
        self.policy.cuda()

        # Define optimizer
        # these are reasonable values but probably not optimals
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=5e-4, eps=1e-5, weight_decay=0.0001)

        # Define temporary storage
        # we use this to collect transitions during each iteration
        self.storage = Storage(self.env.observation_space.shape, self.num_steps, self.num_envs)
        
        
        # For tracking reward during training
        self.save_step = 1
        
        
        # Used to saved the mean reward
        self.steps_score = []
        self.steps_score_full = []

        d = {'Step': [], 'Average_Reward': []}
        self.df_reward = pd.DataFrame(data=d)

        # Create a folder for this run if there is not already one
        if not os.path.exists(self.run_name):
            os.makedirs(self.run_name)

        
    def clipped_value_loss(self,b_value, new_value, reward):
        # Clipped value function objective
        clipped_val = b_value + (new_value - b_value).clamp( -self.eps, self.eps) 
        val_loss_1 = torch.pow(new_value - reward, 2)
        val_loss_2 = torch.pow(clipped_val - reward, 2) 
        return torch.mean(torch.max(val_loss_1, val_loss_2))

    def save_model_info_to_txt(self):
        param_string = "Model informations : \n"
        param_string += str(self.run_name)
        param_string +="\nTotal steps: "+str(self.total_steps)
        param_string += "\nNum envs: "+str(self.num_envs)
        param_string +="\nNum actions: "+str(self.num_actions)
        param_string +="\nN levels: "+str(self.num_levels)
        param_string +="\nN epochs: "+str(self.num_epochs)
        param_string +="\nBatch size: "+str(self.batch_size)
        param_string +="\neps: "+str(self.eps)
        param_string +="\ngrad_eps: "+str(self.grad_eps)
        param_string +="\nValue coef: "+str(self.value_coef)
        param_string +="\nEntropy coef: "+str(self.entropy_coef)

        with open(self.run_name+'/infos.txt', 'w') as f:
            f.write(param_string)

    def clipped_policy_loss(self,b_log_prob, new_log_prob, b_advantage):
        # Clipped policy objective
        ratio = torch.exp(new_log_prob - b_log_prob)
        pi_1 = ratio * b_advantage
        pi_2 = ratio.clamp(1.0 - self.eps, 1.0 + self.eps) * b_advantage
        return -torch.min(pi_1, pi_2).mean()
        
    def evaluate_policy(self):
        # Use policy to collect data for num_steps steps
        self.policy.eval()
        
        for _ in range(self.num_steps):
            # Use policy
            action, log_prob, value = self.policy.act(self.obs)
            
            # Take step in environment
            next_obs, reward, done, info = self.env.step(action)

            # Store data
            self.storage.store(self.obs, action, reward, done, info, log_prob, value)

            # Update current observation
            self.obs = next_obs

        # Add the last observation to collected data
        _, _, value = self.policy.act(self.obs)
        self.storage.store_last(self.obs, value)

        # Compute return and advantage
        self.storage.compute_return_advantage()

        
    def train_policy_batch(self,batch):
        b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

        # Get current policy outputs
        new_dist, new_value = self.policy(b_obs)
        new_log_prob = new_dist.log_prob(b_action)

        # get clipped policy loss
        pi_loss = self.clipped_policy_loss(b_log_prob, new_log_prob, b_advantage)
        
        # get clipped value loss
        value_loss = self.clipped_value_loss(b_value, new_value, b_returns)
        

        # Entropy loss - Should read up on how other people are handeling this
        entropy_loss = new_dist.entropy().mean()

        # Backpropagate losses
        loss = pi_loss - self.value_coef * value_loss + self.entropy_coef * entropy_loss
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_eps)

        # Update policy
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def run_training(self):
        self.save_model_info_to_txt()
        
        while self.step < self.total_steps:

            self.evaluate_policy()

            # Optimize policy
            self.policy.train()
            
            for epoch in range(self.num_epochs):

                # Iterate over batches of transitions
                generator = self.storage.get_generator(self.batch_size)
                for batch in generator:
                    self.train_policy_batch(batch)


            # Update stats
            self.step += self.num_envs * self.num_steps
            
            self.save_reward_csv()
            
        self.end_train_save()
            
            
    def save_reward_csv(self):
        df_current = {'Step': self.step, 'Average_Reward': self.storage.get_reward().item()}
        self.df_reward = self.df_reward.append(df_current, ignore_index = True) 
        
        print(f"Step: {self.step}\tMean reward: {self.storage.get_reward()}")


        if (self.step > self.save_step):
            self.save_step += 30000
            self.df_reward.to_csv(self.run_name+'/reward.csv')
            self.df_reward.plot(x ='Step', y='Average_Reward', kind = 'line')
            plt.ylabel("Reward")
            plt.xlabel("Training step")
            plt.savefig(self.run_name+'/last_captured_reward_step_.png', format="png")
            plt.show()
            plt.close()
        
    def end_train_save(self):
        print("Completed training!")
        saveArrayAsCSV(self.steps_score, self.run_name,"average")
        saveTensorAsCSV(self.steps_score_full, self.run_name,"full")
        torch.save(self.policy.state_dict(), self.run_name+'/'+"checkpoint.pt")



def main():

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

    train_model = Train(
        args.run_name,
        args.total_steps,
        args.num_envs,
        args.num_levels,
        args.num_steps,
        args.num_epochs,
        args.batch_size,
        args.grad_eps,
        args.value_coef,
        args.entropy_coef,
        args.eps)
    
    train_model.run_training()

    def save_model_info_to_txt(self):
        param_string = "Model informations : \n"
        param_string += str(self.run_name)
        param_string +="\nTotal steps: "+str(self.total_steps)
        param_string += "\nNum envs: "+str(self.num_envs)
        param_string +="\nNum actions: "+str(self.num_actions)
        param_string +="\nN levels: "+str(self.num_levels)
        param_string +="\nN epochs: "+str(self.num_epochs)
        param_string +="\nBatch size: "+str(self.batch_size)
        param_string +="\neps: "+str(self.eps)
        param_string +="\ngrad_eps: "+str(self.grad_eps)
        param_string +="\nValue coef: "+str(self.value_coef)
        param_string +="\nEntropy coef: "+str(self.entropy_coef)
        param_string +="\nDistribution mode: "+str(self.distribution_mode)

        with open(self.run_name+'/infos.txt', 'w') as f:
            f.write(param_string)
            print(self.encoder, file=f)
            print(self.policy, file=f)

    def clipped_policy_loss(self,b_log_prob, new_log_prob, b_advantage):
        # Clipped policy objective
        ratio = torch.exp(new_log_prob - b_log_prob)
        pi_1 = ratio * b_advantage
        pi_2 = ratio.clamp(1.0 - self.eps, 1.0 + self.eps) * b_advantage
        return -torch.min(pi_1, pi_2).mean()
    
    def clipped_value_loss(self,b_value, new_value, reward):
        # Clipped value function objective
        clipped_val = b_value + (new_value - b_value).clamp( -self.eps, self.eps) 
        val_loss_1 = torch.pow(new_value - reward, 2)
        val_loss_2 = torch.pow(clipped_val - reward, 2) 
        return torch.max(val_loss_1, val_loss_2).mean()
    
    def evaluate_policy(self):
        # Use policy to collect data for num_steps steps
        self.policy.eval()
        
        for _ in range(self.num_steps):
            # Use policy
            action, log_prob, value = self.policy.act(self.obs)
            
            # Take step in environment
            next_obs, reward, done, info = self.env.step(action)

            # Store data
            self.storage.store(self.obs, action, reward, done, info, log_prob, value)

            # Update current observation
            self.obs = next_obs

        # Add the last observation to collected data
        _, _, value = self.policy.act(self.obs)
        self.storage.store_last(self.obs, value)

        # Compute return and advantage
        self.storage.compute_return_advantage()

        
    def train_policy_batch(self,batch):
        b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

        # Get current policy outputs
        new_dist, new_value = self.policy(b_obs)
        new_log_prob = new_dist.log_prob(b_action)

        # get clipped policy loss
        pi_loss = self.clipped_policy_loss(b_log_prob, new_log_prob, b_advantage)
        
        # get clipped value loss
        value_loss = self.clipped_value_loss(b_value, new_value, b_returns)
        

        # Entropy loss - Should read up on how other people are handeling this
        entropy_loss = new_dist.entropy().mean()

        # Backpropagate losses
        loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_eps)

        # Update policy
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def run_training(self):
        self.save_model_info_to_txt()
        
        while self.step < self.total_steps:

            self.evaluate_policy()

            # Optimize policy
            self.policy.train()
            
            for epoch in range(self.num_epochs):

                # Iterate over batches of transitions
                generator = self.storage.get_generator(self.batch_size)
                for batch in generator:
                    self.train_policy_batch(batch)


            # Update stats
            self.step += self.num_envs * self.num_steps
            
            self.save_reward_csv()
            
        self.end_train_save()
            
            
    def save_reward_csv(self):
        df_current = {'Step': self.step, 'Average_Reward': self.storage.get_reward().item()}
        self.df_reward = self.df_reward.append(df_current, ignore_index = True) 
        
        print(f"Step: {self.step}\tMean reward: {self.storage.get_reward()}")


        if (self.step > self.save_step):
            self.save_step += 30000
            self.df_reward.to_csv(self.run_name+'/reward.csv')
            self.df_reward.plot(x ='Step', y='Average_Reward', kind = 'line')
            plt.ylabel("Reward")
            plt.xlabel("Training step")
            plt.savefig(self.run_name+'/last_captured_reward_step_.png', format="png")
            plt.show()
            plt.close()
        
    def end_train_save(self):
        print("Completed training!")
        saveArrayAsCSV(self.steps_score, self.run_name,"average")
        saveTensorAsCSV(self.steps_score_full, self.run_name,"full")
        torch.save(self.policy.state_dict(), self.run_name+'/'+"checkpoint.pt")



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="Run_"+str(np.random.random_integers(1,1e5)), help="", required=False)
    parser.add_argument("--distribution_mode", type=str, default="easy", help="", required=False)
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

    train_model = Train(
        args.run_name,
        args.total_steps,
        args.num_envs,
        args.num_levels,
        args.num_steps,
        args.num_epochs,
        args.batch_size,
        args.grad_eps,
        args.value_coef,
        args.entropy_coef,
        args.eps,
        args.distribution_mode)
    
    train_model.run_training()

if __name__ == "__main__":
    main()

