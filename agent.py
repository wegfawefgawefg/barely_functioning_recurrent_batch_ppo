import math
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from models import Actor, Critic, LSTM

'''
TODO:
-seperate actor and critic optimizers
    |-update them on seperate intervals like TD3 if possible
        |- does this interfere with actor epochs??

-add entropy maximization as exploration
-try letting network output the deviation
-add deterministic mode
-add network load and save
-add graphing
-add lstm (might want that to be alternate version)
'''

class Agent():
    def __init__(self, state_shape, action_shape, stats):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.state_shape = state_shape
        self.action_shape = action_shape

        self.stats = stats

        self.learn_rate     = 1e-4
        self.num_epochs     = 8

        self.entropy_weight = 0.001
        self.kl_clip        = 0.8

        self.hidden_state_size = 64
        self.lstm   = LSTM(self.state_shape, self.hidden_state_size).to(self.device)
        self.actor  = Actor(self.hidden_state_size, self.action_shape).to(self.device)
        self.critic = Critic(self.hidden_state_size).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.learn_rate)

    def act(self, state, hiddens, deterministic=False):\
                #   recurrences stuff
        with torch.no_grad():
            state = torch.Tensor(state).float().to(self.device)
            state = state.view(1, 1, self.lstm.data_shape)
            output, (hidden_state, cell_state) = self.lstm(state, hiddens)
            state = output[:, -1, :]

            if deterministic:
                mu = self.actor.forward_deterministic(state)
                action = mu
            else:
                policy_dist = self.actor(state)
                action = policy_dist.sample()
            action = action.clamp(-1, 1)    #   depends on env
            action = action.cpu().numpy()[0]
            return action, (hidden_state, cell_state)

    def learn(self, rollout_collector):
        for _ in range(self.num_epochs):
            for state, action, old_log_probs, advantage, return_ in rollout_collector.random_batch_iter():
                policy_dist = self.actor(state)
                value       = self.critic(state)
                new_log_probs = policy_dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.kl_clip, 1.0 + self.kl_clip) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                entropy = policy_dist.entropy().mean()
                loss = 0.5 * critic_loss + actor_loss - self.entropy_weight * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.stats.update_training_stats(
            num_samples_processed_inc=\
                rollout_collector.batch_size * rollout_collector.rollout_length * self.num_epochs)

