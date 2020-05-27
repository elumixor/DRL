from collections import namedtuple
from typing import List, Tuple, Dict

import gym
import numpy as np
import torch
import torch.nn as nn
from torch import optim, distributions
from torch.distributions import Categorical

from utils import RLAgent, train, torch_device, estimate_advantages, normalize, flatten
from utils.agents import MemoryAgent

# Hyper parameters
epochs = 10000
num_rollouts = 10

actor_hidden = 32
critic_hidden = 32

gamma = 0.99
lam = 0.95

train_pi_iters = 50
train_v_iters = 50

epsilon = 0.2  # clip epsilon
actor_lr = 3e-4
critic_lr = 1e-3
target_kl = 0.01


def discount_cumsum(arr, discount, last=0):
    discounted = [0.] * len(arr)
    for i in reversed(range(len(arr))):
        last = discounted[i] = arr[i] + discount * last

    return discounted


class PPOAgent(MemoryAgent):
    def __init__(self, env):
        super().__init__(env)

        obs_size = env.observation_space.shape[0]
        num_actions = env.action_space.n

        self.actor = nn.Sequential(nn.Linear(obs_size, actor_hidden),
                                   nn.ReLU(),
                                   nn.Linear(actor_hidden, num_actions))

        self.critic = nn.Sequential(nn.Linear(obs_size, actor_hidden),
                                    nn.ReLU(),
                                    nn.Linear(actor_hidden, 1))

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.values = []
        self.advantages = []
        self.discounted = []

    def get_action(self, state: np.ndarray) -> int:
        return Categorical(logits=self.actor(torch.from_numpy(state).float().unsqueeze(0))).sample().item()

    def save_step(self, action: int, reward: float, next_state: np.ndarray) -> None:
        super().save_step(action, reward, next_state)

    def on_trajectory_finished(self) -> None:
        # estimate advantages and value targets
        rewards = self.current_rewards
        states = torch.as_tensor(self.current_states).float()
        values = self.critic(states).flatten().tolist()

        # GAE
        deltas = [rewards[i] + gamma * values[i] - values[i] for i in range(len(rewards))]
        advantages = discount_cumsum(deltas, gamma * lam)

        # Rewards-to-go
        discounted = discount_cumsum(self.current_rewards, gamma)

        self.values += values
        self.advantages += advantages
        self.discounted += discounted

        super().on_trajectory_finished()

    def reset_memory(self):
        super().reset_memory()

        self.values = []
        self.advantages = []
        self.discounted = []

    @property
    def tensored_data(self) -> Dict[str, torch.Tensor]:
        values = torch.as_tensor(self.values, dtype=torch.float)
        advantages = normalize(torch.as_tensor(self.advantages, dtype=torch.float))
        discounted = torch.as_tensor(self.discounted, dtype=torch.float)

        return {**super().tensored_data,
                'values': values, 'advantages': advantages, 'discounted': discounted}

    def update_critic(self, advantages):
        loss = .5 * (advantages ** 2).mean()  # MSE
        self.opt_critic.zero_grad()
        loss.backward()
        self.opt_critic.step()

    def model(self, state):
        dist = distributions.Categorical(logits=self.actor(state))
        value = self.critic(state).squeeze(1)
        return dist, value

    def update(self) -> None:
        self.actor.to(torch_device)
        self.critic.to(torch_device)

        data = self.tensored_data
        for t in data.values():
            assert not t.requires_grad  # all of these tensors are treated as constants!

        data = {k: v.to(torch_device) for k, v in data.items()}

        actions = data['actions']
        states = data['states']
        advantages = data['advantages']

        logits_old = Categorical(logits=self.actor(states)).log_prob(actions).detach()  # constant!

        for i in range(train_pi_iters):
            logits = Categorical(logits=self.actor(states)).log_prob(actions)

            ratio = (logits - logits_old).exp()
            clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

            loss = -(torch.min(ratio, clipped) * advantages).mean()
            kl = (logits_old - logits).mean().item()

            if kl > target_kl:  # Early stopping
                break

            self.opt_actor.zero_grad()
            loss.backward()
            self.opt_actor.step()

        for _ in range(train_v_iters):
            loss = ((self.critic(data['states']) - data['discounted']) ** 2).mean()
            self.opt_critic.zero_grad()
            loss.backward()
            self.opt_critic.step()

        self.reset_memory()

        self.actor.to('cpu')
        self.critic.to('cpu')


train(gym.make('CartPole-v0'), PPOAgent, epochs=epochs, num_rollouts=num_rollouts)
