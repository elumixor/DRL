from collections import namedtuple
from typing import List

import gym
import numpy as np
import torch
import torch.nn as nn
from torch import optim, distributions

from utils import RLAgent, train, torch_device, estimate_advantages, normalize, flatten

Rollout = namedtuple('Rollout', ['states', 'logits', 'actions', 'rewards'])

# Hyper parameters
actor_hidden = 32
critic_hidden = 32

actor_lr = 0.001
critic_lr = 0.005

update_iterations = 5
epsilon = 0.02


def compute_loss(pi_new, pi_old, advantages):
    ratio = torch.exp(pi_new - pi_old)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    return -(torch.min(ratio, clipped) * advantages).mean()


class Agent(RLAgent):
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

        self.rollouts: List[Rollout] = []
        self.rollout_samples = []

        self.starting_state: torch.FloatTensor = None
        self.last_state: torch.FloatTensor = None

        self.last_logits: torch.FloatTensor = None
        self.last_action: torch.LongTensor = None

    @property
    def models(self):
        yield self.actor
        yield self.critic

    @property
    def optimizers(self):
        yield self.opt_actor
        yield self.opt_critic

    def get_action(self, state: np.ndarray) -> int:
        self.last_state = state
        self.last_logits = self.actor(torch.from_numpy(state).float().unsqueeze(0))
        dist = distributions.Categorical(logits=self.last_logits)
        self.last_action = dist.sample()
        return self.last_action.item()

    def save_step(self, reward: float, next_state: np.ndarray) -> None:
        self.rollout_samples.append((self.last_logits, self.last_action, reward, next_state))

    def on_trajectory_started(self, state: np.ndarray) -> None:
        self.starting_state = torch.from_numpy(state).float().to(torch_device)

    def on_trajectory_finished(self) -> None:
        logits, actions, rewards, next_states = zip(*self.rollout_samples)

        states = torch.stack([self.starting_state]
                             + [torch.from_numpy(s).float() for s in next_states]).to(torch_device)

        logits = torch.stack(logits).squeeze(1).to(torch_device)
        actions = torch.as_tensor(actions, device=torch_device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float, device=torch_device).unsqueeze(1)

        self.rollouts.append(Rollout(states, logits, actions, rewards))
        self.rollout_samples = []

    def update_critic(self, advantages):
        loss = .5 * (advantages ** 2).mean()  # MSE
        self.opt_critic.zero_grad()
        loss.backward()
        self.opt_critic.step()

    def update(self) -> None:
        states = torch.cat([r.states[:-1] for r in self.rollouts], dim=0)
        actions = torch.cat([r.actions for r in self.rollouts], dim=0).flatten()
        logits = torch.cat([r.logits for r in self.rollouts], dim=0)

        advantages = []
        for s, _, _, rewards in self.rollouts:
            values = self.critic(s)
            advantages.append(estimate_advantages(values, rewards))  # Append rollout advantages

        advantages = normalize(torch.cat(advantages))

        self.update_critic(advantages)

        advantages = advantages.detach()

        pi_old = distributions.Categorical(logits).log_prob(actions).detach()

        for _ in range(update_iterations):
            logits = self.actor(states)
            pi_new = distributions.Categorical(logits).log_prob(actions)

            loss = compute_loss(pi_new, pi_old, advantages)

            self.opt_actor.zero_grad()
            loss.backward()
            self.opt_actor.step()

        self.rollouts = []


train(gym.make('CartPole-v0'), Agent, num_rollouts=5)
