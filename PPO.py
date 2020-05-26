import gym
import numpy as np
import torch
import torch.nn as nn
from torch import optim, distributions

from utils import RLAgent, train

# Hyper parameters
actor_hidden = 32
critic_hidden = 32

actor_lr = 0.01
critic_lr = 0.005


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

    @property
    def models(self):
        yield self.actor
        yield self.critic

    @property
    def optimizers(self):
        yield self.opt_actor
        yield self.opt_critic

    def get_action(self, state: np.ndarray) -> int:
        logits = self.actor(torch.from_numpy(state).float().unsqueeze(0))
        dist = distributions.Categorical(logits=logits)
        return dist.sample().item()

    # TODO

    def save_step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        raise NotImplementedError

    def on_trajectory_finished(self) -> None:
        raise NotImplementedError

    def update(self) -> None:
        raise NotImplementedError


train(gym.make('CartPole-v0'), Agent)
