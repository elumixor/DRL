import random
from collections import deque
from copy import deepcopy
from typing import Union

from torch import nn
from torch.optim import Adam

from utils import train, RLAgent
import gym
import torch

from utils.agents import State

buffer_size = 50000
sample_size = 128

hidden_size = 128
polyak_factor = 0.99
lr_q = 0.001
lr_pi = 0.0001

noise_scale = 0.1
discount = 0.99


class Agent(RLAgent):
    def __init__(self, env):
        super().__init__(env)

        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        action_low = torch.from_numpy(env.action_space.low).float()
        action_high = torch.from_numpy(env.action_space.high).float()

        self.buffer = deque(maxlen=buffer_size)

        # Create two networks: one for the Q value function, another - to select continuous actions given the state

        class QNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(obs_size + action_size, hidden_size)
                self.l2 = nn.Linear(hidden_size, hidden_size)
                self.l3 = nn.Linear(hidden_size, 1)

            def forward(self, state, action):
                x = torch.cat([state, action], dim=1)
                x = self.l1(x).relu()
                x = self.l2(x).relu()
                x = self.l3(x)
                return x

        class Policy(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(obs_size, hidden_size)
                self.l2 = nn.Linear(hidden_size, hidden_size)
                self.l3 = nn.Linear(hidden_size, action_size)

            def forward(self, state):
                x = self.l1(state).relu()
                x = self.l2(x).relu()
                x = self.l3(x)
                x = x.tanh() * (action_high - action_low) + action_low  # rescale
                return x

        # Instantiate networks
        self.q = QNet()
        self.pi = Policy()

        # Create target networks
        self.q_target = deepcopy(self.q)
        self.pi_target = deepcopy(self.pi)

        # Create optimizers
        self.optim_q = Adam(self.q.parameters(), lr=lr_q)
        self.optim_pi = Adam(self.pi.parameters(), lr=lr_pi)

        # To store starting state
        self.state = None

    def on_trajectory_started(self, state: State):
        self.state = state

    def save_step(self, action: int, reward: float, next_state: State):
        self.buffer.append((torch.from_numpy(self.state).float(),
                            torch.from_numpy(action).float(),
                            torch.tensor(reward, dtype=torch.float).unsqueeze(0),
                            torch.from_numpy(next_state).float()))

        # Store next state as a current state
        self.state = next_state

        # We will update after every single step, if the buffer is full
        if len(self.buffer) >= sample_size:
            self.train()

    def get_action(self, state: State) -> Union[int, float]:
        state = torch.from_numpy(state).unsqueeze(0).float()
        action = self.pi(state)
        action = action + noise_scale * torch.randn_like(action)  # Apply zero-mean normal noise
        return action.detach().squeeze(0).numpy()

    def update(self) -> None:
        # Update is called at the end of the trajectory, we want to update at every single step
        # We still need to implement it to prevent NotImplementedError from being thrown
        pass

    def train(self):
        states, actions, rewards, next_states = zip(*random.choices(self.buffer, k=sample_size))

        # Transform data to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)

        y = rewards + discount * self.q_target(next_states, self.pi_target(next_states)).detach()

        loss_q = ((self.q(states, actions) - y) ** 2).mean()
        loss_pi = -(self.q(states, self.pi(states))).mean()

        # Update pi
        self.optim_pi.zero_grad()
        loss_pi.backward()
        self.optim_pi.step()

        # Update q
        self.optim_q.zero_grad()
        loss_q.backward()
        self.optim_q.step()

        # Copy to the traget networks
        with torch.no_grad():
            # Copy to the targets networks
            for p_target, p in zip(self.pi_target.parameters(), self.pi.parameters()):
                p_target.mul_(polyak_factor)
                p_target.add_((1 - polyak_factor) * p)

            for p_target, p in zip(self.q_target.parameters(), self.q.parameters()):
                p_target.mul_(polyak_factor)
                p_target.add_((1 - polyak_factor) * p)


train(gym.make('Pendulum-v0'), Agent, plot_frequency=10)
