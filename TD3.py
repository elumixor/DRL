import random
from collections import deque
from copy import deepcopy
from typing import Union

from torch import nn
from torch.optim import Adam

from utils import train, RLAgent, polyak
import gym
import torch

from utils.agents import State

buffer_size = 50000
sample_size = 128

hidden_size = 128
polyak_factor = 0.99
lr_q = 0.001
lr_pi = 0.0001

noise_action = 0.1  # Noise used for sampling an action

# Noise for smooth target updates
noise_target = 0.2
noise_clip = 0.5

discount = 0.99

exploration_steps = 500

# Steps, run with the policy, before an update.
# Will perform this number of gradient steps at every update, evening out the frequency.
update_total_frequency = 2

# How often will the policy and target updated
policy_update_frequency = 2  # Once per two Q updates


class Agent(RLAgent):
    def __init__(self, env):
        super().__init__(env)

        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        self.action_low = torch.from_numpy(env.action_space.low).float()
        self.action_high = torch.from_numpy(env.action_space.high).float()

        self.buffer = deque(maxlen=buffer_size)

        self.action_noise = noise_action

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
            def __init__(self, action_low, action_high):
                super().__init__()
                self.l1 = nn.Linear(obs_size, hidden_size)
                self.l2 = nn.Linear(hidden_size, hidden_size)
                self.l3 = nn.Linear(hidden_size, action_size)
                self.action_low = action_low
                self.action_high = action_high

            def forward(self, state):
                x = self.l1(state).relu()
                x = self.l2(x).relu()
                x = self.l3(x)
                x = x.tanh() * (self.action_high - self.action_low) + self.action_low  # rescale
                return x

        # Instantiate networks
        self.q1 = QNet()
        self.q2 = QNet()
        self.pi = Policy(self.action_low, self.action_high)

        # Create target networks
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q1)
        self.pi_target = deepcopy(self.pi)

        # Create optimizers
        self.params_q = list(self.q1.parameters()) + list(self.q2.parameters())
        self.optim_q = Adam(self.params_q, lr=lr_q)
        self.optim_pi = Adam(self.pi.parameters(), lr=lr_pi)

        # To store starting state
        self.state = None
        self.steps_elapsed = 0

        # To track how many Q updates were performed and check if we need to update policy and targets
        self.updates = 0

    def on_trajectory_started(self, state: State):
        self.state = state

    def save_step(self, action: int, reward: float, next_state: State):
        self.buffer.append((torch.from_numpy(self.state).float(),
                            torch.from_numpy(action).float(),
                            torch.tensor(reward, dtype=torch.float).unsqueeze(0),
                            torch.from_numpy(next_state).float()))

        # Store next state as a current state
        self.state = next_state

        # Update is called at the end of the trajectory, we want to update with given frequency, if the buffer is full
        if len(self.buffer) >= sample_size and self.steps_elapsed % update_total_frequency:
            self.train()

    def get_action(self, state: State) -> Union[int, float]:
        self.steps_elapsed += 1

        # Explore at random at the beginning
        if self.steps_elapsed < exploration_steps:
            return self.env.action_space.sample()

        # Pick an action
        state = torch.from_numpy(state).unsqueeze(0).float()
        action = self.pi(state)

        # Apply zero-mean normal noise
        action = action + self.action_noise * torch.randn_like(action)
        action = torch.clamp(action, self.action_low.item(), self.action_high.item())

        return action.detach().squeeze(0).numpy()

    def train(self):
        for _ in range(update_total_frequency):
            states, actions, rewards, next_states = zip(*random.choices(self.buffer, k=sample_size))

            # Transform data to tensors
            states = torch.stack(states)
            actions = torch.stack(actions)
            rewards = torch.stack(rewards)
            next_states = torch.stack(next_states)

            with torch.no_grad():
                target_actions = self.pi_target(next_states)

                # Apply clipped zero-mean normal noise for smoothing
                noise = torch.clamp(noise_target * torch.randn_like(target_actions), -noise_clip, noise_clip)
                target_actions = torch.clamp(target_actions + noise, self.action_low.item(), self.action_high.item())

                # Pick the minimum out of two
                q_target = torch.min(self.q1_target(next_states, target_actions),
                                     self.q2_target(next_states, target_actions))

                y = rewards + discount * q_target

            loss_q = ((self.q1(states, actions) - y) ** 2).mean() + ((self.q2(states, actions) - y) ** 2).mean()

            # Update q
            self.optim_q.zero_grad()
            loss_q.backward()
            self.optim_q.step()

            # Increment the counter, showing that a Q update has occurred
            self.updates += 1

            # Check if we need to update policies and copy targets
            if self.updates % policy_update_frequency == 0:
                loss_pi = -(self.q1(states, self.pi(states))).mean()

                for p in self.params_q:
                    p.requires_grad = False

                # Update pi
                self.optim_pi.zero_grad()
                loss_pi.backward()
                self.optim_pi.step()

                for p in self.params_q:
                    p.requires_grad = True

                # Update targets
                polyak(self.pi_target, self.pi, polyak_factor)
                polyak(self.q1_target, self.q1, polyak_factor)
                polyak(self.q2_target, self.q2, polyak_factor)

    def evaluate(self):
        self.action_noise = 0


train(gym.make('Pendulum-v0'), Agent, plot_frequency=10, tests=5)
