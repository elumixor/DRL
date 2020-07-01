import math
import torch.nn.functional as F
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

# Will sample random actions for this number of steps at the beginning
exploration_steps = 500

# Entropy regulizer term
entropy = 0.2

# Steps, run with the policy, before an update.
# Will perform this number of gradient steps at every update, evening out the frequency.
update_total_frequency = 2

# Clamping log std
log_std_min = -20
log_std_max = 2


class Agent(RLAgent):
    def __init__(self, env):
        super().__init__(env)

        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        self.action_low = torch.from_numpy(env.action_space.low).float()
        self.action_high = torch.from_numpy(env.action_space.high).float()

        self.buffer = deque(maxlen=buffer_size)

        self.action_noise = noise_action
        self.deterministic = False

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

        # Here our policy outputs sampled action and log probability, aookying tanh squashing
        class Policy(nn.Module):
            def __init__(self, action_low, action_high):
                super().__init__()
                self.l1 = nn.Linear(obs_size, hidden_size)
                self.l2 = nn.Linear(hidden_size, hidden_size)
                self.mu = nn.Linear(hidden_size, action_size)  # Head for the mean
                self.log_std = nn.Linear(hidden_size, action_size)  # Head for log std
                self.action_low = action_low
                self.action_high = action_high

            def forward(self, state, deterministic=False):
                x = self.l1(state).relu()
                x = self.l2(x).relu()
                mu, log_std = self.mu(x), torch.clamp(self.log_std(x), log_std_min, log_std_max)
                std = log_std.exp()
                dist = torch.distributions.Normal(mu, std)

                if deterministic:
                    action = mu
                else:
                    action = dist.rsample()  # rsample uses re-parametrization trick to allow back-propagation

                # Get log probability from Normal distribution and apply correction (arXiv 1801.01290, appendix C)
                # This is also numerically more stable
                log_prob = dist.log_prob(action).sum(axis=-1)
                log_prob -= ((math.log(2) - action - F.softplus(-2 * action)) * 2).sum(axis=-1)

                action = action.tanh() * (self.action_high - self.action_low) + self.action_low  # rescale
                return action, log_prob

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
            for _ in range(update_total_frequency):
                self.train()

    def get_action(self, state: State) -> Union[int, float]:
        self.steps_elapsed += 1

        # Explore at random at the beginning
        if self.steps_elapsed < exploration_steps:
            return self.env.action_space.sample()

        # Pick an action
        state = torch.from_numpy(state).unsqueeze(0).float()
        action, _ = self.pi(state, deterministic=self.deterministic)

        # Apply zero-mean normal noise
        return action.detach().squeeze(0).numpy()

    def train(self):
        states, actions, rewards, next_states = zip(*random.choices(self.buffer, k=sample_size))

        # Transform data to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)

        with torch.no_grad():
            # Output log probabilities to compute entropy
            target_actions, log_prob_actions = self.pi(next_states)

            # Pick the minimum out of two
            q_target = torch.min(self.q1_target(next_states, target_actions),
                                 self.q2_target(next_states, target_actions))

            # Add entropy regularization, everything else remains unchanged
            y = rewards + discount * (q_target - entropy * log_prob_actions)

        loss_q = ((self.q1(states, actions) - y) ** 2).mean() + ((self.q2(states, actions) - y) ** 2).mean()

        # Update q
        self.optim_q.zero_grad()
        loss_q.backward()
        self.optim_q.step()

        # We now update Q and policy at every step, unlike TD3 or DDPG
        for p in self.params_q:
            p.requires_grad = False

        # Update pi
        # Check if we need to update policies and copy targets
        actions, log_prob_actions = self.pi(states)
        loss_pi = (-torch.min(self.q1(states, actions),
                              self.q2(states, actions)) + entropy * log_prob_actions).mean()

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
        self.deterministic = True


train(gym.make('Pendulum-v0'), Agent, plot_frequency=10, tests=5)
