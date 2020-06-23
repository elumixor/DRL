from collections import deque

import torch
from torch.nn import Sequential, Linear, ReLU, Tanh
import gym
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.optim import Adam


def polyak(target, source, factor=0.99):
    """
    In-place polyak averaging between two networks.
    Copies the parameters from the second network into the first one
    """
    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p_target, p in zip(target.parameters(), source.parameters()):
            p_target.data.mul_(factor)
            p_target.data.add_((1 - factor) * p.data)


class Agent:
    def __init__(self):
        class Rescale(torch.nn.Module):
            def __init__(self, min, max):
                super().__init__()
                self.max = max
                self.delta = max - min

            def forward(self, x):
                return x * self.delta - self.max

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        self.action_low = env.action_space.low[0]
        self.action_high = env.action_space.high[0]

        # We want to output q values for the given state
        self.q = Sequential(Linear(state_size + action_size, q_hidden), ReLU(), Linear(q_hidden, q_hidden), ReLU(),
                            Linear(q_hidden, 1))

        # We want to approximate the best action in the continuous action space for the state
        self.pi = Sequential(Linear(state_size, policy_hidden), ReLU(), Linear(policy_hidden, policy_hidden), ReLU(),
                             Linear(policy_hidden, action_size), Tanh(), Rescale(self.action_low, self.action_high))

        # Create target networks for polyak averaging
        self.q_target = deepcopy(self.q)
        self.pi_target = deepcopy(self.pi)

        # Create optimizers for the networks
        self.optim_q = Adam(self.q.parameters(), lr=lr_q)
        self.optim_pi = Adam(self.pi.parameters(), lr=lr_pi)

        # Set noise factor to facilitate exploration
        self.noise_factor = 0.1

    def get_action(self, state):
        action = self.pi(state)
        return torch.clamp(action + torch.randn_like(action) * self.noise_factor, self.action_low, self.action_high)

    def train(self, training_samples):
        states, actions, rewards, next_states = zip(*training_samples)

        states = torch.stack(states).squeeze(1)
        actions = torch.stack(actions).detach().squeeze(1)
        rewards = torch.as_tensor(rewards).unsqueeze(1)
        next_states = torch.stack(next_states).squeeze(1)

        y = rewards + gamma * self.q_target(torch.cat((next_states, self.pi_target(next_states).detach()), dim=1))
        loss_q = ((self.q(torch.cat((states, actions), dim=1)) - y) ** 2).mean()
        loss_pi = -self.q(torch.cat((states, self.pi(states)), dim=1)).mean()

        self.optim_pi.zero_grad()
        loss_pi.backward()
        self.optim_pi.step()

        self.optim_q.zero_grad()
        loss_q.backward()
        self.optim_q.step()

        polyak(self.pi_target, self.pi)
        polyak(self.q_target, self.q)


def main():
    buffer = deque(maxlen=buffer_size)
    agent = Agent()

    total_steps = 0
    total_rewards = []

    for epoch in range(epochs):
        obs = torch.from_numpy(env.reset()).float().unsqueeze(0)
        done = False
        step = 0
        total_reward = 0

        while not done and step < max_steps:
            if epoch % render_frequency == 0:
                env.render()

            action = agent.get_action(obs)
            new_obs, reward, done, _ = env.step(action.squeeze(0).detach().numpy())
            new_obs = torch.from_numpy(new_obs).float().unsqueeze(0)

            buffer.append((obs, action, reward, new_obs))

            if len(buffer) >= train_sample_size:
                agent.train(random.choices(buffer, k=train_sample_size))

            obs = new_obs

            total_reward += reward
            step += 1
            total_steps += 1

        total_rewards.append(total_reward)

        if epoch % print_frequency == 0:
            print(f'Epoch: {epoch}. Total reward: {total_reward}')

    plt.plot(total_rewards)
    plt.show()


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    epochs = 50
    max_steps = 500

    q_hidden = 128
    policy_hidden = 128

    buffer_size = 50000
    train_sample_size = 128

    gamma = 0.99

    lr_q = 0.001
    lr_pi = 0.0001

    polyak_factor = 0.99

    print_frequency = 1
    render_frequency = 5

    main()

    # Record samples
