from time import sleep

import torch
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from torch.distributions import Categorical
from torch.nn import Linear
from torch.optim import Adam

plt.ion()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PGAgent:
    def __init__(self, state_size, num_actions):
        self.num_actions = num_actions
        self.state_size = state_size

        self.l1 = Linear(state_size, 8)
        self.l2 = Linear(8, num_actions)

        self.l1.float()
        self.l2.float()

        self.optimizer = Adam(list(self.l1.parameters()) + list(self.l2.parameters()), lr=0.001)

    def cpu(self):
        self.l1.cpu()
        self.l2.cpu()

    def cuda(self):
        self.l1.to(device)
        self.l2.to(device)

    def get_probabilities(self, state):
        x = self.l1(state)
        x = F.relu(x)
        x = self.l2(x)
        return Categorical(logits=x)

    def get_action(self, state):
        return self.get_probabilities(state).sample()

    def train(self, states, actions, weights):
        self.optimizer.zero_grad()
        loss = (-self.get_probabilities(states).log_prob(actions) * weights).mean()
        loss.backward()
        self.optimizer.step()


def running_average(arr, smoothing=0.8):
    res = np.zeros(len(arr))
    res[0] = arr[0]

    for i in range(1, len(arr)):
        res[i] = res[i - 1] * smoothing + (1 - smoothing) * arr[i]

    return res


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    print(f"State size: {state_size}, num_actions: {num_actions}")

    agent = PGAgent(state_size, num_actions)

    epochs = 2000
    batch_size = 10
    visualizations = 10

    mean_total_rewards = []
    min_total_rewards = []
    max_total_rewards = []

    # Policy gradient
    for epoch in range(epochs):
        batch_states = []
        batch_actions = []
        batch_weights = []

        total_rewards = []

        agent.cpu()

        # Sample a batch of trajectories (do roll-outs)
        for episode in range(batch_size):
            state = env.reset()
            done = False

            rewards = []

            while not done:
                if epoch % 100 == 0 and episode == 0:
                    env.render()

                action = agent.get_action(torch.FloatTensor(state))
                next_state, reward, done, _ = env.step(action.numpy())

                batch_states.append(state)
                batch_actions.append(action)
                rewards.append(reward)

                state = next_state

            total_reward = sum(rewards)

            batch_weights += [total_reward] * len(rewards)
            total_rewards.append(total_reward)

        agent.cuda()

        agent.train(torch.FloatTensor(batch_states).to(device),
                    torch.FloatTensor(batch_actions).to(device),
                    torch.FloatTensor(batch_weights).to(device))

        mean_total_rewards.append(np.mean(total_rewards))
        min_total_rewards.append(np.min(total_rewards))
        max_total_rewards.append(np.max(total_rewards))

        if epoch % 100 == 0:
            print(f'Epoch {epoch}. Mean total reward: {np.mean(total_rewards[-100:])}')

            plt.plot(mean_total_rewards)
            # plt.plot(min_total_rewards)
            # plt.plot(max_total_rewards)

            plt.plot(running_average(mean_total_rewards, smoothing=0.9))

            plt.draw()
            plt.pause(0.0001)
            plt.clf()

    for t in range(visualizations):
        for episode in range(batch_size):
            env.render()

            state = env.reset()
            done = False
            while not done:
                action = agent.get_action(torch.FloatTensor(state))
                next_state, reward, done, _ = env.step(action.numpy())
                state = next_state
