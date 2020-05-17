import torch
import numpy as np

import gym
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam

from rnn_utils import running_average, one_hot, Memory

env = gym.make('CartPole-v0')

# See what we are working with
state_shape = env.observation_space.shape
num_actions = env.action_space.n

print(f'Observation shape: {state_shape}. Actions number: {num_actions}')


# Transform action to one-hot encoding
def encode_action(action):
    return one_hot(num_actions, action)


# Create out Q predictor
class QNet(nn.Module):
    def __init__(self, state_shape, num_actions, hidden_size):
        super().__init__()
        self.l1 = nn.Linear(np.prod(state_shape) + num_actions, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


q_net = QNet(state_shape, num_actions, 32)
lr = 0.001
optimizer = Adam(q_net.parameters(), lr=lr)

# Test network's output on a dummy observation
obs = env.reset()
obs = torch.tensor(obs).float()
print(f's = {obs.tolist()}.')
print(f'Q(left |s) = {q_net(obs, encode_action(0)).item()}.')
print(f'Q(right|s) = {q_net(obs, encode_action(1)).item()}.')

batch_size = 200
discount = 0.99

# Create memory buffer to store memories
memory_capacity = 1000
mem = Memory(memory_capacity)

# We will follow epsilon-greedy policy
epsilon = 1
min_epsilon = 0.01
epsilon_history = []  # record decay


def get_action(state):
    # return random action with probability epsilon
    if np.random.random() < epsilon:
        return np.random.choice(np.arange(num_actions))

    # Otherwise return action, associated with max Q value from our predictor
    return np.argmax([q_net(state, encode_action(a)) for a in np.arange(num_actions)])


def experience_replay():
    # Our goal is to minimize the difference between the current state Q estimate
    # and the reward + next state V estimate, which is the max of Q
    # min delta = (Q(s) - (r + max_a Q(s'))

    # We will sample a random batch of experiences
    batch = mem.sample(batch_size)

    # Convert an array of tuples into a tuple of arrays
    states, actions, rewards, next_states = zip(*batch)

    # Convert our data to tensors
    states = torch.stack(states)
    actions = torch.stack([encode_action(a) for a in actions]).float()
    rewards = torch.tensor(rewards)
    next_states = torch.stack(next_states)

    # Calculate next V, which is max over actions for Q for the next state
    possible_actions = [encode_action(a).repeat((states.shape[0], 1)) for a in range(num_actions)]
    q_next = torch.stack([q_net(next_states, pa) for pa in possible_actions], dim=1)
    v_next, _ = torch.max(q_next, dim=1)

    q_current = q_net(states, actions).flatten()
    v_next = v_next.flatten()

    # Smooth l1 loss behaves like L2 near zero, but otherwise it's L1
    loss = F.smooth_l1_loss(q_current, rewards + discount * v_next)

    # Perform training step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


num_episodes = 1000

sum_rewards = []

for episode in range(num_episodes):
    state = torch.tensor(env.reset()).float()
    done = False

    rewards = []

    with torch.no_grad():
        while not done:
            if episode % 100 < 5:
                env.render()

            action = get_action(state)
            next_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            next_state = torch.tensor(next_state).float()

            mem.push((state, action, reward, next_state))

            state = next_state

    if mem.is_full:
        experience_replay()

    epsilon_history.append(epsilon)
    if epsilon > min_epsilon:
        epsilon *= 0.99

    # print(f'Episode {episode}: {sum(rewards)}')
    sum_rewards.append(sum(rewards))

    if episode % 10 == 9:
        fig, (ax0, ax1) = plt.subplots(2)

        ax0.set_title('Total reward for an episode')
        ax0.show(sum_rewards)
        ax0.show(running_average(sum_rewards))

        ax1.set_title('Epsilon')
        ax1.show(epsilon_history)
        plt.show()

    if episode % 100 == 99:
        print(f'Ep {episode + 1} | Mean total reward across last 100 episodes: {np.mean(sum_rewards[-100:])}')
