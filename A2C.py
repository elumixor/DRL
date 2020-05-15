import torch
import numpy as np

import gym
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import Linear
from torch.optim import Adam
from torch.autograd import Variable

# Train on balancing a pole on a cart
from utils import running_average, bootstrap

env = gym.make('CartPole-v0')

# See what we are working with
state_shape = env.observation_space.shape
num_actions = env.action_space.n

print(f'Observation shape: {state_shape}. Actions number: {num_actions}')


# We will extend VPG by estimating the Advantage function as
# A(s) = (r + V(s+1)) - V(s)
# So we will need a network approximating V(s)

# Also, we will need to approximate the policy pi(s)

# We will use a network with two heads. One for V(s), and another for pi(s)
class A2C(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_size):
        super().__init__()
        self.l1 = Linear(int(np.prod(input_shape)), hidden_size)
        self.l2 = Linear(hidden_size, hidden_size)

        self.critic = Linear(hidden_size, 1)
        self.actor = Linear(hidden_size, num_actions)

    def forward(self, state):
        x = state
        # x = Variable(torch.from_numpy(state).float().unsqueeze(0))

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        V = self.critic(x)
        pi = F.softmax(self.actor(x), dim=-1)
        return V, pi


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.l1 = Linear(num_inputs, hidden_size)
        self.l2 = Linear(hidden_size, hidden_size)

        self.critic = Linear(hidden_size, 1)
        self.actor = Linear(hidden_size, num_actions)

    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        # value = F.relu(self.critic_linear1(state))
        value = self.critic(x)

        # policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor(x), dim=1)

        return value, policy_dist


net = A2C(state_shape, num_actions, 32)
# net = ActorCritic(np.prod(state_shape), num_actions, 32)

lr = 0.001
discount = 0.99
optimizer = Adam(net.parameters(), lr=lr)


# # Test network's output on a dummy observation
# obs = env.reset()
# obs = torch.tensor(obs).float()
# print(f's = {obs.tolist()}.')
#
# V, pi = net(obs)
# print(f'V(s) = {V.item()}.')
# print(f'pi(s) = {pi.tolist()}.')


# This is an on-policy method, meaning we will train from every experience we collect
# while following the same policy we are trying to learn

# We will sample an action based on the probabilities, outputted by our actor network
def get_action(state):
    state = torch.tensor(state).float()
    _, pi = net(state)
    action = torch.distributions.Categorical(pi).sample().item()
    return action


def rewards_to_go(rewards, disounting=0.99):
    res = torch.zeros(len(rewards))
    last = 0

    for i in reversed(range(len(rewards))):
        last = res[i] = rewards[i] + disounting * last

    return res


def learn(samples):
    # Transpose a batch
    states, actions, rewards, next_states = zip(*samples)

    # Transform to tensors
    states = torch.stack([torch.tensor(s) for s in states]).float()
    actions = torch.tensor(actions).long()
    rewards = torch.tensor(rewards).float()
    next_states = torch.stack([torch.tensor(s) for s in next_states]).float()

    # Calculate advantage: get V(states) and pi(states)
    values, pi = net(states)

    with torch.no_grad():
        # Bootstrap V(states + 1) with rewards
        # Use that to calculate advantage
        values = values.flatten()
        next_value, _ = net(next_states[-1])
        advantage = bootstrap(rewards, next_value) - values

    loss_actor = (-torch.log(pi[range(pi.shape[0]), actions]) * advantage).mean()
    loss_critic = .5 * (advantage ** 2).mean()
    loss = loss_actor + loss_critic

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# And now the main learning loop

episodes = 2000

total_rewards = []

for episode in range(episodes):
    samples = []

    state = env.reset()
    done = False

    total_reward = 0

    with torch.no_grad():
        while not done:
            if episode % 100 < 5:
                env.render()

            action = get_action(state)
            next_state, reward, done, _ = env.step(action)

            samples.append((state, action, reward, next_state))
            state = next_state

            total_reward += reward

    learn(samples)

    total_rewards.append(total_reward)

    if episode % 100 == 0:
        print(f'Ep: {episode} | Mean total reward over last 100 episodes: {np.mean(total_rewards[-100:])}')

        _, ax = plt.subplots()
        ax.plot(total_rewards)
        ax.plot(running_average(total_rewards))
        ax.set_title('Mean total reward')

        plt.show()
