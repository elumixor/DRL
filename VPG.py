import gym
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

from utils import get_env, train, torch_device, rewards_to_go, ModelSaver, bootstrap

env, (obs_size,), num_actions = get_env('CartPole-v0')

hidden_actor = 16
hidden_critic = 16

# Actor maps state to actions' probabilities
actor = nn.Sequential(nn.Linear(obs_size, hidden_actor),
                      nn.ReLU(),
                      nn.Linear(hidden_actor, num_actions),
                      nn.Softmax(dim=1))

# Optimizers
optim_actor = Adam(actor.parameters(), lr=0.001)

discounting = 0.99

saver = ModelSaver({'actor': actor, 'optim_actor': optim_actor}, './models/VPG')
saver.load(ignore_errors=True)

# We need to copy optimizers' parameters, due to this issue: https://github.com/pytorch/pytorch/issues/2830
# In case when we first created models with parameters on cpu, and then loaded them with paramters on gpu,
# Optimizer's weights will still be located on cpu. This will cause errors
for state in list(optim_actor.state.values()):
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(torch_device)


def get_action(state):
    state = torch.tensor(state).float().unsqueeze(0)
    probabilities = actor(state)
    action = Categorical(probabilities).sample()
    return action.item()


def train_epoch(rollouts):
    actor.train()
    actor.to(torch_device)

    loss = 0
    total_len = 0

    for r in rollouts:
        states, actions, rewards, _ = zip(*r)

        # Simplest strategy: use the total reward
        # weights = [sum(rewards)] * len(rewards)

        # Improvement: use discounted rewards to go
        weights = rewards_to_go(rewards, discounting)

        # Convert to tensors
        states = torch.stack([torch.from_numpy(s) for s in states]).float().to(torch_device)
        actions = torch.LongTensor(actions).to(torch_device)
        weights = torch.FloatTensor(weights).to(torch_device)  # Improvement

        # Get probabilities, shape (episode_length * num_actions)
        # Then select only the probabilities corresponding to sampled actions
        probabilities = actor(states)
        probabilities = probabilities[range(states.shape[0]), actions]
        loss_actor = (-torch.log(probabilities) * weights).mean()

        # Take the weighted average (helps convergence)
        loss += loss_actor * len(r)
        total_len += len(r)

    loss = loss / total_len

    optim_actor.zero_grad()
    loss.backward()
    optim_actor.step()

    actor.eval()
    actor.to('cpu')

    saver.save()


# Train, provide an env, function to get an action from state, and training function that accepts rollouts
train(env, get_action, train_epoch, render_frequency=(100, 1), print_frequency=100, epochs=2000, num_trajectories=10)
