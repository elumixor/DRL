import gym
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

from utils import get_env, train, torch_device, rewards_to_go, ModelSaver, bootstrap, normalize

env, (obs_size,), num_actions = get_env('CartPole-v0')

hidden_actor = 16
hidden_critic = 16

# Actor maps state to actions' probabilities
actor = nn.Sequential(nn.Linear(obs_size, hidden_actor),
                      nn.ReLU(),
                      nn.Linear(hidden_actor, num_actions),
                      nn.Softmax(dim=1))

# Critic maps state to a value, expected total reward from acting optimally, starting in that state
critic = nn.Sequential(nn.Linear(obs_size, hidden_critic),
                       nn.ReLU(),
                       nn.Linear(hidden_critic, 1))

# Optimizers
optim_actor = Adam(actor.parameters(), lr=0.01)
optim_critic = Adam(critic.parameters(), lr=0.005)

discounting = 0.99

# We need to copy optimizers' parameters, due to this issue: https://github.com/pytorch/pytorch/issues/2830
# In case when we first created models with parameters on cpu, and then loaded them with paramters on gpu,
# Optimizer's weights will still be located on cpu. This will cause errors
# for state in list(optim_actor.state.values()) + list(optim_critic.state.values()):
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.to(torch_device)


def get_action(state):
    state = torch.tensor(state).float().unsqueeze(0)
    probabilities = actor(state)
    action = Categorical(probabilities).sample()
    return action.item()


def train_epoch(rollouts):
    actor.train()
    critic.train()

    loss = 0
    total_len = 0

    for states, actions, rewards, next_states in rollouts:
        # 2nd step: use advantage function, estimated by critic
        # bootstrap estimated next state values with rewards TD-1
        values = critic(states)

        last_state = next_states[-1].unsqueeze(0)
        last_value = critic(last_state).item()
        next_values = bootstrap(rewards, last_value, discounting)

        advantage = normalize(next_values - values).flatten()

        loss_critic = .5 * (advantage ** 2).sum()

        # Get probabilities, shape (episode_length * num_actions)
        # Then select only the probabilities corresponding to sampled actions
        probabilities = actor(states)
        probabilities = probabilities[range(states.shape[0]), actions.flatten()]
        loss_actor = (-torch.log(probabilities) * advantage).sum()

        # Take the weighted average (helps convergence)
        loss += loss_critic + loss_actor
        total_len += states.shape[0]

    loss = loss / total_len

    optim_actor.zero_grad()
    optim_critic.zero_grad()
    loss.backward()
    optim_actor.step()
    optim_critic.step()

    actor.eval()
    critic.eval()


# Train, provide an env, function to get an action from state, and training function that accepts rollouts
train(env, get_action, train_epoch, epochs=2000, num_trajectories=10, print_frequency=10, plot_frequency=50)
