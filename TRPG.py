from typing import List

import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.optim import Adam

from utils import conjugate_gradient, train, get_env, torch_device, rewards_to_go, jacobian, hessian, bootstrap

# Build the environment
from utils.trainer import Rollout

env, obs_shape, num_actions = get_env("CartPole-v0")

# We will need an actor to take actions in environment,
# and a critic, for estimating state value, and, therefore, advantage

# Actor takes a state and returns actions' probabilities
actor_hidden = 32
actor = nn.Sequential(nn.Linear(obs_shape[0], actor_hidden),
                      # nn.ReLU(),
                      nn.Linear(actor_hidden, num_actions),
                      nn.Softmax(dim=1))

# Critic takes a state and returns its values
critic_hidden = 32
critic = nn.Sequential(nn.Linear(obs_shape[0], critic_hidden),
                       nn.ReLU(),
                       nn.Linear(critic_hidden, 1),
                       nn.Softmax(dim=1))
critic_optimizer = Adam(critic.parameters(), lr=0.001)


# Critic will be updated to give more accurate advantages
def update_critic(advantages):
    advantages = (advantages - advantages.mean()) / advantages.std()  # Normalize to reduce skewness
    loss = (advantages ** 2).mean()  # MSE
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()


# Actor decides what action to take
def get_action(state: List[float]) -> int:
    state = torch.tensor(state).float().unsqueeze(0)  # Turn it into a batch with a single element
    probabilities = actor(state)
    action = Categorical(probabilities).sample()
    return action.item()


def surrogate_loss(new_probabilities, old_probabilities, advantages):
    return (new_probabilities / old_probabilities * advantages).mean()


def estimate_advantages(states, last_state, rewards):
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0))
    next_values = bootstrap(rewards, last_value, discounting=0.99)
    advantages = -values + next_values
    return advantages


def kl_div(p, q):
    p = p.detach()
    return (p * (p / q).log()).sum()


def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.flatten() for t in g])
    return g


def HVP(df, v, x):
    return flat_grad(df @ v, x, retain_graph=True)


delta = 0.025


def line_search(step, criterion, alpha=0.9, max_iterations=10):
    i = 0
    while not criterion((alpha ** i) * step):
        i += 1


def apply_update(grad_flattened):
    n = 0
    for p in actor.parameters():
        numel = p.numel()
        g = grad_flattened[n:n + numel].view(p.shape)
        p += g
        n += numel


# Our main training function
def update_agent(rollouts: List[Rollout]) -> None:
    states = torch.cat([r.states for r in rollouts], dim=0)
    actions = torch.cat([r.actions for r in rollouts], dim=0)
    rewards = torch.cat([r.rewards for r in rollouts], dim=0)
    next_states = torch.cat([r.next_states for r in rollouts], dim=0)

    # for states, actions, rewards, next_states in rollouts:
    advantages = estimate_advantages(states, next_states[-1], rewards)
    update_critic(advantages)

    probabilities = actor(states)

    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    KL = kl_div(probabilities, probabilities)

    parameters = list(actor.parameters())

    g = flat_grad(L, parameters, retain_graph=True)
    d_kl = flat_grad(KL, parameters, create_graph=True)

    def HVP(x):
        return flat_grad(d_kl @ x, parameters, retain_graph=True)

    search_dir = conjugate_gradient(HVP, g, max_iterations=10)
    max_step = torch.sqrt(2 * delta / (g @ search_dir)) * search_dir

    def criterion(current_step):
        with torch.no_grad():
            apply_update(current_step)
            probabilities_new = actor(states)

            L_new = surrogate_loss(probabilities_new, probabilities, advantages)
            KL_new = kl_div(probabilities, probabilities_new)

            loss_improvement = L_new - L

            if loss_improvement > 0 and KL_new <= delta:
                return True

            apply_update(-current_step)
            return False

    line_search(max_step, criterion)


# Train using our get_action() and update() functions
train(env, get_action, update_agent, num_trajectories=10)
