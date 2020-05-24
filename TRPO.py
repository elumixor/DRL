import random
from typing import List

import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from torch.distributions import Categorical, kl_divergence, Independent, Normal
from torch.optim import Adam

from utils import conjugate_gradient, train, get_env, torch_device, rewards_to_go, jacobian, hessian, bootstrap, \
    normalize

# Build the environment
from utils.trainer import Rollout

env, obs_shape, num_actions = get_env("CartPole-v0")

# We will need an actor to take actions in environment,
# and a critic, for estimating state value, and, therefore, advantage

# Actor takes a state and returns actions' probabilities
actor_hidden = 32
actor = nn.Sequential(nn.Linear(obs_shape[0], actor_hidden),
                      nn.ReLU(),
                      nn.Linear(actor_hidden, num_actions),
                      nn.Softmax(dim=1))

# Critic takes a state and returns its values
critic_hidden = 32
critic = nn.Sequential(nn.Linear(obs_shape[0], critic_hidden),
                       nn.ReLU(),
                       nn.Linear(critic_hidden, 1))
critic_optimizer = Adam(critic.parameters(), lr=0.005)


# Critic will be updated to give more accurate advantages
def update_critic(advantages):
    loss = .5 * (advantages ** 2).mean()  # MSE
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()


# Actor decides what action to take
def get_action(state: List[float]) -> int:
    state = torch.tensor(state).float().unsqueeze(0)  # Turn it into a batch with a single element
    probs = actor(state)
    if torch.any(torch.isnan(probs)):
        for p in actor.parameters():
            print(p)
    action = Categorical(probs=probs).sample()
    return action.item()


def estimate_advantages(states, last_state, rewards):
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0))
    next_values = bootstrap(rewards, last_value, discounting=0.99)
    advantages = next_values - values
    return advantages


def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g


def HVP(df, v, x):
    return flat_grad(df @ v, x, retain_graph=True)


delta = 0.01
iterations = 10


def line_search(step, criterion, alpha=0.9, max_iterations=10):
    i = 0
    while not criterion((alpha ** i) * step) and i < max_iterations:
        i += 1


def apply_update(grad_flattened):
    n = 0
    for p in actor.parameters():
        numel = p.numel()
        g = grad_flattened[n:n + numel].view(p.shape)
        p.data += g
        n += numel


def surrogate_loss(new_probabilities, old_probabilities, advantages):
    return (new_probabilities / old_probabilities * advantages).mean()


def kl_div(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1).mean()


# Our main training function
def update_agent(rollouts: List[Rollout]) -> None:
    states = torch.cat([r.states for r in rollouts], dim=0)
    actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

    advantages = [estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in rollouts]
    advantages = normalize(torch.cat(advantages, dim=0).flatten())

    update_critic(advantages)

    distribution = actor(states)
    distribution = torch.distributions.utils.clamp_probs(distribution)
    probabilities = distribution[range(distribution.shape[0]), actions]

    # Now we have all the data we need for the algorithm

    # We will calculate the gradient wrt to the new probabilities (surrogate function),
    # so second probabilities should be treated as a constant
    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    KL = kl_div(distribution, distribution)

    inputs = list(actor.parameters())

    g = flat_grad(L, actor.parameters(), retain_graph=True)
    d_kl = flat_grad(KL, inputs, create_graph=True)  # Create graph, because we will call backward() on it (for HVP)

    def HVP(v):
        return flat_grad(d_kl @ v, inputs, retain_graph=True)

    search_dir = conjugate_gradient(HVP, g, max_iterations=iterations)
    max_length = torch.sqrt(2 * delta / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir

    def criterion(step):
        apply_update(step)

        with torch.no_grad():
            distribution_new = actor(states)
            distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
            probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

            L_new = surrogate_loss(probabilities_new, probabilities, advantages)
            KL_new = kl_div(distribution, distribution_new)

        L_improvement = L_new - L

        if L_improvement > 0 and KL_new <= delta:
            return True

        apply_update(-step)
        return False

    line_search(max_step, criterion, max_iterations=10)


# Train using our get_action() and update() functions
train(env, get_action, update_agent, num_trajectories=10, render_frequency=None, print_frequency=10,
      plot_frequency=None, epochs=1000)
