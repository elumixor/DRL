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
                      nn.LogSoftmax(dim=1))

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
    logits = actor(state)
    action = Categorical(logits=logits).sample()
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
line_search_accept_ratio = 0.1


def line_search(search_dir, max_step_len, criterion, alpha=0.9, max_iterations=10):
    step_len = max_step_len / alpha

    for i in range(max_iterations):
        step_len *= alpha

        if criterion(step_len * search_dir, step_len):
            return step_len

    return torch.tensor(0.0)

    # i = 0
    # while not criterion((alpha ** i) * step) and i < max_iterations:
    #     i += 1


def apply_update(grad_flattened):
    n = 0
    for p in actor.parameters():
        numel = p.numel()
        g = grad_flattened[n:n + numel].view(p.shape)
        p.data += g
        n += numel


def surrogate_loss(log_action_probs, imp_sample_probs, advantages):
    return torch.mean(torch.exp(log_action_probs - imp_sample_probs) * advantages)
    # return (new_probabilities / old_probabilities * advantages).mean()


def detach_dist(dist):
    if type(dist) is Categorical:
        detached_dist = Categorical(logits=dist.logits.detach())
    elif type(dist) is Independent:
        detached_dist = Normal(loc=dist.mean.detach(), scale=dist.stddev.detach())
        detached_dist = Independent(detached_dist, 1)

    return detached_dist


def kl_div(p, q):
    # p = p.detach()
    # return (p * (p / q).log()).mean()
    dist_1_detached = detach_dist(p)
    mean_kl = torch.mean(kl_divergence(dist_1_detached, q))

    return mean_kl


# Our main training function
def update_agent(rollouts: List[Rollout]) -> None:
    states = torch.cat([r.states for r in rollouts], dim=0)
    actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

    advantages = [estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in rollouts]
    advantages = torch.cat(advantages, dim=0)
    advantages = normalize(advantages).flatten()

    action_dists = Categorical(logits=actor(states))
    log_probabilities = action_dists.log_prob(actions)
    # probabilities = probabilities[range(probabilities.shape[0]), actions]

    # Now we have all the data we need for the algorithm

    L = surrogate_loss(log_probabilities, log_probabilities.detach(), advantages)
    g = flat_grad(L, actor.parameters(), retain_graph=True)

    KL = kl_div(action_dists, action_dists)

    inputs = list(actor.parameters())
    d_kl = flat_grad(KL, inputs, create_graph=True)

    def HVP(v):
        return flat_grad(d_kl @ v, inputs, retain_graph=True)

    search_dir = conjugate_gradient(HVP, g, max_iterations=10)
    max_length = torch.sqrt(2 * delta / (search_dir @ HVP(search_dir)))
    # max_step = max_length * search_dir

    expected_improvement = g @ search_dir

    def criterion(step, beta):
        # apply_update(current_step)
        #
        # with torch.no_grad():
        #     action_dists_new = Categorical(actor(states))
        #     log_probabilities_new = action_dists_new.log_prob(actions)
        #
        #     L_new = surrogate_loss(log_probabilities_new, log_probabilities, advantages)
        #     KL_new = kl_div(action_dists, action_dists_new)
        #
        # L_improvement = L_new - L
        #
        # if L_improvement > 0 and KL_new <= delta:
        #     return True
        #
        # apply_update(-current_step)
        # return False

        apply_update(step)

        with torch.no_grad():
            new_action_dists = Categorical(logits=actor(states))
            new_log_action_probs = new_action_dists.log_prob(actions)

            new_loss = surrogate_loss(new_log_action_probs, log_probabilities, advantages)

            mean_kl = kl_div(action_dists, new_action_dists)

        actual_improvement = new_loss - L
        improvement_ratio = actual_improvement / (expected_improvement * beta)

        apply_update(-step)

        surrogate_cond = improvement_ratio >= line_search_accept_ratio and actual_improvement > 0.0
        kl_cond = mean_kl <= delta

        return surrogate_cond and kl_cond

    step_len = line_search(search_dir, max_length, criterion, max_iterations=10)
    opt_step = step_len * search_dir
    apply_update(opt_step)

    update_critic(advantages)


# Train using our get_action() and update() functions
train(env, get_action, update_agent, num_trajectories=10, render_frequency=10, print_frequency=10,
      plot_frequency=None,
      epochs=1000)
