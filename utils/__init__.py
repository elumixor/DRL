import gym
import numpy as np

from .model_saver import ModelSaver
from .plotting import Plotter
from .replay_buffer import ReplayBuffer
from .trainer import train

import torch

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_env(name):
    env = gym.make(name)

    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    return env, obs_shape, num_actions


def rewards_to_go(rewards, discounting=0.99):
    res = [0.] * len(rewards)
    last = 0.
    for i in reversed(range(len(rewards))):
        last = res[i] = rewards[i] + discounting * last

    return res


def running_average(arr, smoothing=0.8):
    size = len(arr)
    res = np.zeros(size)

    if size == 0:
        return res

    res[0] = arr[0]
    for i in range(1, size):
        res[i] = res[i - 1] * smoothing + arr[i] * (1 - smoothing)

    return res


def one_hot(x, size):
    res = torch.zeros(size)
    res[x] = 1
    return res


def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.flatten()
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)


def bootstrap(rewards, last, discounting=0.99):
    res = torch.zeros_like(rewards)
    for i in reversed(range(rewards.shape[0])):
        last = res[i] = rewards[i] + discounting * last
    return res


# Conjugate gradient algorithm (accepts tensors)
def conjugate_gradient(A, b, delta=0., max_iterations=float('inf')):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    i = 0
    while i < max_iterations:
        AVP = A(p)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)  # try without epsilon in the denominator

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new

        i += 1
        r = r - alpha * AVP

        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new
    return x
