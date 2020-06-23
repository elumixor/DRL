import numpy as np

from .model_saver import ModelSaver
from .plotting import Plotter
from .replay_buffer import ReplayBuffer
from .trainer import train
from .agents import RLAgent

import torch

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rewards_to_go(rewards, discounting=0.99):
    res = torch.zeros_like(rewards)
    last = 0.
    for i in reversed(range(rewards.shape[0])):
        last = res[i][0] = rewards[i][0] + discounting * last

    return res


def estimate_advantages(values, rewards):
    """
    Bootstraps rewards with last value and returns the difference with predicted values.
    There should be n rewards and n+1 values (extra one for start/terminal state)
    """
    last_value = values[-1].unsqueeze(0)
    next_values = bootstrap(rewards, last_value, discounting=0.99)
    return next_values - values[:-1]


def flatten(list_of_lists):
    return [item for list in list_of_lists for item in list]


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


def normalize(tensor):
    return (tensor - tensor.mean()) / tensor.std()


# Conjugate gradient algorithm (accepts tensors)
def conjugate_gradient(A, b, delta=0., max_iterations=10):
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


def update_network(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
