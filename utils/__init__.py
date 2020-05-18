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


def bootstrap(rewards, last, discounting=0.99):
    res = [0.] * len(rewards)
    for i in reversed(range(len(rewards))):
        last = res[i] = rewards[i] + discounting * last
    return res
