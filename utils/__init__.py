import random
import warnings

import numpy as np
import torch


def running_average(arr, smoothing=0.8):
    size = len(arr)
    res = np.zeros(size)

    if size == 0:
        return res

    res[0] = arr[0]
    for i in range(1, size):
        res[i] = res[i - 1] * smoothing + arr[i] * (1 - smoothing)

    return res


def one_hot(size):
    res = torch.zeros(size).float()
    res[pos] = 1.
    return res


def bootstrap(rewards, last, discount=0.99):
    values = torch.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        last = values[i] = rewards[i] + discount * last
    return values


class Memory:
    def __init__(self, capacity):
        self.data = []
        self.capacity = capacity
        self.position = 0

    @property
    def size(self):
        return len(self.data)

    @property
    def is_full(self):
        return self.size >= self.capacity

    def push(self, item):
        if not self.is_full:
            self.data.append(item)
        else:
            self.data[self.position] = item

        self.position = (self.position + 1) % self.capacity

    def sample(self, size):
        # If size is less than self.capacity, this will throw an error...
        # Should we train on the same samples multiple times?
        if size >= self.size:
            warnings.warn(f'Trying to sample a batch with size: {size},'
                          f' which is greater than the current stored number of samples: {self.size}.'
                          f' Will return a sample of size {self.size}')
        return random.sample(self.data, min(self.size, size))


def form_vocabulary(text):
    chars = sorted(list(set(text)))
    data_size, vocab_size = len(text), len(chars)

    # char to index and index to char maps
    char2ix = {ch: i for i, ch in enumerate(chars)}
    ix2char = {i: ch for i, ch in enumerate(chars)}
    return (data_size, vocab_size), (char2ix, ix2char)


def string2tensor(s, char2ix):
    s = list(s)
    for i, ch in enumerate(s):
        s[i] = char2ix[ch]
    return torch.tensor(s).unsqueeze(1)


def tensor2string(t, ix2char):
    return ''.join([ix2char[i] for i in t.flatten().tolist()])
