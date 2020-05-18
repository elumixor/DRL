import random
import warnings


class ReplayBuffer:
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
