import random

import torch

from .functions import one_hot


class Vocabulary:
    def __init__(self, text):
        self.__chars = sorted(list(set(text)))
        self.size = len(self.__chars)

        # char to index and index to char maps
        self.char2ix = {ch: i for i, ch in enumerate(self.__chars)}
        self.ix2char = {i: ch for i, ch in enumerate(self.__chars)}

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.char2ix[item]
        if isinstance(item, int):
            return self.ix2char[item]
        raise ValueError(f"Should be either integer or string. Got {type(item)}")

    def string2tensor(self, s):
        s = list(s)
        for i, ch in enumerate(s):
            s[i] = self.char2ix[ch]
        return torch.tensor(s).unsqueeze(1)

    def text2tensor(self, text, max_length=-1):
        data_length = len(text) if max_length < 0 else min(len(text), max_length)
        text = text[:data_length]
        data = self.string2tensor(text)
        return data

    def string2one_hot(self, s):
        return self.tensor2one_hot(self.string2tensor(s))

    def tensor2one_hot(self, t):
        return one_hot(t, self.size)

    def tensor2string(self, t):
        return ''.join([self.ix2char[i] for i in t.flatten().tolist()])

    def random_character(self):
        return random.choice(self.__chars)
