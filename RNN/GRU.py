import torch
from torch import nn

from utils import torch_device


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        combined_size = input_size + hidden_size
        self.update = nn.Linear(combined_size, hidden_size)
        self.select = nn.Linear(combined_size, hidden_size)
        self.predict = nn.Linear(combined_size, hidden_size)

        self.h = torch.zeros(1, hidden_size).to(torch_device)

    def forward(self, x):
        combined = torch.cat((x, self.h), dim=1)
        to_update = self.update(combined).sigmoid()
        to_select = self.select(combined).sigmoid()

        updated = self.h * to_update
        new_combined = torch.cat((x, updated), dim=1)

        predictions = self.predict(new_combined).tanh()
        selected = predictions * to_select

        self.h = self.h * (1 - to_select) + selected

        return self.h

    def zero_state(self):
        self.h = torch.zeros_like(self.h).to(torch_device)
