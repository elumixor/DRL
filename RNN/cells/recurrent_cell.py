import torch
from torch import nn
from rnn_utils import torch_device


class RecurrentCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.xh2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h = torch.zeros(1, hidden_size).to(torch_device)

    def forward(self, x):
        self.h = self.xh2h(torch.cat((x, self.h), dim=1)).tanh()
        return self.h

    def zero_state(self):
        self.h = torch.zeros_like(self.h)
