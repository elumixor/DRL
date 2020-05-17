import torch
import torch.nn as nn

from rnn_utils import torch_device


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.h = torch.zeros(1, hidden_size).to(torch_device)
        self.c = torch.zeros(1, hidden_size).to(torch_device)

        combined_size = input_size + hidden_size

        self.forget = nn.Linear(combined_size, hidden_size)
        self.ignore = nn.Linear(combined_size, hidden_size)
        self.predict = nn.Linear(combined_size, hidden_size)
        self.select = nn.Linear(combined_size, hidden_size)

    def forward(self, x):
        hx = torch.cat((x, self.h), dim=1)
        forget = self.forget(hx).sigmoid()
        ignore = self.ignore(hx).sigmoid()
        select = self.select(hx).sigmoid()
        predict = self.predict(hx).tanh()

        self.c = self.c * forget + (ignore * predict)
        self.h = self.c.tanh() * select

        return self.h

    def zero_state(self):
        self.h = torch.zeros_like(self.h).to(torch_device)
        self.c = torch.zeros_like(self.c).to(torch_device)
