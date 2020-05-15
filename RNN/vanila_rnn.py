import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from plotting import Plotter
from utils import form_vocabulary, string2tensor, tensor2string


class Recurrent(nn.Module):
    """
    Recurrent layer, that remembers a state
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.hidden_size = output_size

        self.Wxh = nn.Linear(input_size, output_size)
        self.Whh = nn.Linear(output_size, output_size)
        self.h = torch.zeros(output_size)

    def forward(self, x):
        # We need to differentiate between single-item input and a batch.
        # For a batch, we have to process it sequentially: (by each sequence, which
        # in this case is a sequence of length 1, i.e. single character), because
        # the hidden state changes with each processed sequence.
        if len(x.shape) > 2:
            # We receive a tensor batch * sequence length * encoding size
            # We output a tensor batch * sequence length * hidden size
            out = torch.zeros(list(x.shape)[:-1] + [self.hidden_size])

            for i, x1 in enumerate(x):
                # Process each sequence through a memory cell
                res = torch.tanh(self.Whh(self.h) + self.Wxh(x1))
                self.h = res.detach()
                out[i] = res
        else:
            # case with no batch - just a sequence
            out = torch.tanh(self.Whh(self.h) + self.Wxh(x))
            self.h = out.detach()

        return out

    def zero_state(self):
        self.h = torch.zeros_like(self.h)


class RNN(nn.Module):
    """
    Recurrent NN, that has a single Recurrent layer
    """

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(input_size, input_size)
        self.recurrent = Recurrent(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.recurrent(x)
        x = self.decoder(x)
        return x

    def predict(self, ix_string):
        # Forward pass
        output = self(ix_string)

        # construct categorical distribution and sample a character
        output = F.softmax(torch.squeeze(output), dim=-1)
        dist = Categorical(output)
        return dist.sample()

    def zero_state(self):
        self.recurrent.zero_state()


# Form a vocabulary from a string "hello"
(data_size, vocab_size), (char2ix, ix2char) = form_vocabulary("hello")

# model instance
rnn = RNN(input_size=vocab_size, output_size=vocab_size, hidden_size=128)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.002)

plotter = Plotter()

# training loop
epochs = 50
for epoch in range(epochs):
    rnn.zero_state()

    input_seq = string2tensor("hell", char2ix)
    target_seq = string2tensor("ello", char2ix)

    # forward pass
    output = rnn(input_seq)

    # compute loss
    loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
    running_loss = loss.item()

    plotter['loss'] += running_loss

    # compute gradients and take optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Epoch: {0} \t Loss: {1:.8f}".format(epoch, running_loss), end='\t')

    # sample / generate a text sequence after every epoch
    rnn.zero_state()

    test = 'hell'
    print(f'{test} -> {tensor2string(rnn.predict(string2tensor(test, char2ix)), ix2char)}', end='\t')

    # Sample again, but start from 'h', and then feed the output prediction back into the network as next input
    rnn.zero_state()
    out = 'h'
    for _ in range(4):
        print(out, end='')
        out = rnn.predict(torch.tensor([char2ix[out]]))
        out = tensor2string(out, ix2char)
    print(out)

    # Show a graph displaying the current progress
    if epoch % int(epochs / 10) == 0 or epoch == epochs - 1:
        plotter.show()
