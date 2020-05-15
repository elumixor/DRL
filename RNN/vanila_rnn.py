import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from plotting import Plotter
from utils import form_vocabulary, string2tensor, tensor2string


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.hidden_size = hidden_size

        self.Wxh = nn.Linear(input_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)
        self.h = torch.zeros(hidden_size)

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state=None):
        x = self.embedding(input_seq)

        if hidden_state is None:
            self.h = torch.zeros_like(self.h)
        else:
            self.h = hidden_state

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

        out = self.decoder(out)
        return out, self.h

    def predict(self, ix_string):
        sequence_length = ix_string.shape[0]
        res = [None] * sequence_length

        hidden_state = None

        for i in range(sequence_length):
            # forward pass
            sequence = ix_string[i:i + 1]  # selects a single char, but as a sequence.
            output, hidden_state = rnn.forward(sequence, hidden_state)

            # construct categorical distribution and sample a character
            output = F.softmax(torch.squeeze(output), dim=0)
            dist = Categorical(output)
            index = dist.sample().item()

            # print the sampled character
            res[i] = index

        return torch.tensor(res)


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
    hidden_state = None

    input_seq = string2tensor("hell", char2ix)
    target_seq = string2tensor("ello", char2ix)

    # forward pass
    output, hidden_state = rnn.forward(input_seq, hidden_state)

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
    hidden_state = None

    test = 'hell'
    print(f'{test} -> {tensor2string(rnn.predict(string2tensor(test, char2ix)), ix2char)}')

    if epoch % int(epochs / 10) == 0 or epoch == epochs - 1:
        plotter.show()
