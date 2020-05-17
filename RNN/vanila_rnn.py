import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from RNN.RecurrentCell import RecurrentCell
from utils import form_vocabulary, string2tensor, tensor2string, running_average, torch_device


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.l1 = nn.Linear(input_size, input_size)
        self.r1 = RecurrentCell(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        h = self.r1(x)
        x = self.l2(h).log_softmax(dim=1)
        return x

    def zero_state(self):
        return self.r1.zero_state()


text = "hello top kek vlad"
(data_size, vocab_size), (char2ix, ix2char) = form_vocabulary(text)

generated_texts = 10
generated_length = 100
epochs = 2000
print_every = 25
load_previous = False

max_data_length = 1000
data_length = len(text) if max_data_length < 0 else min(len(text), max_data_length)
text = text[:data_length]
data = string2tensor(text, char2ix)

seq_length = 6
seq_length = min(seq_length, data_length - 1)

# model instance
rnn = RNN(input_size=vocab_size, output_size=vocab_size, hidden_size=512).to(torch_device)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

model_path = './models/vanilla_rnn'
models_dir_path = '/'.join(model_path.split('/')[:-1])

if not os.path.exists(models_dir_path):
    os.makedirs(models_dir_path)

if load_previous and os.path.exists(model_path):
    print('Existing model found. Loading...')
    checkpoint = torch.load(model_path)
    rnn.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def sample(word_tensor, temperature):
    letter_probabilities = (word_tensor / temperature).exp()
    return torch.distributions.Categorical(letter_probabilities).sample()


losses = []

for epoch in range(epochs):
    loss = 0

    for p in range(data_length - seq_length - 1):
        rnn.zero_state()
        x_sequence = data[p:p + seq_length]
        target_sequence = data[p + 1:p + seq_length + 1]

        word = torch.zeros(seq_length, 1, vocab_size)

        for i in range(seq_length):
            x = x_sequence[i]
            target = target_sequence[i]

            res = torch.zeros(1, vocab_size)
            res[0][x[0]] = 1

            out = rnn(res.to(torch_device))
            loss += criterion(out, target.to(torch_device))

            word[i] = out.detach()

    optimizer.zero_grad()
    loss = loss / (data_length - seq_length - 1)
    loss.backward()
    optimizer.step()

    loss = loss.item()
    losses.append(loss)

    if epoch % print_every == 0:
        print(f'Epoch:\t{epoch}\tLast running loss:\t{running_average(losses)[-1]}')

        x = tensor2string(x_sequence, ix2char)
        actual = tensor2string(target_sequence, ix2char)

        for temperature in [0.2, 0.5, 1., 1.2]:
            predicted = tensor2string(sample(word, temperature), ix2char)

            correct = predicted == actual

            print(f'\tt: {temperature}\t{x}'
                  f' -> {predicted} | '
                  f'{f"({actual})" if not correct else "✓"}')

        torch.save({'model': rnn.state_dict(), 'optimizer': optimizer.state_dict()}, model_path)
