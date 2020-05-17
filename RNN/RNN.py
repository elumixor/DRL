import itertools
import random

import torch
import torch.nn as nn
from torch.nn import NLLLoss
from torch.optim import Adam

from cells import *
from plotting import Plotter
from rnn_utils import read_file_string, Vocabulary, one_hot, generate_phrase, sample_temperature


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_cells):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        # self.cells = nn.ModuleList([LSTMCell(hidden_size, hidden_size) for _ in range(num_cells)])
        self.cells = nn.ModuleList([GRU(hidden_size, hidden_size) for _ in range(num_cells)])
        # self.c = LSTMCell(hidden_size, hidden_size)
        # self.c = RecurrentCell(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        # self.drop = nn.Dropout(0.05)

    def forward(self, x):
        x = self.l1(x).relu()
        for c in self.cells:
            x = c(x)
        x = self.l2(x)
        x = x.log_softmax(dim=1)
        return x

    def zero_state(self):
        for c in self.cells:
            c.zero_state()
        # self.c.zero_state()


# hyperparameters
max_text_length = 100
seq_length = 20
lr = 0.01
hidden_size = 32
num_lst_layers = 1

# Our text to learn
text = read_file_string("./data/shakespeare.txt")[:1000]
# text = read_file_string("./data/messages.txt", encoding='utf-8')
# text = "hello top kek my name is vladyuha"
voc = Vocabulary(text)

# Text as one-hot encoded tensor of shape (data_length, 1, vocab_size).
# The extra 1 dimension is because PyTorch assumes everything is in batches - we’re just using a batch size of 1 here.
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
data = voc.text2tensor(text, max_length=max_text_length)
data_length = data.shape[0]

# Each time we are going to sample a sequence, and try to make our prediction be the same as
# the sequence, shifted by one character. E.g.: 'hello worl' -> 'ello world'
# Sequence length should roughly correspond to the desired output length (?)
seq_length = min(seq_length, data_length - 1)

# Model, optimizer and loss criterion
model = Model(voc.size, voc.size, hidden_size=hidden_size, num_cells=num_lst_layers).to(torch_device)
optim = Adam(model.parameters(), lr=lr)
criterion = NLLLoss()

saver = ModelSaver({'model': model, 'optim': optim}, './models/lstm-gru')
saver.load(ignore_errors=True)

print(generate_phrase(model, seq_length, voc, "menu "))

model.to(torch_device)


def process_sequence(start, size=seq_length):
    """
    Processes the sequence of text as a tensor (one to one rnn)
    :param start: Index of the start of the sequence in data
    :return: Tuple (loss, probabilities) where loss is the tensor of achieved loss on the sequence, and
    probabilities is a tensor of shape (sequence_length, vocabulary_size) with log probabilities of characters.
    """
    model.zero_state()
    x_sequence = data[start:start + size]
    target_sequence = data[start + 1:start + size + 1]

    probabilities = torch.zeros(size, 1, voc.size)
    loss = 0

    for i in range(size):
        x = x_sequence[i]
        target = target_sequence[i]

        x = one_hot(x, voc.size)

        out = model(x.to(torch_device))
        loss += criterion(out, target.to(torch_device))

        probabilities[i] = out.detach()

    return loss / seq_length, probabilities


# Two strategies are available:
#
# 1) We can either move with this sequence on the whole dataset, taking some small steps, accumulate the loss
# and make a big gradient step.
#
# 2) We can move stochastically: e.g. sample random sequences at various locations in the text, and then either:
# - accumulate gradients and make a step
# - make a step after processing each sequence
def random_data_pointer():
    return round(random.random() * (data_length - seq_length - 1))


def sequential_generator(start=0, step=1):
    """
    Sequentially steps through the data, generates indices starting at 'start' with step 'step'.
    """
    return range(start, data_length - seq_length, step)


def random_generator(size):
    """
    Randomly generates 'size' indices in the data
    """
    for _ in range(size):
        yield random_data_pointer()


def random_start_generator():
    """
    Starts from a random point and goes to the end
    """
    return sequential_generator(random_data_pointer())


def process_text(generator, accumulate=False, print_progress=False):
    total_loss = 0
    losses = []
    words_probabilities = []

    if print_progress:
        generator, y_backup = itertools.tee(generator)
        generator, l = itertools.tee(y_backup)
        total_len = len(list(l))
        i = 0

    for p in generator:
        loss, word_probabilities = process_sequence(p)
        losses.append(loss.item())
        words_probabilities.append(word_probabilities.detach())

        if accumulate:
            total_loss += loss
        else:
            optim.zero_grad()
            loss.backward()
            optim.step()

        if print_progress:
            i += 1
            print(f'{100 * i / total_len:.2f}%')

    if accumulate:
        optim.zero_grad()
        (total_loss / len(losses)).backward()
        optim.step()

    return losses, words_probabilities


# We will generate samples of data
def generate_samples(start=None, size=seq_length, temperatures=None):
    if temperatures is None:
        temperatures = [1.]

    model.eval()
    samples = []

    with torch.no_grad():
        p = random_data_pointer() if start is None else start

        for t in temperatures:
            x_sequence = data[p:p + size]
            target_sequence = data[p + 1:p + size + 1]

            _, probabilities = process_sequence(p, size)
            predicted = sample_temperature(probabilities, t)
            predicted = voc.tensor2string(predicted)

            x_word = voc.tensor2string(x_sequence)
            target_word = voc.tensor2string(target_sequence)

            target_word = f'({target_word})' if target_word != predicted else '✓'
            samples.append(f'{x_word}\t->\t{predicted}\t|\t{target_word}')

    model.train()
    return samples


plotter = Plotter()

epochs = 1000
for epoch in range(epochs):
    losses, words_probabilities = process_text(random_generator(30), accumulate=True, print_progress=False)
    samples = "\n\t".join(generate_samples(start=0, temperatures=[0.2, 0.5, 1, 1.2]))

    mean_loss = sum(losses) / len(losses)
    plotter['loss'] += mean_loss

    print(f'Epoch \t {epoch} \t Loss \t {mean_loss}\n\t'
          # f'{samples}\n'
          f'Generate sentence: {generate_phrase(model, seq_length, voc, "menu ")}')

    if epoch % 5 == 0:
        plotter.show()

    saver.save()
